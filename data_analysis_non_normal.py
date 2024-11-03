import json
import os
from collections import defaultdict
from datetime import datetime

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.stats.multicomp
from kneed import KneeLocator
from matplotlib.gridspec import GridSpec
from scikit_posthocs import posthoc_dunn
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from vizualization import visualize_cluster_analysis


def print_kmeans_stats(assignments, importance, stats):
    print('\n K-MEANS ANALYSIS RESULTS')
    print("\nCluster Assignments:")
    for file, cluster in assignments.items():
        print(f"{file}: Cluster {cluster}")

    print("\nFeature Importance:")
    for feature, importance_score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance_score:.3f}")

    print("\nCluster Statistics:")
    for cluster, cluster_data in stats.items():
        print(f"\n{cluster}:")
        print(f"Size: {cluster_data['size']}")
        print("Files:", cluster_data['files'])


def calculate_cluster_statistics(data_dict, filenames, features, cluster_labels, X_scaled):
    cluster_stats = defaultdict(dict)
    unique_clusters = set(label for label in cluster_labels if label != -1)

    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        cluster_files = np.array(filenames)[cluster_mask]

        for feature in features:
            feature_values = [data_dict[fname][feature] for fname in cluster_files]
            cluster_stats[f'Cluster {cluster_id}'][feature] = {
                'mean': np.mean(feature_values),
                'std': np.std(feature_values),
                'min': np.min(feature_values),
                'max': np.max(feature_values)
            }

        cluster_stats[f'Cluster {cluster_id}']['size'] = len(cluster_files)
        cluster_stats[f'Cluster {cluster_id}']['files'] = list(cluster_files)

        points_scaled = X_scaled[cluster_mask]
        if len(points_scaled) > 1:
            volume = np.prod(np.max(points_scaled, axis=0) - np.min(points_scaled, axis=0))
            cluster_stats[f'Cluster {cluster_id}']['density'] = len(points_scaled) / (volume if volume > 0 else 1)

    return dict(cluster_stats)


def display_pca(data, cluster_dict, shaded=False):
    filenames = list(data.keys())
    features = list(data[filenames[0]].keys())

    # convert to numpy array
    X = np.array([[data[fname][feat] for feat in features] for fname in filenames])

    # standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    principal_df = pd.DataFrame(principal_components, columns=['principal component 1', 'principal component 2'])
    finalDf = pd.concat([principal_df, pd.DataFrame(filenames)], axis=1)
    finalDf.rename(columns={0: 'filename'}, inplace=True)

    finalDf['cluster'] = [cluster_dict[name] for name in filenames]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA')
    cluster_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
                      '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']

    if not shaded:
        for c in set(finalDf['cluster']):
            cluster_data = finalDf.loc[finalDf['cluster'] == c]
            ax.scatter(cluster_data['principal component 1'], cluster_data['principal component 2'], c=cluster_colors[c],
                       label=str(c))
    else:
        for c in set(finalDf['cluster']):
            cluster_data = finalDf.loc[finalDf['cluster'] == c]
            points = cluster_data[['principal component 1', 'principal component 2']].values

            # Plot data points
            ax.scatter(points[:, 0], points[:, 1], c=cluster_colors[c], label=str(c))

            # Create a lighter shade for cluster region
            color_lighter = mcolors.to_rgba(cluster_colors[c], alpha=0.2)

            # Draw convex hull around the points
            if len(points) >= 3:  # ConvexHull requires at least 3 points
                hull = ConvexHull(points)
                vertices = hull.vertices
                hull_points = points[vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1], color=color_lighter)

    ax.legend()

    return plt, finalDf


def load_and_preprocess_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Reorganize data by feature
    feature_data = {}

    for cluster_id, cluster_files in data.items():
        for file_name, features in cluster_files.items():
            for feature_name, value in features.items():
                if feature_name not in feature_data:
                    feature_data[feature_name] = {}
                if cluster_id not in feature_data[feature_name]:
                    feature_data[feature_name][cluster_id] = []
                feature_data[feature_name][cluster_id].append(value)

    return feature_data


def write_tukey_hsd_to_file(feature_name, tukey_result):
    file_name = f"Tukey_HSD_{feature_name}.txt"
    with open(file_name, "w") as file:
        file.write(f"Tukey's HSD Test Results for Feature: {feature_name}\n")
        file.write("=" * 50 + "\n\n")

        # Write summary table with headers
        file.write("{:<15} {:<15} {:<10} {:<10} {:<10}\n".format(
            "Group1", "Group2", "Meandiff", "p-adj", "Reject"
        ))
        file.write("-" * 50 + "\n")

        # Write each result row neatly
        for res in tukey_result.summary().data[1:]:
            file.write("{:<15} {:<15} {:<10.4f} {:<10.4f} {:<10}\n".format(
                res[0], res[1], res[2], res[4], "Yes" if res[5] else "No"
            ))


def test_anova_assumptions(feature_data):
    """
    Perform Shapiro-Wilk test for normality and Levene's test for homogeneity of variances
    on clustered feature data.

    Parameters:
    feature_data (dict): Dictionary with feature names as keys and nested dictionaries containing
                        cluster IDs and their corresponding values

    Returns:
    dict: Dictionary containing test results for each feature
    """
    assumption_results = {}

    for feature_name, cluster_values in feature_data.items():
        # Initialize results dictionary for this feature
        feature_results = {
            'normality_tests': {},
            'homogeneity_test': None,
            'assumptions_met': True
        }

        # Prepare data for tests
        raw_values = []
        cluster_data = []

        for cluster_id, values in cluster_values.items():
            # Store values for Levene's test
            cluster_data.append(np.array(values))

            # Perform Shapiro-Wilk test for each cluster
            if len(values) >= 3:  # Shapiro-Wilk test requires at least 3 samples
                statistic, p_value = stats.shapiro(values)
                feature_results['normality_tests'][f'cluster_{cluster_id}'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'normal_distribution': bool(p_value > 0.05)
                }

                if p_value <= 0.05:
                    feature_results['assumptions_met'] = False
            else:
                feature_results['normality_tests'][f'cluster_{cluster_id}'] = {
                    'error': 'Insufficient samples for Shapiro-Wilk test (n < 3)'
                }
                feature_results['assumptions_met'] = False

        # Perform Levene's test
        if len(cluster_data) > 1:  # Need at least 2 groups for Levene's test
            statistic, p_value = stats.levene(*cluster_data)
            feature_results['homogeneity_test'] = {
                'statistic': statistic,
                'p_value': p_value,
                'equal_variances': bool(p_value > 0.05)
            }

            if p_value <= 0.05:
                feature_results['assumptions_met'] = False
        else:
            feature_results['homogeneity_test'] = {
                'error': 'Insufficient groups for Levene\'s test'
            }
            feature_results['assumptions_met'] = False

        assumption_results[feature_name] = feature_results

    # Create summary of violations
    summary = create_assumption_violation_summary(assumption_results)
    assumption_results['summary'] = summary

    return assumption_results


def create_assumption_violation_summary(results):
    """
    Create a summary of assumption violations across all features.

    Parameters:
    results (dict): Results dictionary from test_anova_assumptions

    Returns:
    dict: Summary of violations
    """
    summary = {
        'total_features': 0,
        'normality_violations': 0,
        'homogeneity_violations': 0,
        'features_with_violations': [],
        'features_meeting_assumptions': []
    }

    for feature_name, feature_results in results.items():
        if feature_name == 'summary':
            continue

        summary['total_features'] += 1

        # Check normality violations
        normality_violated = False
        for cluster_result in feature_results['normality_tests'].values():
            if 'normal_distribution' in cluster_result and not cluster_result['normal_distribution']:
                normality_violated = True
                break

        # Check homogeneity violations
        homogeneity_violated = (
                feature_results['homogeneity_test'] and
                'equal_variances' in feature_results['homogeneity_test'] and
                not feature_results['homogeneity_test']['equal_variances']
        )

        if normality_violated:
            summary['normality_violations'] += 1
        if homogeneity_violated:
            summary['homogeneity_violations'] += 1

        if normality_violated or homogeneity_violated:
            summary['features_with_violations'].append(feature_name)
        else:
            summary['features_meeting_assumptions'].append(feature_name)

    return summary


def analyze_clusters_raw(feature_data):
    feature_comparisons = {}

    for feature_name, cluster_values in feature_data.items():
        # Calculate statistics for each cluster
        cluster_stats = {}
        raw_values = []
        labels = []

        for cluster_id, values in cluster_values.items():
            cluster_stats[cluster_id] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'n': len(values)
            }
            raw_values.append(values)
            labels.extend([cluster_id] * len(values))

        # Perform one-way ANOVA using raw values
        if len(raw_values) > 1:
            f_statistic, p_value = stats.f_oneway(*raw_values)

            feature_comparisons[feature_name] = {
                'cluster_stats': cluster_stats,
                'anova': {
                    'f_statistic': f_statistic,
                    'p_value': p_value
                },
                'significant_difference': bool(p_value < 0.05)
            }

            if p_value < 0.05:
                all_values = np.concatenate(raw_values)
                tukey_result = statsmodels.stats.multicomp.pairwise_tukeyhsd(endog=all_values,
                                                                             groups=labels, alpha=0.05)

                write_tukey_hsd_to_file(feature_name, tukey_result)

    analysis_results = {
        'num_clusters': len(next(iter(feature_data.values()))),
        'feature_comparisons': feature_comparisons
    }

    return analysis_results


def perform_anova_tukey(json_file_path, metric='p_bullet_dmg'):
    # Step 1: Load JSON data
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Step 2: Convert JSON to DataFrame
    records = []
    for cluster, games in data.items():
        for game, metrics in games.items():
            if metric in metrics:
                records.append({
                    'Cluster': cluster,
                    'Game': game,
                    'Metric': metrics[metric]
                })

    df = pd.DataFrame(records)

    # Step 3: Perform ANOVA
    model = sm.formula.ols('Metric ~ C(Cluster)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print("ANOVA Results:\n", anova_table)

    # Step 4: Perform Tukey's HSD test if ANOVA is significant
    if anova_table['PR(>F)'][0] < 0.05:
        tukey = statsmodels.stats.multicomp.pairwise_tukeyhsd(df['Metric'], df['Cluster'])
        print("\nTukey's HSD Results:\n", tukey)
    else:
        print("\nANOVA was not significant; Tukey's HSD not performed.")


# def visualize_cluster_analysis(analysis_results, show_plots=True, save_plots=False):
#     figures = []
#
#     for feature, comparison in analysis_results['feature_comparisons'].items():
#         # Create a new figure for each feature
#         fig, ax = plt.subplots(figsize=(12, 7))
#
#         # Prepare data for plotting
#         cluster_ids = list(comparison['cluster_stats'].keys())
#         means = [stats['mean'] for stats in comparison['cluster_stats'].values()]
#         stds = [stats['std'] for stats in comparison['cluster_stats'].values()]
#         ns = [stats['n'] for stats in comparison['cluster_stats'].values()]
#
#         # Create bar plot with error bars
#         bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5,
#                       color='skyblue', edgecolor='black')
#
#         # Customize the plot
#         ax.set_title(f'{feature} Means Across Clusters', fontsize=15)
#         ax.set_xlabel('Cluster ID', fontsize=12)
#         ax.set_ylabel('Mean Value', fontsize=12)
#         ax.set_xticks(range(len(means)))
#         ax.set_xticklabels(cluster_ids)
#
#         # Add value labels on top of each bar
#         for i, bar in enumerate(bars):
#             height = bar.get_height()
#             ax.text(bar.get_x(), height,
#                     f'Mean: {means[i]:.3f}\n±{stds[i]:.3f}\nn={ns[i]}',
#                     ha='left', va='bottom')
#
#         # Annotate ANOVA results
#         anova_text = (
#             f"ANOVA Results:\n"
#             f"F-statistic: {comparison['anova']['f_statistic']:.3f}\n"
#             f"p-value: {comparison['anova']['p_value']:.4f}\n"
#         )
#         significance_text = "Statistically Significant" if comparison[
#             'significant_difference'] else "Not Statistically Significant"
#
#         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#         if feature in ['p_left', 'p_bullets_fired', 'p_bullets_missed', 'average_bullet_time']:
#             ax.text(0.80, 0.97, anova_text + significance_text,
#                     transform=ax.transAxes, fontsize=10,
#                     verticalalignment='top', bbox=props)
#         elif feature in ['p_right']:
#             ax.text(0.38, 0.97, anova_text + significance_text,
#                     transform=ax.transAxes, fontsize=10,
#                     verticalalignment='top', bbox=props)
#         else:
#             ax.text(0.02, 0.97, anova_text + significance_text,
#                     transform=ax.transAxes, fontsize=10,
#                     verticalalignment='top', bbox=props)
#
#         plt.tight_layout()
#
#         if show_plots:
#             plt.show()
#
#         if save_plots:
#             plt.savefig(f'./figures/{feature}_cluster_analysis.png')
#
#         figures.append(fig)
#
#     return figures


def plot_optimization_results(optimization_results):
    k_range = optimization_results['k_range']
    metrics = optimization_results['metrics']
    optimal = optimization_results['optimal_clusters']

    # Create figure with subplot grid
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Plot 1: Elbow Method (Distortion)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(k_range, metrics['distortions'], 'bo-')
    if optimal['elbow_distortion']:
        ax1.axvline(x=optimal['elbow_distortion'], color='r', linestyle='--',
                    label=f'Elbow at k={optimal["elbow_distortion"]}')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Average Distortion')
    ax1.set_title('Elbow Method: Distortion')
    ax1.legend()

    # Plot 2: Elbow Method (Inertia)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(k_range, metrics['inertias'], 'go-')
    if optimal['elbow_inertia']:
        ax2.axvline(x=optimal['elbow_inertia'], color='r', linestyle='--',
                    label=f'Elbow at k={optimal["elbow_inertia"]}')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Inertia')
    ax2.set_title('Elbow Method: Inertia')
    ax2.legend()

    # Plot 3: Silhouette Score
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(k_range, metrics['silhouette_scores'], 'ro-')
    ax3.axvline(x=optimal['silhouette'], color='g', linestyle='--',
                label=f'Optimal k={optimal["silhouette"]}')
    ax3.set_xlabel('Number of Clusters (k)')
    ax3.set_ylabel('Silhouette Score')
    ax3.set_title('Silhouette Analysis')
    ax3.legend()

    # Plot 4: Calinski-Harabasz Index
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(k_range, metrics['calinski_scores'], 'mo-')
    ax4.axvline(x=optimal['calinski_harabasz'], color='g', linestyle='--',
                label=f'Optimal k={optimal["calinski_harabasz"]}')
    ax4.set_xlabel('Number of Clusters (k)')
    ax4.set_ylabel('Calinski-Harabasz Score')
    ax4.set_title('Calinski-Harabasz Index')
    ax4.legend()

    # Plot 5: Davies-Bouldin Index
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(k_range, metrics['davies_scores'], 'co-')
    ax5.axvline(x=optimal['davies_bouldin'], color='g', linestyle='--',
                label=f'Optimal k={optimal["davies_bouldin"]}')
    ax5.set_xlabel('Number of Clusters (k)')
    ax5.set_ylabel('Davies-Bouldin Score')
    ax5.set_title('Davies-Bouldin Index')
    ax5.legend()

    plt.tight_layout()
    return fig


def get_elbow_summary(optimization_results):
    """
    Provides a text summary of the elbow analysis results.
    """
    optimal = optimization_results['optimal_clusters']

    summary = []
    if optimal['elbow_distortion']:
        summary.append(f"Distortion-based elbow point: k = {optimal['elbow_distortion']}")
    if optimal['elbow_inertia']:
        summary.append(f"Inertia-based elbow point: k = {optimal['elbow_inertia']}")

    if optimal['elbow_distortion'] == optimal['elbow_inertia']:
        summary.append("\nBoth methods suggest the same optimal number of clusters, "
                       "providing strong evidence for this choice.")
    else:
        summary.append("\nThe methods suggest different optimal points. Consider:"
                       "\n- Examining the plots visually"
                       "\n- Using additional metrics (Silhouette, etc.)"
                       "\n- Domain knowledge about expected number of clusters")

    return "\n".join(summary)


def plot_elbow(optimization_results, figsize=(12, 6)):
    """
    Creates a publication-quality elbow plot with automatic elbow point detection.
    Plots both distortion and inertia metrics side by side.
    """

    k_range = optimization_results['k_range']
    metrics = optimization_results['metrics']
    optimal = optimization_results['optimal_clusters']

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Distortion
    ax1.plot(k_range, metrics['distortions'], 'bo-', linewidth=2, markersize=8)
    if optimal['elbow_distortion']:
        ax1.axvline(x=optimal['elbow_distortion'], color='r', linestyle='--',
                    label=f'Elbow at k={optimal["elbow_distortion"]}')
        # Add marker at elbow point
        elbow_idx = k_range.index(optimal['elbow_distortion'])
        ax1.plot(optimal['elbow_distortion'], metrics['distortions'][elbow_idx],
                 'rx', markersize=12, markeredgewidth=3)

    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Average Distortion', fontsize=12)
    ax1.set_title('Elbow Method: Distortion', fontsize=14, pad=20)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    # Plot 2: Inertia
    ax2.plot(k_range, metrics['inertias'], 'go-', linewidth=2, markersize=8)
    if optimal['elbow_inertia']:
        ax2.axvline(x=optimal['elbow_inertia'], color='r', linestyle='--',
                    label=f'Elbow at k={optimal["elbow_inertia"]}')
        # Add marker at elbow point
        elbow_idx = k_range.index(optimal['elbow_inertia'])
        ax2.plot(optimal['elbow_inertia'], metrics['inertias'][elbow_idx],
                 'rx', markersize=12, markeredgewidth=3)

    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Inertia', fontsize=12)
    ax2.set_title('Elbow Method: Inertia', fontsize=14, pad=20)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    return fig


def analyze_clusters_nonparametric(feature_data, output_dir="analysis_results"):
    """
    Analyze cluster differences using non-parametric tests (Kruskal-Wallis and Dunn's test)
    for non-normal distributions.

    Parameters:
    feature_data (dict): Dictionary with feature names as keys and nested dictionaries
                        containing cluster IDs and their corresponding values
    output_dir (str): Directory to save the analysis results

    Returns:
    dict: Analysis results including test statistics and cluster statistics
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    feature_comparisons = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for feature_name, cluster_values in feature_data.items():
        # Calculate statistics for each cluster
        cluster_stats = {}
        raw_values = []
        labels = []

        # Prepare data for analysis
        for cluster_id, values in cluster_values.items():
            # Calculate descriptive statistics
            cluster_stats[cluster_id] = {
                'median': np.median(values),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            }
            raw_values.extend(values)
            labels.extend([cluster_id] * len(values))

        # Convert to numpy arrays for analysis
        raw_values = np.array(raw_values)
        labels = np.array(labels)

        # Perform Kruskal-Wallis H-test
        h_statistic, p_value = stats.kruskal(*[
            cluster_values[cluster_id] for cluster_id in cluster_values.keys()
        ])

        # Store results
        feature_results = {
            'cluster_stats': cluster_stats,
            'kruskal_wallis': {
                'h_statistic': h_statistic,
                'p_value': p_value
            },
            'significant_difference': bool(p_value < 0.05)
        }

        # If significant difference found, perform Dunn's test
        if p_value < 0.05:
            # Create DataFrame for Dunn's test
            df = pd.DataFrame({
                'values': raw_values,
                'groups': labels
            })

            # Perform Dunn's test
            dunn_result = posthoc_dunn(
                df,
                val_col='values',
                group_col='groups',
                p_adjust='bonferroni'
            )

            # Store Dunn's test results
            feature_results['dunns_test'] = {
                'p_values': dunn_result.to_dict()
            }

            # Generate pairwise comparison summary
            significant_pairs = []
            for idx, row in dunn_result.iterrows():
                for col in dunn_result.columns:
                    if idx < col:  # Avoid duplicate comparisons
                        p_val = dunn_result.loc[idx, col]
                        if p_val < 0.05:
                            significant_pairs.append({
                                'cluster_pair': f'{idx} vs {col}',
                                'p_value': p_val
                            })

            feature_results['significant_pairs'] = significant_pairs

        feature_comparisons[feature_name] = feature_results

        # Save detailed results to file
        save_detailed_results(
            feature_name,
            feature_results,
            output_dir,
            timestamp
        )

    # Create and save summary report
    summary_report = create_summary_report(feature_comparisons)
    save_summary_report(summary_report, output_dir, timestamp)

    return {
        'num_clusters': len(next(iter(feature_data.values()))),
        'feature_comparisons': feature_comparisons,
        'summary': summary_report
    }


def save_detailed_results(feature_name, results, output_dir, timestamp):
    """Save detailed analysis results for each feature."""
    filename = f"{output_dir}/detailed_{feature_name}_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write(f"Analysis Results for {feature_name}\n")
        f.write("=" * 50 + "\n\n")

        # Write cluster statistics
        f.write("Cluster Statistics:\n")
        f.write("-" * 20 + "\n")
        for cluster_id, stats in results['cluster_stats'].items():
            f.write(f"\nCluster {cluster_id}:\n")
            f.write(f"  Median: {stats['median']:.3f}\n")
            f.write(f"  Q1: {stats['q1']:.3f}\n")
            f.write(f"  Q3: {stats['q3']:.3f}\n")
            f.write(f"  Min: {stats['min']:.3f}\n")
            f.write(f"  Max: {stats['max']:.3f}\n")
            f.write(f"  N: {stats['n']}\n")

        # Write Kruskal-Wallis results
        f.write("\nKruskal-Wallis Test Results:\n")
        f.write("-" * 20 + "\n")
        f.write(f"H-statistic: {results['kruskal_wallis']['h_statistic']:.3f}\n")
        f.write(f"p-value: {results['kruskal_wallis']['p_value']:.3e}\n")

        # Write Dunn's test results if available
        if 'dunns_test' in results:
            f.write("\nDunn's Test Results:\n")
            f.write("-" * 20 + "\n")
            f.write("Significant Pairwise Comparisons:\n")
            for pair in results['significant_pairs']:
                f.write(f"  {pair['cluster_pair']}: p = {pair['p_value']:.3e}\n")


def create_summary_report(feature_comparisons):
    """Create a summary report of all analyses."""
    summary = {
        'total_features': len(feature_comparisons),
        'significant_features': 0,
        'feature_summary': {}
    }

    for feature_name, results in feature_comparisons.items():
        feature_summary = {
            'significant': results['significant_difference'],
            'p_value': results['kruskal_wallis']['p_value'],
            'num_significant_pairs': len(results.get('significant_pairs', []))
        }

        if results['significant_difference']:
            summary['significant_features'] += 1

        summary['feature_summary'][feature_name] = feature_summary

    return summary


def save_summary_report(summary, output_dir, timestamp):
    """Save the summary report to a file."""
    filename = f"{output_dir}/summary_report_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write("Analysis Summary Report\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total features analyzed: {summary['total_features']}\n")
        f.write(f"Features with significant differences: {summary['significant_features']}\n\n")

        f.write("Feature-by-Feature Summary:\n")
        f.write("-" * 20 + "\n")

        for feature_name, results in summary['feature_summary'].items():
            f.write(f"\n{feature_name}:\n")
            f.write(f"  Significant differences: {'Yes' if results['significant'] else 'No'}\n")
            if results['significant']:
                f.write(f"  p-value: {results['p_value']:.3e}\n")
                f.write(f"  Number of significant pairs: {results['num_significant_pairs']}\n")


class Clustering:
    def __init__(self, data):
        self.data = data

    def optimize_clusters(self, max_clusters=10, random_state=42):
        # Prepare data
        filenames = list(self.data.keys())
        features = list(self.data[filenames[0]].keys())
        X = np.array([[self.data[fname][feat] for feat in features] for fname in filenames])

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize metrics storage
        results = {
            'distortions': [],
            'inertias': [],
            'silhouette_scores': [],
            'calinski_scores': [],
            'davies_scores': []
        }

        # Calculate metrics for different numbers of clusters
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            labels = kmeans.fit_predict(X_scaled)

            # Calculate various metrics
            results['distortions'].append(float(sum(np.min(cdist(X_scaled, kmeans.cluster_centers_,
                                                           'euclidean'), axis=1)) / X_scaled.shape[0]))
            results['inertias'].append(float(kmeans.inertia_))
            results['silhouette_scores'].append(float(silhouette_score(X_scaled, labels)))
            results['calinski_scores'].append(float(calinski_harabasz_score(X_scaled, labels)))
            results['davies_scores'].append(float(davies_bouldin_score(X_scaled, labels)))

        # Calculate optimal k using the elbow method
        k_range = range(2, max_clusters + 1)

        # Find elbow points
        elbow_distortion = KneeLocator(k_range, results['distortions'], curve='convex',
                                       direction='decreasing').knee
        elbow_inertia = KneeLocator(k_range, results['inertias'], curve='convex',
                                    direction='decreasing').knee

        # Find optimal k using silhouette score
        optimal_k_silhouette = k_range[np.argmax(results['silhouette_scores'])]

        # Find optimal k using Calinski-Harabasz Index
        optimal_k_calinski = k_range[np.argmax(results['calinski_scores'])]

        # Find optimal k using Davies-Bouldin Index
        optimal_k_davies = k_range[np.argmin(results['davies_scores'])]

        # Compile results
        optimal_clusters = {
            'elbow_distortion': int(elbow_distortion),
            'elbow_inertia': int(elbow_inertia),
            'silhouette': int(optimal_k_silhouette),
            'calinski_harabasz': int(optimal_k_calinski),
            'davies_bouldin': int(optimal_k_davies)
        }

        return {
            'metrics': results,
            'optimal_clusters': optimal_clusters,
            'k_range': list(k_range)
        }

    def kmeans_clustering(self, n_clusters=3, random_state=42):
        # split filenames and features
        filenames = list(self.data.keys())
        features = list(self.data[filenames[0]].keys())

        # convert features to numpy array
        X = np.array([[self.data[fname][feat] for feat in features] for fname in filenames])

        # standardize features since units are different
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X_scaled)
        cluster_assignments = {fname: label for fname, label in zip(filenames, cluster_labels)}

        # compute importance of features
        feature_importance = {}
        centroids_scaled = kmeans.cluster_centers_
        centroids = scaler.inverse_transform(centroids_scaled)

        for i, feature in enumerate(features):
            # calculate the spread of feature values across centroids
            spread = np.std(centroids[:, i])
            feature_importance[feature] = spread

        # normalize feature importance
        max_importance = max(feature_importance.values())
        feature_importance = {k: v / max_importance for k, v in feature_importance.items()}

        # calculate cluster statistics
        cluster_stats = defaultdict(dict)
        for cluster_id in range(n_clusters):
            # get indices for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_files = np.array(filenames)[cluster_mask]

            # calculate statistics for each feature in this cluster
            for feature in features:
                feature_values = [self.data[fname][feature] for fname in cluster_files]
                cluster_stats[f'Cluster {cluster_id}'][feature] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values)
                }

            cluster_stats[f'Cluster {cluster_id}']['size'] = len(cluster_files)
            cluster_stats[f'Cluster {cluster_id}']['files'] = list(cluster_files)

        return cluster_assignments, feature_importance, dict(cluster_stats)


def find_optimal_k(X, k_range):
    """
    Find optimal k using silhouette scores.

    Parameters:
    X (numpy.ndarray): Input data
    k_range (list): Range of k values to test

    Returns:
    tuple: (optimal k value, dictionary of silhouette scores)
    """
    silhouette_scores = {}

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, cluster_labels)
        silhouette_scores[k] = silhouette_avg

    optimal_k = max(silhouette_scores.items(), key=lambda x: x[1])[0]
    return optimal_k, silhouette_scores


def plot_silhouette_scores(silhouette_scores, optimal_k):
    """
    Plot silhouette scores for different k values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()),
             marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.legend()
    plt.show()


def pca_analysis(df):
    df = df.drop(['cluster'], axis=1)

    X = df[['principal component 1', 'principal component 2']].values

    k_range = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Find optimal k
    optimal_k, silhouette_scores = find_optimal_k(X, k_range)

    # Plot silhouette scores
    # plot_silhouette_scores(silhouette_scores, optimal_k)

    # Perform final clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X)

    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['Cluster'] = cluster_labels

    # Create final clustering visualization
    # plt.figure(figsize=(10, 6))
    #
    # # Create a color map
    # colors = ['#4363d8', '#3cb44b', '#ffe119', '#e6194b', '#f58231', '#911eb4',
    #                   '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
    #
    # # Create mesh grid for decision boundary
    # x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    # y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    # xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
    #                      np.linspace(y_min, y_max, 100))
    #
    # # Predict labels for all points in the mesh
    # Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
    # Z = Z.reshape(xx.shape)
    #
    # # Plot the decision boundary for each cluster
    # for i in range(optimal_k):
    #     # Create lighter version of the color for shading
    #     # color_rgba = to_rgba(colors[i], 0.3)  # Increased alpha to 0.3 for better visibility
    #
    #     # Plot shaded region
    #     # plt.contourf(xx, yy, Z == i, colors=[color_rgba], alpha=0.3)
    #
    #     # Plot points
    #     mask = cluster_labels == i
    #     plt.scatter(X[mask, 0], X[mask, 1], c=colors[i], label=f'Cluster {i}', zorder=2)
    #     # color_lighter = mcolors.to_rgba(colors[i], alpha=0.2)
    #     #
    #     # # Draw convex hull around the points
    #     # if len(X) >= 3:  # ConvexHull requires at least 3 points
    #     #     hull = ConvexHull(X)
    #     #     vertices = hull.vertices
    #     #     hull_points = X[vertices]
    #     #     plt.fill(hull_points[:, 0], hull_points[:, 1], color=color_lighter)
    #
    # # Plot cluster centers
    # centers = kmeans.cluster_centers_
    # plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100,
    #             linewidth=2, label='Cluster Centers', zorder=3)
    #
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.title(f'K-means Clustering of PCA Results (k={optimal_k})')
    # plt.legend()
    #
    # # Set the plot limits
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)
    #
    # plt.show()
    #
    best_silhouette = silhouette_scores[optimal_k]
    # print(f"Optimal number of clusters: {optimal_k}")
    # print(f"Best silhouette score: {best_silhouette:.3f}")

    return df_with_clusters, kmeans, optimal_k, best_silhouette


if __name__ == "__main__":
    with open('features.json', 'r') as file:
        sample_data = json.load(file)

    model = Clustering(sample_data)

    # K-MEANS CLUSTERING:
    k_assignments, k_importance, k_stats = model.kmeans_clustering(n_clusters=3)
    # print_kmeans_stats(k_assignments, k_importance, k_stats)

    # K-MEANS OPTIMIZATION:
    # results = model.optimize_clusters(max_clusters=10)
    #
    # # Create and display the elbow plot
    # elbow_plot = plot_elbow(results)
    # plt.show()
    #
    # # Get a text summary of the results
    # summary = get_elbow_summary(results)
    # print(summary)

    # PCA PLOTTING:
    c_dict = {}
    for file, cluster in k_assignments.items():
        c_dict[file] = int(cluster)

    plot, dataframe = display_pca(sample_data, c_dict, shaded=False)
    # plot.show()

    clusters = {
        'Cluster 0': {},
        'Cluster 1': {},
        'Cluster 2': {}
    }

    df_with_clusters, kmeans, optimal_k, best_silhouette = pca_analysis(dataframe)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA')
    cluster_colors = ['#4363d8', '#3cb44b', '#ffe119', '#e6194b', '#f58231', '#911eb4',
                      '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=100,
                linewidth=2, label='Cluster Centers', zorder=3)
    shaded = True
    if not shaded:
        for c in set(df_with_clusters['Cluster']):
            cluster_data = df_with_clusters.loc[df_with_clusters['Cluster'] == c]
            ax.scatter(cluster_data['principal component 1'], cluster_data['principal component 2'], c=cluster_colors[c],
                       label=str(c))
    else:
        for c in set(df_with_clusters['Cluster']):
            cluster_data = df_with_clusters.loc[df_with_clusters['Cluster'] == c]
            points = cluster_data[['principal component 1', 'principal component 2']].values

            # Plot data points
            ax.scatter(points[:, 0], points[:, 1], c=cluster_colors[c], label=str(c))

            # Create a lighter shade for cluster region
            color_lighter = mcolors.to_rgba(cluster_colors[c], alpha=0.2)

            # Draw convex hull around the points
            if len(points) >= 3:  # ConvexHull requires at least 3 points
                hull = ConvexHull(points)
                vertices = hull.vertices
                hull_points = points[vertices]
                ax.fill(hull_points[:, 0], hull_points[:, 1], color=color_lighter)

    ax.legend()

    plt.show()
    # print(f'\nPCA:\n{df_with_clusters.head()}')

    # NON-PARAMETRIC ANALYSIS:
    # dataframe = df_with_clusters.drop(['principal component 1', 'principal component 2'], axis=1)
    #
    # # Iterate through sample_data and add to the corresponding cluster
    # for filename, data in sample_data.items():
    #     # Find the class of the current filename in file_data
    #     cluster_class = df_with_clusters.loc[dataframe['filename'] == filename, 'Cluster'].values
    #     if cluster_class.size > 0:  # Check if filename was found in file_data
    #         cluster_key = f'Cluster {cluster_class[0]}'
    #         clusters[cluster_key][filename] = data
    #
    # with open('clusters_2.json', 'w') as file:
    #     json.dump(clusters, file)
    #
    # feature_data = load_and_preprocess_data('clusters_2.json')
    # # anova_assumptions = test_anova_assumptions(feature_data)
    # results = analyze_clusters_nonparametric(feature_data)
    # visualize_cluster_analysis(feature_data, results)
    # # analysis_results = analyze_clusters_raw(feature_data)
    # # figures = visualize_cluster_analysis(analysis_results)

    # TUKEY ANALYSIS:
    # items = ['average_bullet_time', 'average_healing_percentage', 'p_accelerated', 'p_bullet_dmg',
    #          'p_bullets_fired', 'p_bullets_missed', 'p_collision_dmg', 'p_left', 'p_right', 'p_water_dmg']
    #
    # for i in items:
    #     perform_anova_tukey('clusters_3.json', metric=i)




