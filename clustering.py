import json
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import kruskal
from scipy.spatial import ConvexHull

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def plot_cumulative_explained_variance(df, n_components=10):
    # perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(df)

    # calculate cumulative explained variance ratio
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

    # plot cumulative explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), cumulative_explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Ratio over Number of PCA Components')
    plt.grid()
    plt.show()

    return cumulative_explained_variance


def plot_correlation_matrix(data):
    # compute correlation matrix
    corr_matrix = data.corr()

    # plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


def perform_pca(data, n_components=2, weights=None):
    filenames = list(data.keys())
    features = list(data[filenames[0]].keys())

    # convert features to a Numpy array
    feats = np.array([[data[fname][feat] for feat in features] for fname in filenames])

    # standardize the features
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    # perform weighting if weights are provided
    if weights is not None:
        # normalize the weights
        weights = weights / np.sum(weights)
        feats_scaled = feats_scaled * np.sqrt(weights)

    # perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(feats_scaled)

    # feature importance by component
    # dataset_pca = pd.DataFrame(abs(pca.components_), columns=features, index=['PC_1', 'PC_2'])
    # print(dataset_pca)

    # create dataframe
    pca_df = pd.DataFrame(principal_components, columns=['principal component 1', 'principal component 2'])
    pca_df['filename'] = filenames

    return pca_df


def plot_silhouette_scores(df, max_k=10):
    # prepare the data to be clustered
    feats = df[['principal component 1', 'principal component 2']].values

    silhouette_scores = []

    # compute silhouette scores for k values in range 2 to max_k
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(feats)
        score = silhouette_score(feats, labels)
        silhouette_scores.append(score)

    # plot the silhouette scores
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(2, max_k + 1), y=silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different k Values')
    plt.grid(True)
    plt.show()


def perform_kmeans_clustering(df, k):
    # prepare the data to be clustered
    feats = df[['principal component 1', 'principal component 2']].values

    # perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(feats)
    centroids = kmeans.cluster_centers_

    return df, centroids


def plot_kmeans_clustering(df, centroids, shaded=False):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='principal component 1', y='principal component 2', hue='cluster', palette='viridis',
                    s=100, alpha=0.6, edgecolor='w')

    # highlight centroids
    plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='red', marker='X', label='Centroids')

    if shaded:
        for cluster in df['cluster'].unique():
            cluster_points = df[df['cluster'] == cluster][['principal component 1', 'principal component 2']].values
            if len(cluster_points) >= 3:
                hull = ConvexHull(cluster_points)
                plt.fill(cluster_points[hull.vertices, 0], cluster_points[hull.vertices, 1], alpha=0.2, color=sns.color_palette('viridis', as_cmap=True)(cluster / len(df['cluster'].unique())))

    plt.title('K-Means Clustering with PCA Components')
    plt.legend()
    plt.grid(True)
    plt.show()


def shapiro_wilks_test(df, output_file='processed/normality_test_results_weighted.json'):
    results = {}
    for col in df.columns[1:]:  # skip the first column (filename)
        stat, p_value = stats.shapiro(df[col])
        results[col] = {'statistic': stat, 'p_value': p_value, 'significant': bool(p_value < 0.05)}

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f'Normality test results saved to {output_file}')


def levenes_test(df, output_file='processed/homogeneity_test_results_weighted.json'):
    stat, p_value = stats.levene(*[df[col] for col in df.columns[1:]])  # skip the first column (filename)
    results = {'levene_statistic': stat, 'p_value': p_value, 'significant': bool(p_value < 0.05)}

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f'Homogeneity test results saved to {output_file}')


def merge_clusters(df_1, df_2):
    # ensure the 'filename' columns are of the same type
    df_1['filename'] = df_1['filename'].astype(str)
    df_2['filename'] = df_2['filename'].astype(str)

    # merge the dataframes on the 'filename' column and add the 'cluster' column from df_2 to df_1
    raw_df_with_clusters = df_1.copy()
    raw_df_with_clusters = raw_df_with_clusters.merge(df_2[['filename', 'cluster']], on='filename', how='left')
    return raw_df_with_clusters


def kruskal_wallis_test(df, output_file='processed/kruskal_wallis_test_results_weighted.json', save=False, output_dir="analysis_plots"):
    results = {}
    features = df.columns[1:-1]  # Exclude the first column (filename) and the last column (cluster)
    clusters = df['cluster'].unique()
    feature_data = {}

    # create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for feature in features:
        data_by_cluster = [df[df['cluster'] == cluster][feature] for cluster in clusters]
        stat, p_value = kruskal(*data_by_cluster)
        significant = p_value < 0.05
        results[feature] = {'statistic': stat, 'p_value': p_value, 'significant': significant}

        feature_data[feature] = {f'Cluster {cluster}': df[df['cluster'] == cluster][feature].tolist() for cluster in clusters}

        plt.figure(figsize=(12, 6))

        plt.style.use('seaborn-v0_8-bright')
        sns.set_palette("husl")

        ax1 = plt.subplot(121)
        create_boxplot(ax1, feature_data[feature], feature)

        ax2 = plt.subplot(122)
        create_violin_plot(ax2, feature_data[feature], results[feature], feature)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_analysis_1.png", dpi=300, bbox_inches='tight')
        plt.close()

    if save:
        with open(output_file, 'w') as file:
            json.dump(results, file, indent=4)
        print(f'Kruskal-Wallis test results saved to {output_file}')


def create_boxplot(ax, cluster_values, feature_name):
    data = []
    labels = []

    for cluster_id, values in cluster_values.items():
        data.extend(values)
        labels.extend([f"{cluster_id}"] * len(values))

    df = pd.DataFrame({
        'Cluster': labels,
        'Value': data
    })

    sns.boxplot(x='Cluster', y='Value', data=df, ax=ax, width=0.5, showfliers=False)

    sns.stripplot(x='Cluster', y='Value', data=df, ax=ax, color='0.3', alpha=0.5, size=3, jitter=0.2)

    ax.set_title(f"{feature_name}\nDistribution by Cluster", pad=20)
    ax.set_ylabel("Value")

    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)


def create_violin_plot(ax, cluster_values, feature_results, feature_name):
    data = []
    labels = []

    for cluster_id, values in cluster_values.items():
        data.extend(values)
        labels.extend([f"{cluster_id}"] * len(values))

    df = pd.DataFrame({
        'Cluster': labels,
        'Value': data
    })

    # create violin plot
    sns.violinplot(x='Cluster', y='Value', data=df, ax=ax, inner='box', width=0.7)

    # add individual points with high transparency and small size
    sns.stripplot(x='Cluster', y='Value', data=df, ax=ax, color='0.3', alpha=0.5, size=2, jitter=0.2)

    # add significance indicators if available
    if feature_results['significant']:
        add_significance_indicators(ax, feature_results, df['Value'].max())

    ax.set_title(f"{feature_name}\nDistribution and Significance", pad=20)
    ax.set_ylabel("Value")
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)


def add_significance_indicators(ax, feature_results, max_val):
    gap = max_val * 0.05
    level = 0

    # assuming significant_pairs is a list of tuples (cluster1, cluster2, p_value)
    significant_pairs = [(0, 1, feature_results['p_value'])]

    for c1, c2, p_val in significant_pairs:
        # calculate position for significance bar
        y = max_val + gap + (level * gap * 2)

        # draw significance bar
        ax.plot([c1, c1, c2, c2], [y - gap / 2, y, y, y - gap / 2], 'k-', linewidth=1)

        # add significance asterisks
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        else:
            sig = '*'

        ax.text((c1 + c2) / 2, y, sig, ha='center', va='bottom')
        level += 1

    ax.set_ylim(ax.get_ylim()[0], max_val + gap * (level + 1) * 2)


def main():
    # load the raw data
    with open('processed/features.json', 'r') as file:
        sample_data = json.load(file)

    raw_df = pd.DataFrame(sample_data).T.reset_index()
    raw_df.rename(columns={'index': 'filename'}, inplace=True)

    # PRE-PCA METRICS:
    # plot_correlation_matrix(raw_df[raw_df.columns[1:]])
    # plot_cumulative_explained_variance(raw_df[raw_df.columns[1:]], n_components=10)

    # CONVERT TO PCA:
    weights = [0.8, 0.4, 0.6, 0.5, 0.8, 0.2, 0.8, 0.7, 0.5, 0.5, 0.7]
    pca_df = perform_pca(sample_data, n_components=2, weights=weights)

    # PERFORM OPTIMIZATION:
    # plot_silhouette_scores(pca_df, max_k=10)

    # PERFORM KMEANS CLUSTERING:
    # clustered_df, centroids = perform_kmeans_clustering(pca_df, k=2)
    # plot_kmeans_clustering(clustered_df, centroids, shaded=True)

    # NORMALITY TEST:
    # shapiro_wilks_test(raw_df)

    # HOMOGENEITY TEST:
    # levenes_test(raw_df)

    # STATISTICAL ANALYSIS:
    # merged_df = merge_clusters(raw_df, clustered_df)
    # feature_columns = merged_df.columns[1:12]

    # # apply the weights to the features
    # for i, col in enumerate(feature_columns):
    #     merged_df[col] = merged_df[col] * weights[i]

    # kruskal_wallis_test(merged_df)


if __name__ == '__main__':
    main()

