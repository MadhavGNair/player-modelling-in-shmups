import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path


def visualize_cluster_analysis(feature_data, analysis_results, output_dir="analysis_plots"):
    """
    Create visualizations for cluster analysis results including boxplots and significance indicators.

    Parameters:
    feature_data (dict): Dictionary with feature names and cluster values
    analysis_results (dict): Results from analyze_clusters_nonparametric function
    output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Set style
    plt.style.use('seaborn-v0_8-bright')
    sns.set_palette("husl")

    for feature_name, cluster_values in feature_data.items():
        feature_results = analysis_results['feature_comparisons'][feature_name]

        # Create figure with two subplots
        fig = plt.figure(figsize=(12, 6))

        # 1. Boxplot with individual points
        ax1 = fig.add_subplot(121)
        create_boxplot(ax1, cluster_values, feature_name)

        # 2. Violin plot with significance indicators
        ax2 = fig.add_subplot(122)
        create_violin_plot(ax2, cluster_values, feature_results, feature_name)

        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature_name}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create additional statistical summary plot
        create_statistical_summary(feature_name, feature_results, output_dir)


def create_boxplot(ax, cluster_values, feature_name):
    """Create a boxplot with individual points using stripplot."""
    data = []
    labels = []

    for cluster_id, values in cluster_values.items():
        data.extend(values)
        labels.extend([f"{cluster_id}"] * len(values))

    df = pd.DataFrame({
        'Cluster': labels,
        'Value': data
    })

    # Create boxplot
    sns.boxplot(x='Cluster', y='Value', data=df, ax=ax, width=0.5, showfliers=False)

    # Add strip plot with jittered points
    sns.stripplot(x='Cluster', y='Value', data=df, ax=ax,
                  color='0.3', alpha=0.5, size=3, jitter=0.2)

    # Customize plot
    ax.set_title(f"{feature_name}\nDistribution by Cluster", pad=20)
    # ax.set_xlabel("Cluster")
    ax.set_ylabel("Value")

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # Adjust y-axis limits to accommodate all points
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)


def create_violin_plot(ax, cluster_values, feature_results, feature_name):
    """Create a violin plot with significance indicators."""
    data = []
    labels = []

    for cluster_id, values in cluster_values.items():
        data.extend(values)
        labels.extend([f"{cluster_id}"] * len(values))

    df = pd.DataFrame({
        'Cluster': labels,
        'Value': data
    })

    # Create violin plot with reduced width
    sns.violinplot(x='Cluster', y='Value', data=df, ax=ax,
                   inner='box', width=0.7)

    # Add individual points with high transparency and small size
    sns.stripplot(x='Cluster', y='Value', data=df, ax=ax,
                  color='0.3', alpha=0.5, size=2, jitter=0.2)

    # Add significance indicators if available
    if feature_results['significant_difference'] and 'significant_pairs' in feature_results:
        add_significance_indicators(ax, feature_results['significant_pairs'], df['Value'].max())

    # Customize plot
    ax.set_title(f"{feature_name}\nDistribution and Significance", pad=20)
    # ax.set_xlabel("Cluster")
    ax.set_ylabel("Value")

    # Add grid
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)


def add_significance_indicators(ax, significant_pairs, max_val):
    """Add significance indicators (bars and asterisks) to the plot."""
    gap = max_val * 0.05
    level = 0

    for pair in significant_pairs:
        # Extract cluster numbers from pair string (e.g., "0 vs 1")
        pairs = pair['cluster_pair'].split(' vs ')
        c1 = int(pairs[0].split()[-1])
        c2 = int(pairs[1].split()[-1])

        # Calculate position for significance bar
        y = max_val + gap + (level * gap * 2)

        # Draw significance bar
        x1, x2 = c1, c2
        ax.plot([x1, x1, x2, x2], [y - gap / 2, y, y, y - gap / 2], 'k-', linewidth=1)

        # Add significance asterisks
        p_val = pair['p_value']
        if p_val < 0.001:
            sig = '***'
        elif p_val < 0.01:
            sig = '**'
        else:
            sig = '*'

        ax.text((x1 + x2) / 2, y, sig, ha='center', va='bottom')
        level += 1

    # Adjust y-axis limit to show all significance bars
    ax.set_ylim(ax.get_ylim()[0], max_val + gap * (level + 1) * 2)


def create_statistical_summary(feature_name, feature_results, output_dir):
    """Create a summary plot with statistical test results."""
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()

    # Create text summary
    text_content = [
        f"Statistical Summary for {feature_name}",
        "=" * 40,
        f"\nKruskal-Wallis Test:",
        f"H-statistic: {feature_results['kruskal_wallis']['h_statistic']:.3f}",
        f"p-value: {feature_results['kruskal_wallis']['p_value']:.3e}",
        f"\nSignificant Differences: {'Yes' if feature_results['significant_difference'] else 'No'}"
    ]

    if feature_results['significant_difference'] and 'significant_pairs' in feature_results:
        text_content.extend([
            "\nSignificant Pairwise Comparisons (Dunn's test):",
            "-" * 40
        ])
        for pair in feature_results['significant_pairs']:
            text_content.append(f"{pair['cluster_pair']}: p = {pair['p_value']:.3e}")

    # Add cluster statistics
    text_content.extend([
        "\nCluster Statistics:",
        "-" * 40
    ])
    for cluster_id, stats in feature_results['cluster_stats'].items():
        text_content.extend([
            f"\nCluster {cluster_id}:",
            f"  Median: {stats['median']:.3f}",
            f"  Q1: {stats['q1']:.3f}",
            f"  Q3: {stats['q3']:.3f}",
            f"  N: {stats['n']}"
        ])

    # Plot text
    ax.text(0.05, 0.95, '\n'.join(text_content),
            transform=ax.transAxes,
            verticalalignment='top',
            fontfamily='monospace',
            fontsize=10)

    # Remove axes
    ax.axis('off')

    # Save plot
    plt.savefig(f"{output_dir}/{feature_name}_stats_summary.png",
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import pandas as pd
# from pathlib import Path
#
#
# def visualize_cluster_analysis(feature_data, analysis_results, output_dir="analysis_plots"):
#     """
#     Create visualizations for cluster analysis results including boxplots and significance indicators.
#
#     Parameters:
#     feature_data (dict): Dictionary with feature names and cluster values
#     analysis_results (dict): Results from analyze_clusters_nonparametric function
#     output_dir (str): Directory to save the plots
#     """
#     # Create output directory if it doesn't exist
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#
#     # Set style
#     plt.style.use('seaborn-v0_8-dark')
#     sns.set_palette("husl")
#
#     for feature_name, cluster_values in feature_data.items():
#         feature_results = analysis_results['feature_comparisons'][feature_name]
#
#         # Create figure with two subplots
#         fig = plt.figure(figsize=(12, 6))
#
#         # 1. Boxplot with individual points
#         ax1 = fig.add_subplot(121)
#         create_boxplot(ax1, cluster_values, feature_name)
#
#         # 2. Violin plot with significance indicators
#         ax2 = fig.add_subplot(122)
#         create_violin_plot(ax2, cluster_values, feature_results, feature_name)
#
#         # Adjust layout and save
#         plt.tight_layout()
#         plt.savefig(f"{output_dir}/{feature_name}_analysis.png", dpi=300, bbox_inches='tight')
#         plt.close()
#
#         # Create additional statistical summary plot
#         create_statistical_summary(feature_name, feature_results, output_dir)
#
#
# def create_boxplot(ax, cluster_values, feature_name):
#     """Create a boxplot with individual points."""
#     data = []
#     labels = []
#
#     for cluster_id, values in cluster_values.items():
#         data.extend(values)
#         labels.extend([f"Cluster {cluster_id}"] * len(values))
#
#     df = pd.DataFrame({
#         'Cluster': labels,
#         'Value': data
#     })
#
#     # Create boxplot with individual points
#     sns.boxplot(x='Cluster', y='Value', data=df, ax=ax, width=0.5)
#     sns.swarmplot(x='Cluster', y='Value', data=df, ax=ax, color='0.25', alpha=0.5, size=4)
#
#     # Customize plot
#     ax.set_title(f"{feature_name}\nDistribution by Cluster", pad=20)
#     ax.set_xlabel("Cluster")
#     ax.set_ylabel("Value")
#
#     # Add grid for better readability
#     ax.yaxis.grid(True, linestyle='--', alpha=0.7)
#     ax.set_axisbelow(True)
#
#
# def create_violin_plot(ax, cluster_values, feature_results, feature_name):
#     """Create a violin plot with significance indicators."""
#     data = []
#     labels = []
#
#     for cluster_id, values in cluster_values.items():
#         data.extend(values)
#         labels.extend([f"{cluster_id}"] * len(values))
#
#     df = pd.DataFrame({
#         'Cluster': labels,
#         'Value': data
#     })
#
#     # Create violin plot
#     sns.violinplot(x='Cluster', y='Value', data=df, ax=ax, inner='box')
#
#     # Add significance indicators if available
#     if feature_results['significant_difference'] and 'significant_pairs' in feature_results:
#         add_significance_indicators(ax, feature_results['significant_pairs'], df['Value'].max())
#
#     # Customize plot
#     ax.set_title(f"{feature_name}\nDistribution and Significance", pad=20)
#     ax.set_ylabel("Value")
#
#     # Add grid
#     ax.yaxis.grid(True, linestyle='--', alpha=0.7)
#     ax.set_axisbelow(True)
#
#
# def add_significance_indicators(ax, significant_pairs, max_val):
#     """Add significance indicators (bars and asterisks) to the plot."""
#     gap = max_val * 0.05
#     level = 0
#
#     for pair in significant_pairs:
#         # Extract cluster numbers from pair string (e.g., "0 vs 1")
#         pairs = pair['cluster_pair'].split(' vs ')
#         c1 = int(pairs[0].split()[-1])
#         c2 = int(pairs[1].split()[-1])
#
#         # Calculate position for significance bar
#         y = max_val + gap + (level * gap * 2)
#
#         # Draw significance bar
#         x1, x2 = c1, c2
#         ax.plot([x1, x1, x2, x2], [y - gap / 2, y, y, y - gap / 2], 'k-', linewidth=1)
#
#         # Add significance asterisks
#         p_val = pair['p_value']
#         if p_val < 0.001:
#             sig = '***'
#         elif p_val < 0.01:
#             sig = '**'
#         else:
#             sig = '*'
#
#         ax.text((x1 + x2) / 2, y, sig, ha='center', va='bottom')
#         level += 1
#
#     # Adjust y-axis limit to show all significance bars
#     ax.set_ylim(ax.get_ylim()[0], max_val + gap * (level + 1) * 2)
#
#
# def create_statistical_summary(feature_name, feature_results, output_dir):
#     """Create a summary plot with statistical test results."""
#     fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
#
#     # Create text summary
#     text_content = [
#         f"Statistical Summary for {feature_name}",
#         "=" * 40,
#         f"\nKruskal-Wallis Test:",
#         f"H-statistic: {feature_results['kruskal_wallis']['h_statistic']:.3f}",
#         f"p-value: {feature_results['kruskal_wallis']['p_value']:.3e}",
#         f"\nSignificant Differences: {'Yes' if feature_results['significant_difference'] else 'No'}"
#     ]
#
#     if feature_results['significant_difference'] and 'significant_pairs' in feature_results:
#         text_content.extend([
#             "\nSignificant Pairwise Comparisons (Dunn's test):",
#             "-" * 40
#         ])
#         for pair in feature_results['significant_pairs']:
#             text_content.append(f"{pair['cluster_pair']}: p = {pair['p_value']:.3e}")
#
#     # Add cluster statistics
#     text_content.extend([
#         "\nCluster Statistics:",
#         "-" * 40
#     ])
#     for cluster_id, stats in feature_results['cluster_stats'].items():
#         text_content.extend([
#             f"\nCluster {cluster_id}:",
#             f"  Median: {stats['median']:.3f}",
#             f"  Q1: {stats['q1']:.3f}",
#             f"  Q3: {stats['q3']:.3f}",
#             f"  N: {stats['n']}"
#         ])
#
#     # Plot text
#     ax.text(0.05, 0.95, '\n'.join(text_content),
#             transform=ax.transAxes,
#             verticalalignment='top',
#             fontfamily='monospace',
#             fontsize=10)
#
#     # Remove axes
#     ax.axis('off')
#
#     # Save plot
#     plt.savefig(f"{output_dir}/{feature_name}_stats_summary.png",
#                 dpi=300,
#                 bbox_inches='tight',
#                 facecolor='white')
#     plt.close()