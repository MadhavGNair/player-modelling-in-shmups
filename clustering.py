import json
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import scipy.stats as stats
from scipy.stats import kruskal
from scipy.spatial import ConvexHull

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def plot_cumulative_explained_variance(df, n_components=10):
    """
    Plot the cumulative explained variance ratio over the number of PCA components
    :param df: the dataframe to perform PCA on
    :param n_components: number of PCA components
    :return: the cumulative explained variance ratio list
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca.fit(df)

    # Calculate cumulative explained variance ratio
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()

    # Plot cumulative explained variance ratio
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, n_components + 1), cumulative_explained_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance Ratio over Number of PCA Components')
    plt.grid()
    plt.show()

    return cumulative_explained_variance


def plot_correlation_matrix(data):
    """
    Plot the correlation matrix of the features
    :param data: dataframe of features
    :return: the correlation matrix plot
    """
    # Compute correlation matrix
    corr_matrix = data.corr()

    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


def perform_pca(data, n_components=2, weights=None):
    """
    Perform PCA on the data
    :param data: the dataframe of features
    :param n_components: number of PCA components
    :param weights: None or array of weights for each feature
    :return: dataframe of PCA components
    """
    filenames = list(data.keys())
    features = list(data[filenames[0]].keys())

    # Convert features to a Numpy array
    feats = np.array([[data[fname][feat] for feat in features] for fname in filenames])

    # Standardize the features
    scaler = StandardScaler()
    feats_scaled = scaler.fit_transform(feats)

    # Perform weighting if weights are provided
    if weights is not None:
        # Normalize the weights
        weights = weights / np.sum(weights)
        feats_scaled = feats_scaled * np.sqrt(weights)

    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(feats_scaled)

    # Calculate feature importance by component
    # dataset_pca = pd.DataFrame(abs(pca.components_), columns=features, index=['PC_1', 'PC_2'])
    # print(dataset_pca)

    # Create dataframe
    pca_columns = [f'principal component {i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(principal_components, columns=pca_columns)
    pca_df['filename'] = filenames

    return pca_df


def plot_silhouette_scores(df, max_k=10):
    """
    Plot silhouette scores for different k values
    :param df: the dataframe of PCA components
    :param max_k: maximum number of clusters to test
    :return: the plot of silhouette scores
    """
    # Prepare the data to be clustered
    feats = df[['principal component 1', 'principal component 2']].values

    silhouette_scores = []

    # Compute silhouette scores for k values in range 2 to max_k
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(feats)
        score = silhouette_score(feats, labels)
        silhouette_scores.append(score)

    # Plot the silhouette scores
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(2, max_k + 1), y=silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores for Different k Values')
    plt.grid(True)
    plt.show()


def perform_kmeans_clustering(df, k):
    """
    Perform k-means clustering on the PCA components
    :param df: dataframe of PCA components
    :param k: number of clusters
    :return: dataframe with cluster labels and cluster centroids
    """
    # Prepare the data to be clustered
    feats = df[['principal component 1', 'principal component 2']].values

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(feats)
    centroids = kmeans.cluster_centers_

    return df, centroids


def plot_kmeans_clustering(df, centroids, shaded=False):
    """
    Plot the k-means clustering results with PCA components
    :param df: dataframe of PCA components with cluster labels
    :param centroids: the cluster centroids
    :param shaded: boolean to determine if clusters should be shaded or not
    :return: the plot of clustering results
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='principal component 1', y='principal component 2', hue='cluster', palette='viridis',
                    s=100, alpha=0.6, edgecolor='w')

    # Highlight centroids
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


def shapiro_wilks_test(df, output_file='processed/normality_test_results.json'):
    """
    Perform the Shapiro-Wilks normality test on the data
    :param df: the dataframe of features
    :param output_file: output file to save the test results
    """
    results = {}
    for col in df.columns[1:]: # skip the first column (filename)
        stat, p_value = stats.shapiro(df[col])
        results[col] = {'statistic': stat, 'p_value': p_value, 'significant': bool(p_value < 0.05)}

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f'Normality test results saved to {output_file}')


def levenes_test(df, output_file='processed/homogeneity_test_results.json'):
    """
    Perform Levene's test for homogeneity of variance
    :param df: the dataframe of features
    :param output_file: output file to save the test results
    """
    stat, p_value = stats.levene(*[df[col] for col in df.columns[1:]]) # skip the first column (filename)
    results = {'levene_statistic': stat, 'p_value': p_value, 'significant': bool(p_value < 0.05)}

    with open(output_file, 'w') as file:
        json.dump(results, file, indent=4)

    print(f'Homogeneity test results saved to {output_file}')


def merge_clusters(df_1, df_2):
    """
    Merge the original dataframe of features with cluster labels from dataframe with PCA components
    :param df_1: dataframe of features
    :param df_2: dataframe with of PCA components and cluster labels
    :return: merged cluster with original features and cluster labels
    """
    # Ensure the 'filename' columns are of the same type
    df_1['filename'] = df_1['filename'].astype(str)
    df_2['filename'] = df_2['filename'].astype(str)

    # Merge the dataframes on the 'filename' column and add the 'cluster' column from df_2 to df_1
    raw_df_with_clusters = df_1.copy()
    raw_df_with_clusters = raw_df_with_clusters.merge(df_2[['filename', 'cluster']], on='filename', how='left')
    return raw_df_with_clusters


def kruskal_wallis_test(df, output_file='processed/kruskal_wallis_test_results.json', save=True, output_dir="analysis_plots"):
    """
    Perform the Kruskal-Wallis test on the data
    :param df: the dataframe of features with cluster labels
    :param output_file: output file to save the results
    :param save: boolean to determine if the results should be saved
    :param output_dir: directory to save the analysis plots
    """
    results = {}
    features = df.columns[1:-1] # exclude the first column (filename) and the last column (cluster)
    clusters = df['cluster'].unique()
    feature_data = {}

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for feature in tqdm(features, desc="Processing features"):
        data_by_cluster = [df[df['cluster'] == cluster][feature] for cluster in clusters]
        stat, p_value = kruskal(*data_by_cluster)
        significant = p_value < 0.05
        results[feature] = {'statistic': stat, 'p_value': p_value, 'significant': bool(significant)}

        feature_data[feature] = {f'Cluster {cluster}': df[df['cluster'] == cluster][feature].tolist() for cluster in clusters}

        plt.figure(figsize=(12, 6))

        plt.style.use('seaborn-v0_8-bright')
        sns.set_palette("husl")

        ax1 = plt.subplot(121)
        create_boxplot(ax1, feature_data[feature], feature)

        ax2 = plt.subplot(122)
        create_violin_plot(ax2, feature_data[feature], results[feature], feature)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    if save:
        with open(output_file, 'w') as file:
            json.dump(results, file, indent=4)
        print(f'Kruskal-Wallis test results saved to {output_file}')


def create_boxplot(ax, cluster_values, feature_name):
    """
    Create a boxplot of the feature values by cluster
    :param ax: the axis to plot on
    :param cluster_values: the feature values by cluster
    :param feature_name: the name of the feature
    """
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
    """
    Create a violin plot of the feature values by cluster
    :param ax: the axis to plot on
    :param cluster_values: the feature values by cluster
    :param feature_results: the results of the Kruskal-Wallis test
    :param feature_name: the name of the feature
    """
    data = []
    labels = []

    for cluster_id, values in cluster_values.items():
        data.extend(values)
        labels.extend([f"{cluster_id}"] * len(values))

    df = pd.DataFrame({
        'Cluster': labels,
        'Value': data
    })

    # Create violin plot
    sns.violinplot(x='Cluster', y='Value', data=df, ax=ax, inner='box', width=0.7)

    # Add individual points with high transparency and small size
    sns.stripplot(x='Cluster', y='Value', data=df, ax=ax, color='0.3', alpha=0.5, size=2, jitter=0.2)

    # Add significance indicators if available
    if feature_results['significant']:
        add_significance_indicators(ax, feature_results, df['Value'].max())

    ax.set_title(f"{feature_name}\nDistribution and Significance", pad=20)
    ax.set_ylabel("Value")
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)


def add_significance_indicators(ax, feature_results, max_val):
    """
    Add significance indicators to the violin plot (line with asterisks)
    :param ax: the axis to plot on
    :param feature_results: the results of the Kruskal-Wallis test
    :param max_val: the maximum value of the feature (for positioning the line)
    """
    gap = max_val * 0.05
    level = 0

    # Assuming significant_pairs is a list of tuples (cluster1, cluster2, p_value)
    significant_pairs = [(0, 1, feature_results['p_value'])]

    for c1, c2, p_val in significant_pairs:
        # Calculate position for significance bar
        y = max_val + gap + (level * gap * 2)

        # Draw significance bar
        ax.plot([c1, c1, c2, c2], [y - gap / 2, y, y, y - gap / 2], 'k-', linewidth=1)

        # Add significance asterisks
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
    # Load the raw data
    with open('processed/features.json', 'r') as file:
        sample_data = json.load(file)

    raw_df = pd.DataFrame(sample_data).T.reset_index()
    raw_df.rename(columns={'index': 'filename'}, inplace=True)

    """
    In order to uncomment and run the code properly, follow the steps below:
    1. Uncomment all lines under 1 (392) and 2 (395) and run the code
    2. Comment out 2 (395) and uncomment all lines under 3 (399-400) and run the code
    3. Comment out line 400 (plot_kmeans_clustering) and uncomment all lines under 4 (404) and 5 (407) and run the code
    4. Comment out all lines under 4 (404) and 5 (407) and uncomment all lines under 6 (411-412) and run the code
    5. Comment out all previously uncommented lines
    """

    # 0. (Optional) PRE-PCA METRICS:
    # plot_correlation_matrix(raw_df[raw_df.columns[1:]])
    # plot_cumulative_explained_variance(raw_df[raw_df.columns[1:]], n_components=6)

    # 1. CONVERT TO PCA:
    # pca_df = perform_pca(sample_data, n_components=2)

    # 2. PERFORM OPTIMIZATION:
    # plot_silhouette_scores(pca_df, max_k=10)

    # 3. PERFORM KMEANS CLUSTERING:
    # clustered_df, centroids = perform_kmeans_clustering(pca_df, k=2)
    # plot_kmeans_clustering(clustered_df, centroids, shaded=True)

    # 4. NORMALITY TEST:
    # shapiro_wilks_test(raw_df)

    # 5. HOMOGENEITY TEST:
    # levenes_test(raw_df)

    # 6. STATISTICAL ANALYSIS:
    # merged_df = merge_clusters(raw_df, clustered_df)
    # kruskal_wallis_test(merged_df)


if __name__ == '__main__':
    main()