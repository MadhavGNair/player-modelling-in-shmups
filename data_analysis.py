import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


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


def print_dbscan_stats(assignments, importance, stats, noise):
    print("\nCluster Assignments:")
    for file, cluster in assignments.items():
        if cluster == -1:
            print(f"{file}: Noise")
        else:
            print(f"{file}: Cluster {cluster}")

    print("\nNoise Points:")
    print(noise)

    print("\nCluster Statistics:")
    for cluster, cluster_data in stats.items():
        print(f"\n{cluster}:")
        print(f"Size: {cluster_data['size']}")
        print("Files:", cluster_data['files'])
        if 'density' in cluster_data:
            print(f"Density: {cluster_data['density']:.3f}")


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


def prepare_visualization_data(X_scaled, cluster_labels, filenames, features):
    """Prepare PCA and t-SNE visualizations with adaptable parameters for small datasets"""
    viz_data = {
        'labels': cluster_labels,
        'filenames': filenames,
        'features': features
    }

    # PCA for 2D and 3D visualization
    n_components_2d = min(2, X_scaled.shape[1])
    n_components_3d = min(3, X_scaled.shape[1])

    pca_2d = PCA(n_components=n_components_2d)
    pca_3d = PCA(n_components=n_components_3d)

    viz_data['pca_2d'] = pca_2d.fit_transform(X_scaled)
    viz_data['pca_3d'] = pca_3d.fit_transform(X_scaled)
    viz_data['pca_explained_variance_2d'] = pca_2d.explained_variance_ratio_
    viz_data['pca_explained_variance_3d'] = pca_3d.explained_variance_ratio_

    # t-SNE with adaptive perplexity for small datasets
    n_samples = X_scaled.shape[0]
    if n_samples > 10:  # Only perform t-SNE if we have enough samples
        perplexity = min(30, n_samples - 1)  # Adjust perplexity based on sample size
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            viz_data['tsne'] = tsne.fit_transform(X_scaled)
            viz_data['tsne_available'] = True
        except ValueError:
            viz_data['tsne_available'] = False
    else:
        viz_data['tsne_available'] = False

    return viz_data


def plot_clusters(viz_data, plot_type='all', save_path=None):
    """
    Create visualizations of the clusters with handling for small datasets.

    Parameters:
    -----------
    viz_data : dict
        Visualization data from prepare_visualization_data
    plot_type : str
        'all', 'pca2d', 'pca3d', or 'tsne'
    save_path : str, optional
        Path to save the plots
    """
    plt.style.use('seaborn')
    unique_labels = np.unique(viz_data['labels'])
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    if plot_type in ['all', 'pca2d']:
        # 2D PCA plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(viz_data['pca_2d'][:, 0],
                              viz_data['pca_2d'][:, 1] if viz_data['pca_2d'].shape[1] > 1 else np.zeros_like(
                                  viz_data['pca_2d'][:, 0]),
                              c=viz_data['labels'], cmap='tab10')

        # Add file names as annotations
        for i, txt in enumerate(viz_data['filenames']):
            plt.annotate(txt, (viz_data['pca_2d'][i, 0],
                               viz_data['pca_2d'][i, 1] if viz_data['pca_2d'].shape[1] > 1 else 0),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.title('Clusters Visualized using PCA (2D)')
        plt.xlabel(f'First Principal Component ({viz_data["pca_explained_variance_2d"][0]:.1%} variance)')
        if viz_data['pca_2d'].shape[1] > 1:
            plt.ylabel(f'Second Principal Component ({viz_data["pca_explained_variance_2d"][1]:.1%} variance)')
        plt.colorbar(scatter, label='Cluster')
        if save_path:
            plt.savefig(f"{save_path}_pca2d.png", bbox_inches='tight')
        plt.show()

    if plot_type in ['all', 'pca3d'] and viz_data['pca_3d'].shape[1] > 2:
        # 3D PCA plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(viz_data['pca_3d'][:, 0],
                             viz_data['pca_3d'][:, 1],
                             viz_data['pca_3d'][:, 2],
                             c=viz_data['labels'], cmap='tab10')

        # Add file names as annotations
        for i, txt in enumerate(viz_data['filenames']):
            ax.text(viz_data['pca_3d'][i, 0],
                    viz_data['pca_3d'][i, 1],
                    viz_data['pca_3d'][i, 2], txt, fontsize=8)

        ax.set_title('Clusters Visualized using PCA (3D)')
        ax.set_xlabel(f'PC1 ({viz_data["pca_explained_variance_3d"][0]:.1%})')
        ax.set_ylabel(f'PC2 ({viz_data["pca_explained_variance_3d"][1]:.1%})')
        ax.set_zlabel(f'PC3 ({viz_data["pca_explained_variance_3d"][2]:.1%})')
        plt.colorbar(scatter, label='Cluster')
        if save_path:
            plt.savefig(f"{save_path}_pca3d.png", bbox_inches='tight')
        plt.show()

    if plot_type in ['all', 'tsne'] and viz_data.get('tsne_available', False):
        # t-SNE plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(viz_data['tsne'][:, 0], viz_data['tsne'][:, 1],
                              c=viz_data['labels'], cmap='tab10')

        # Add file names as annotations
        for i, txt in enumerate(viz_data['filenames']):
            plt.annotate(txt, (viz_data['tsne'][i, 0], viz_data['tsne'][i, 1]),
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        plt.title('Clusters Visualized using t-SNE')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(scatter, label='Cluster')
        if save_path:
            plt.savefig(f"{save_path}_tsne.png", bbox_inches='tight')
        plt.show()
    elif plot_type in ['all', 'tsne']:
        print("t-SNE visualization not available due to small sample size")


def display_pca(data, cluster_dict):
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

    finalDf['cluster'] = [cluster_dict[name] for name in filenames]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA')
    cluster_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4',
                      '#46f0f0', '#f032e6', '#bcf60c', '#fabebe']
    for c in set(finalDf['cluster']):
        cluster_data = finalDf.loc[finalDf['cluster'] == c]
        ax.scatter(cluster_data['principal component 1'], cluster_data['principal component 2'], c=cluster_colors[c],
                   label=str(cluster))
    plt.show()


class Clustering:
    def __init__(self, data):
        self.data = data

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

    def dbscan_clustering(self, eps=0.5, min_samples=2, metric='euclidean'):
        # extract features and filenames
        filenames = list(self.data.keys())
        features = list(self.data[filenames[0]].keys())

        # convert to numpy array
        X = np.array([[self.data[fname][feat] for feat in features] for fname in filenames])

        # standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # adjust min_samples if necessary (avoid edge case)
        min_samples = min(min_samples, len(filenames))

        # perform DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        cluster_labels = dbscan.fit_predict(X_scaled)

        # create cluster assignments dictionary
        cluster_assignments = {fname: label for fname, label in zip(filenames, cluster_labels)}

        # identify noise points (points labeled as -1)
        noise_points = [fname for fname, label in cluster_assignments.items() if label == -1]

        # calculate feature distributions and cluster statistics
        feature_distributions = defaultdict(dict)

        cluster_stats = calculate_cluster_statistics(self.data, filenames, features, cluster_labels, X_scaled)

        viz_data = prepare_visualization_data(X_scaled, cluster_labels, filenames, features)

        return cluster_assignments, dict(feature_distributions), dict(cluster_stats), noise_points, viz_data


if __name__ == "__main__":
    with open('features.json', 'r') as file:
        sample_data = json.load(file)

    model = Clustering(sample_data)

    # # K-MEANS CLUSTERING:
    k_assignments, k_importance, k_stats = model.kmeans_clustering(n_clusters=3)
    print_kmeans_stats(k_assignments, k_importance, k_stats)

    c_dict = {}
    for file, cluster in k_assignments.items():
        c_dict[file] = int(cluster)

    # DBSCAN CLUSTERING:
    # d_assignments, d_distributions, d_stats, d_noise, plot_data = model.dbscan_clustering(
    #     eps=1,
    #     min_samples=5
    # )
    # # plot_clusters(plot_data, plot_type='all', save_path='cluster_viz')
    # print_dbscan_stats(d_assignments, d_distributions, d_stats, d_noise)
    #
    # c_dict = {}
    # for file, cluster in d_assignments.items():
    #     c_dict[file] = int(cluster)

    display_pca(sample_data, c_dict)
