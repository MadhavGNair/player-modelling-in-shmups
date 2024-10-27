import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_game_data(data):
    """
    Comprehensive unsupervised learning analysis of game data

    Parameters:
    data (pd.DataFrame): DataFrame with columns like 'bullets_fired', 'collisions',
                        'survival_time', 'left_movement_pct', 'right_movement_pct',
                        'acceleration_pct'
    """
    # Normalize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_df = pd.DataFrame(scaled_data, columns=data.columns)

    # 1. K-means Clustering
    def perform_clustering(data, n_clusters=3):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(data)

        # Analyze cluster characteristics
        data['Cluster'] = clusters
        cluster_stats = data.groupby('Cluster').mean()

        return clusters, cluster_stats

    # 2. PCA Analysis
    def perform_pca(data):
        pca = PCA()
        pca_result = pca.fit_transform(data)

        # Calculate explained variance ratio
        explained_variance = pca.explained_variance_ratio_

        # Create DataFrame with first two principal components
        pca_df = pd.DataFrame(data=pca_result[:, :2],
                              columns=['PC1', 'PC2'])

        return pca_df, explained_variance

    # 3. Anomaly Detection
    def detect_anomalies(data):
        iso_forest = IsolationForest(random_state=42)
        anomalies = iso_forest.fit_predict(data)
        return anomalies == -1  # True for anomalies

    # 4. Correlation Analysis
    def analyze_correlations(data):
        correlation_matrix = data.corr()
        return correlation_matrix

    # Perform analyses
    clusters, cluster_stats = perform_clustering(scaled_df)
    pca_df, explained_variance = perform_pca(scaled_df)
    anomalies = detect_anomalies(scaled_df)
    correlations = analyze_correlations(data)

    # Visualization functions
    def plot_clusters(pca_df, clusters):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'],
                              c=clusters, cmap='viridis')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Cluster Analysis of Game Sessions')
        plt.colorbar(scatter)
        plt.show()

    def plot_correlations(correlations):
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlations, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Game Metrics')
        plt.show()

    # Example usage:
    plot_clusters(pca_df, clusters)
    plot_correlations(correlations)

    # Return analysis results
    results = {
        'clusters': clusters,
        'cluster_stats': cluster_stats,
        'pca_explained_variance': explained_variance,
        'anomalies': anomalies,
        'correlations': correlations
    }

    return results


# Example usage with sample data
sample_data = pd.DataFrame({
    'bullets_fired': np.random.randint(50, 200, 100),
    'collisions': np.random.randint(0, 20, 100),
    'survival_time': np.random.uniform(30, 300, 100),
    'left_movement_pct': np.random.uniform(0, 100, 100),
    'right_movement_pct': np.random.uniform(0, 100, 100),
    'acceleration_pct': np.random.uniform(0, 100, 100)
})

results = analyze_game_data(sample_data)