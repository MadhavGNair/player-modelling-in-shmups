import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def perform_pca(json_file, n_components=10):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Convert JSON data to DataFrame
    df = pd.DataFrame(data).T  # Transpose to get features as columns

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
    plt.title('PCA Cumulative Explained Variance Ratio')
    plt.grid()
    plt.show()

    return cumulative_explained_variance


def plot_correlation_matrix(json_file):
    # Load JSON data
    with open(json_file, 'r') as file:
        data = json.load(file)

    # Convert JSON data to DataFrame
    df = pd.DataFrame(data).T  # Transpose to get features as columns

    # Compute correlation matrix
    corr_matrix = df.corr()

    # Plot correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()


# Example usage
perform_pca('./processed/features.json')
