import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data_for_clustering(data, features):
    '''
    Preprocess the data for clustering analysis. This includes scaling and encoding.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the data
    features (list): The list of feature variable names

    Returns:
    DataFrame: The processed data ready for clustering
    '''
    cluster_data = data[features]
    # Scaling the data
    scaler = StandardScaler()
    cluster_data = pd.DataFrame(scaler.fit_transform(cluster_data), columns=cluster_data.columns)
    
    return cluster_data

def perform_kmeans_clustering(data, n_clusters):
    '''
    Perform K-Means clustering on the data.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the processed data for clustering
    n_clusters (int): The number of clusters to form

    Returns:
    array: The cluster labels
    '''
    kmeans = KMeans(n_clusters=n_clusters, random_state=0,n_init=10).fit(data)
    return kmeans.labels_

def plot_clusters(data, labels, title='K-Means Clustering'):
    '''
    Plot the clusters for visualization.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the data used for clustering
    labels (array): The cluster labels
    title (str): The title of the graph

    Returns:
    None
    '''
    plt.figure(figsize=(12, 6))
    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.title(title)
    plt.xlabel(data.columns[0])
    plt.ylabel(data.columns[1])
    plt.savefig(f'outputs/{title.replace(" ", "_")}.png')
    plt.show()
