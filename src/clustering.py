"""
PCA dimensionality reduction and K-Means clustering module.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd


def run_pca_experiments(X_tfidf, n_components_list=[20, 50, 100, 200], random_state=42):
    """
    Run PCA experiments with multiple component sizes.
    
    Args:
        X_tfidf (sparse matrix): TF-IDF feature matrix
        n_components_list (list): List of component counts to test
        random_state (int): Random seed
        
    Returns:
        dict: Dictionary with PCA results keyed by n_components
              Values contain: pca_object, X_pca, explained_variance_ratio, cumulative_variance
    """
    results = {}
    
    for n_comp in n_components_list:
        pca = PCA(n_components=n_comp, random_state=random_state)
        X_pca = pca.fit_transform(X_tfidf.toarray())
        
        variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(variance_ratio)
        
        results[n_comp] = {
            'pca': pca,
            'X_pca': X_pca,
            'variance_ratio': variance_ratio,
            'cumulative_variance': cumulative_variance,
            'total_variance': cumulative_variance[-1]
        }
    
    return results


def run_kmeans(X_pca, n_clusters=2, random_state=42, n_init=10):
    """
    Run K-Means clustering on PCA-reduced data.
    
    Args:
        X_pca (ndarray): PCA-transformed feature matrix
        n_clusters (int): Number of clusters
        random_state (int): Random seed
        n_init (int): Number of initializations
        
    Returns:
        dict: Dictionary with clustering results
              Contains: cluster_labels, silhouette_score, kmeans_object
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )
    
    cluster_labels = kmeans.fit_predict(X_pca)
    sil_score = silhouette_score(X_pca, cluster_labels)
    
    return {
        'labels': cluster_labels,
        'silhouette_score': sil_score,
        'kmeans': kmeans
    }


def get_top_words(X_tfidf, cluster_labels, tfidf_vectorizer, cluster_id=0, top_n=10):
    """
    Extract top keywords for a specific cluster.
    
    Args:
        X_tfidf (sparse matrix): Original TF-IDF matrix
        cluster_labels (array): Cluster assignments
        tfidf_vectorizer: Fitted TfidfVectorizer
        cluster_id (int): Cluster to analyze
        top_n (int): Number of top words to return
        
    Returns:
        list: Top n words in the cluster
    """
    # Get indices of samples in this cluster
    cluster_mask = cluster_labels == cluster_id
    cluster_tfidf = X_tfidf[cluster_mask]
    
    # Sum TF-IDF scores for each term in the cluster
    term_scores = np.asarray(cluster_tfidf.sum(axis=0)).ravel()
    
    # Get top word indices
    top_indices = term_scores.argsort()[-top_n:][::-1]
    
    # Get vocabulary
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # Return top words
    top_words = [feature_names[i] for i in top_indices]
    
    return top_words


def analyze_clusters(X_tfidf, labels, vectorizer, top_n=15):
    """
    Analyze clusters to find distinctive keywords.
    
    Args:
        X_tfidf (sparse matrix): Original TF-IDF matrix
        labels (array): Cluster assignments
        vectorizer: Fitted TfidfVectorizer
        top_n (int): Number of top words to return per cluster
        
    Returns:
        dict: Dictionary with cluster IDs as keys and lists of top words as values
    """
    feature_names = vectorizer.get_feature_names_out()
    
    cluster_analysis = {}
    
    overall_mean = X_tfidf.mean(axis=0)

    for cluster_id in np.unique(labels):
        cluster_docs = X_tfidf[labels == cluster_id]
        
        cluster_mean = cluster_docs.mean(axis=0)
        
        # Difference from global average
        distinctive_scores = np.asarray(
            cluster_mean - overall_mean
        ).flatten()

        top_indices = distinctive_scores.argsort()[::-1][:top_n]
        
        top_words = [feature_names[i] for i in top_indices]

        cluster_analysis[f"cluster_{cluster_id}"] = top_words

    return cluster_analysis
