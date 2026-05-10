"""
Evaluation and comparison module for IMDB sentiment analysis.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def align_pseudo_labels(cluster_labels, real_labels):
    """
    Align cluster labels with real labels by checking both normal and flipped versions.
    Cluster labels are arbitrary, so we need to find the best alignment.
    
    Args:
        cluster_labels (array): Cluster assignments from K-Means
        real_labels (array): True sentiment labels
        
    Returns:
        tuple: (aligned_labels, is_flipped, alignment_accuracy)
    """
    # Compare normal alignment
    accuracy_normal = accuracy_score(real_labels, cluster_labels)
    
    # Compare flipped alignment
    flipped_labels = 1 - cluster_labels
    accuracy_flipped = accuracy_score(real_labels, flipped_labels)
    
    # Choose the better alignment
    if accuracy_flipped > accuracy_normal:
        aligned_labels = flipped_labels
        is_flipped = True
        alignment_accuracy = accuracy_flipped
    else:
        aligned_labels = cluster_labels
        is_flipped = False
        alignment_accuracy = accuracy_normal
    
    return aligned_labels, is_flipped, alignment_accuracy


def evaluate_classification(y_true, y_pred, dataset_name=""):
    """
    Evaluate classification performance with multiple metrics.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        dataset_name (str): Name of the dataset (for reporting)
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classification_report': classification_report(y_true, y_pred, output_dict=True)
    }
    
    return metrics


def compare_hybrid_vs_supervised(y_test, y_pred_hybrid, y_pred_supervised):
    """
    Compare performance of hybrid learning vs direct supervised learning.
    
    Args:
        y_test (array): True labels
        y_pred_hybrid (array): Predictions from hybrid learning (pseudo-labels)
        y_pred_supervised (array): Predictions from direct supervised learning
        
    Returns:
        pd.DataFrame: Comparison table
    """
    import pandas as pd
    
    metrics_hybrid = evaluate_classification(y_test, y_pred_hybrid)
    metrics_supervised = evaluate_classification(y_test, y_pred_supervised)
    
    comparison = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Hybrid Learning': [
            metrics_hybrid['accuracy'],
            metrics_hybrid['precision'],
            metrics_hybrid['recall'],
            metrics_hybrid['f1_score']
        ],
        'Direct Supervised': [
            metrics_supervised['accuracy'],
            metrics_supervised['precision'],
            metrics_supervised['recall'],
            metrics_supervised['f1_score']
        ]
    })
    
    # Calculate performance gap
    comparison['Gap'] = (comparison['Direct Supervised'] - comparison['Hybrid Learning']).abs()
    
    return comparison


def find_misclassified(y_true, y_pred, X=None, max_samples=10):
    """
    Find indices of misclassified samples.
    
    Args:
        y_true (array): True labels
        y_pred (array): Predicted labels
        X (array, optional): Feature matrix (for additional context)
        max_samples (int): Maximum number of misclassified samples to return
        
    Returns:
        list: Indices of misclassified samples
    """
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    # Return first max_samples misclassified indices
    return misclassified_indices[:max_samples].tolist()
