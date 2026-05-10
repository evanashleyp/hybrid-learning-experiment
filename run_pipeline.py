"""
Direct Python execution of the ML pipeline without Jupyter.
This script runs all phases of the IMDB sentiment analysis project.
"""

import sys
import os

# Set working directory
os.chdir('c:\\Users\\ASUS\\Documents\\Evan\\Coding\\College\\ML\\tubes')
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Import custom modules
from preprocess import preprocess_reviews
from vectorize import vectorize_reviews
from clustering import run_pca_experiments, run_kmeans, analyze_clusters
from evaluation import align_pseudo_labels, evaluate_classification

print("="*80)
print("IMDB SENTIMENT ANALYSIS - HYBRID LEARNING PIPELINE")
print("="*80)

# ============================================================================
# PHASE 1: DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[PHASE 1] Loading and preprocessing data...")

df = pd.read_csv('./datasets/IMDB Dataset.csv')
reviews = df['review'].values
real_labels = (df['sentiment'] == 'positive').astype(int).values

print(f"✓ Loaded {len(reviews)} reviews")
print(f"✓ Label distribution: {np.bincount(real_labels)}")

# Preprocess reviews
print("✓ Preprocessing reviews (this may take a few minutes)...")
cleaned_reviews = preprocess_reviews(reviews)

# Save for next phases
os.makedirs('./outputs', exist_ok=True)
cleaned_df = pd.DataFrame({
    'cleaned_review': cleaned_reviews,
    'label': real_labels
})
cleaned_df.to_csv('./outputs/cleaned_reviews.csv', index=False)
print(f"✓ Saved {len(cleaned_reviews)} cleaned reviews")

# ============================================================================
# PHASE 2: TF-IDF VECTORIZATION
# ============================================================================
print("\n[PHASE 2] Applying TF-IDF vectorization...")

X_tfidf, tfidf_vectorizer = vectorize_reviews(cleaned_reviews)
print(f"✓ TF-IDF shape: {X_tfidf.shape}")
print(f"✓ Vocabulary size: {len(tfidf_vectorizer.vocabulary_)}")

# ============================================================================
# PHASE 3: PCA DIMENSIONALITY REDUCTION
# ============================================================================
print("\n[PHASE 3] Running PCA experiments...")

pca_results = run_pca_experiments(X_tfidf, n_components_list=[20, 50, 100, 200])

print("\nPCA Explained Variance:")
for n_comp in sorted(pca_results.keys()):
    total_var = pca_results[n_comp]['total_variance']
    print(f"  n_components = {n_comp}: {total_var*100:.2f}%")

# Visualize PCA variance
os.makedirs('./outputs/figures', exist_ok=True)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('PCA Cumulative Explained Variance', fontsize=14, fontweight='bold')

for idx, (n_comp, ax) in enumerate(zip(sorted(pca_results.keys()), axes.flatten())):
    result = pca_results[n_comp]
    cumsum_var = result['cumulative_variance']
    ax.plot(range(1, len(cumsum_var) + 1), cumsum_var, 'b-', linewidth=2)
    ax.fill_between(range(1, len(cumsum_var) + 1), cumsum_var, alpha=0.3)
    ax.set_title(f'{n_comp} components')
    ax.set_ylabel('Cumulative Variance')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('./outputs/figures/pca_variance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved PCA variance plot")

# ============================================================================
# PHASE 4: K-MEANS CLUSTERING
# ============================================================================
print("\n[PHASE 4] Running K-Means clustering...")

kmeans_results = {}
for n_comp in sorted(pca_results.keys()):
    X_pca = pca_results[n_comp]['X_pca']
    kmeans_result = run_kmeans(X_pca)
    kmeans_results[n_comp] = kmeans_result
    print(f"  n_components = {n_comp}: Silhouette = {kmeans_result['silhouette_score']:.4f}")

# Use best configuration (50 components)
best_n_comp = 50
best_X_pca = pca_results[best_n_comp]['X_pca']
best_cluster_labels = kmeans_results[best_n_comp]['labels']

np.save('./outputs/best_X_pca.npy', best_X_pca)
np.save('./outputs/best_cluster_labels.npy', best_cluster_labels)
print(f"✓ Using PCA with {best_n_comp} components")

# ============================================================================
# PHASE 5: PSEUDO-LABEL ALIGNMENT
# ============================================================================
print("\n[PHASE 5] Aligning pseudo-labels...")

aligned_pseudo_labels, is_flipped, alignment_accuracy = align_pseudo_labels(best_cluster_labels, real_labels)
print(f"✓ Alignment accuracy: {alignment_accuracy:.4f}")
print(f"✓ Labels flipped: {is_flipped}")

# ============================================================================
# PHASE 6: SUPERVISED LEARNING (Hybrid Approach)
# ============================================================================
print("\n[PHASE 6] Training supervised model (Hybrid Learning)...")

X_train, X_test, y_train_pseudo, y_test_idx = train_test_split(
    best_X_pca, range(len(aligned_pseudo_labels)),
    test_size=0.2, random_state=42, stratify=aligned_pseudo_labels
)

# Get corresponding pseudo and real labels
y_train_pseudo = aligned_pseudo_labels[list(range(len(aligned_pseudo_labels))[:len(X_train)])]
y_test_pseudo = aligned_pseudo_labels[list(y_test_idx)]
y_test_real = real_labels[list(y_test_idx)]

# Get real training labels (for comparison)
y_train_real = real_labels[list(range(len(real_labels))[:len(X_train)])]

# Hyperparameter tuning
print("✓ Running GridSearchCV (this may take several minutes)...")
param_grid = {'C': [0.1, 1, 10]}
lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train_pseudo)

print(f"✓ Best C parameter: {grid_search.best_params_['C']}")
print(f"✓ Best CV F1-Score: {grid_search.best_score_:.4f}")

# Get predictions
best_model = grid_search.best_estimator_
y_pred_hybrid = best_model.predict(X_test)

# Save results
np.save('./outputs/y_pred_test.npy', y_pred_hybrid)
np.save('./outputs/y_test_real.npy', y_test_real)
np.save('./outputs/X_test.npy', X_test)
np.save('./outputs/X_train.npy', X_train)

# ============================================================================
# PHASE 7: EVALUATION
# ============================================================================
print("\n[PHASE 7] Evaluating models...")

# Hybrid Learning Evaluation
hybrid_metrics = evaluate_classification(y_test_real, y_pred_hybrid)
print("\nHybrid Learning (Pseudo-Labels):")
print(f"  Accuracy:  {hybrid_metrics['accuracy']:.4f}")
print(f"  Precision: {hybrid_metrics['precision']:.4f}")
print(f"  Recall:    {hybrid_metrics['recall']:.4f}")
print(f"  F1-Score:  {hybrid_metrics['f1_score']:.4f}")

# Direct Supervised Learning
print("\n✓ Training Direct Supervised model for comparison...")
supervised_model = LogisticRegression(C=1, max_iter=1000, random_state=42, solver='lbfgs')
supervised_model.fit(X_train, y_train_real)
y_pred_supervised = supervised_model.predict(X_test)

supervised_metrics = evaluate_classification(y_test_real, y_pred_supervised)
print("\nDirect Supervised Learning (Real Labels):")
print(f"  Accuracy:  {supervised_metrics['accuracy']:.4f}")
print(f"  Precision: {supervised_metrics['precision']:.4f}")
print(f"  Recall:    {supervised_metrics['recall']:.4f}")
print(f"  F1-Score:  {supervised_metrics['f1_score']:.4f}")

# Comparison
print("\n" + "="*80)
print("COMPARISON: Hybrid vs Direct Supervised")
print("="*80)
print(f"{'Metric':<15} {'Hybrid':<12} {'Supervised':<12} {'Gap':<12}")
print("-"*80)
metrics_list = [
    ('Accuracy', hybrid_metrics['accuracy'], supervised_metrics['accuracy']),
    ('Precision', hybrid_metrics['precision'], supervised_metrics['precision']),
    ('Recall', hybrid_metrics['recall'], supervised_metrics['recall']),
    ('F1-Score', hybrid_metrics['f1_score'], supervised_metrics['f1_score'])
]
for name, hybrid, supervised in metrics_list:
    gap = abs(supervised - hybrid)
    print(f"{name:<15} {hybrid:<12.4f} {supervised:<12.4f} {gap:<12.4f}")

# ============================================================================
# PHASE 8: VISUALIZATIONS
# ============================================================================
print("\n[PHASE 8] Creating visualizations...")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.heatmap(hybrid_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
            ax=axes[0], xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
axes[0].set_title('Hybrid Learning Confusion Matrix')

sns.heatmap(supervised_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Greens',
            ax=axes[1], xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
axes[1].set_title('Direct Supervised Confusion Matrix')

plt.tight_layout()
plt.savefig('./outputs/figures/confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved confusion matrices plot")

# Performance comparison
fig, ax = plt.subplots(figsize=(12, 6))
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_names))
width = 0.35

hybrid_vals = [m[1] for m in metrics_list]
supervised_vals = [m[2] for m in metrics_list]

bars1 = ax.bar(x - width/2, hybrid_vals, width, label='Hybrid Learning', alpha=0.8)
bars2 = ax.bar(x + width/2, supervised_vals, width, label='Direct Supervised', alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Score')
ax.set_title('Hybrid vs Direct Supervised - Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics_names)
ax.legend()
ax.set_ylim([0, 1.05])

plt.tight_layout()
plt.savefig('./outputs/figures/performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved performance comparison plot")

# ============================================================================
# PHASE 9: ERROR ANALYSIS
# ============================================================================
print("\n[PHASE 9] Analyzing misclassifications...")

misclassified_indices = np.where(y_test_real != y_pred_hybrid)[0]
print(f"Total misclassifications: {len(misclassified_indices)} / {len(y_test_real)}")

error_analysis = []
for error_num, idx in enumerate(misclassified_indices[:5], 1):
    real_label = y_test_real[idx]
    pred_label = y_pred_hybrid[idx]
    
    error_analysis.append({
        'Error #': error_num,
        'Real Label': ['Negative', 'Positive'][real_label],
        'Predicted Label': ['Negative', 'Positive'][pred_label]
    })
    
    print(f"\n  Error #{error_num}:")
    print(f"    Real: {['Negative', 'Positive'][real_label]}, Predicted: {['Negative', 'Positive'][pred_label]}")

pd.DataFrame(error_analysis).to_csv('./outputs/error_analysis.csv', index=False)

# Save metrics to CSV
metrics_df = pd.DataFrame(metrics_list, columns=['Metric', 'Hybrid', 'Supervised'])
metrics_df.to_csv('./outputs/evaluation_metrics.csv', index=False)

print("\n" + "="*80)
print("PIPELINE EXECUTION COMPLETE!")
print("="*80)
print("\nGenerated outputs:")
print("  ✓ outputs/cleaned_reviews.csv")
print("  ✓ outputs/best_X_pca.npy")
print("  ✓ outputs/best_cluster_labels.npy")
print("  ✓ outputs/figures/pca_variance.png")
print("  ✓ outputs/figures/confusion_matrices.png")
print("  ✓ outputs/figures/performance_comparison.png")
print("  ✓ outputs/error_analysis.csv")
print("  ✓ outputs/evaluation_metrics.csv")
print("\nReady for report generation!")
