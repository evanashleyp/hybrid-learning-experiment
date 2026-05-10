"""
TF-IDF vectorization module for IMDB sentiment analysis.
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_reviews(cleaned_reviews, max_features=10000, ngram_range=(1, 2), min_df=5):
    """
    Convert cleaned review texts to TF-IDF vectors.
    
    Parameters:
        max_features (int): Maximum number of features (vocabulary size)
        ngram_range (tuple): Range of n-grams to use
        min_df (int): Minimum document frequency for a term to be included
        
    Args:
        cleaned_reviews (list): List of preprocessed review texts
        
    Returns:
        tuple: (X_tfidf sparse matrix, fitted TfidfVectorizer object)
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        dtype='float32'
    )
    
    X_tfidf = tfidf.fit_transform(cleaned_reviews)
    
    return X_tfidf, tfidf
