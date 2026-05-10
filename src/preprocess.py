"""
Text preprocessing module for IMDB sentiment analysis.
Handles cleaning, tokenization, and lemmatization.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)

try:
    nltk.data.find('corpora/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)


def preprocess_text(text):
    """
    Preprocess a single review text.
    
    Steps:
    1. Remove HTML tags
    2. Lowercase text
    3. Remove URLs
    4. Remove special characters
    5. Tokenize
    6. Remove stopwords (preserve negations)
    7. Lemmatize
    
    Args:
        text (str): Raw review text
        
    Returns:
        str: Preprocessed review text
    """
    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    
    # Lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)
    
    # Remove special characters (keep alphanumeric and spaces)
    text = re.sub(r"[^a-z\s]", "", text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Define custom stopwords (domain-specific non-sentiment words)
    CUSTOM_STOPWORDS = {
        "movie", "film", "one", "character",
        "show", "story", "time", "watch",
        "see", "make", "would", "really",
        "even", "get"
    }
    
    # Define standard stopwords, but preserve negation words
    stop_words = set(stopwords.words("english"))
    negation_words = {"not", "no", "never", "nor"}
    stop_words = stop_words - negation_words
    
    # Combine standard and custom stopwords
    stop_words = stop_words.union(CUSTOM_STOPWORDS)
    
    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    return " ".join(tokens)


def preprocess_reviews(reviews):
    """
    Preprocess a series of reviews.
    
    Args:
        reviews (list or pd.Series): Collection of review texts
        
    Returns:
        list: Preprocessed reviews
    """
    return [preprocess_text(review) for review in reviews]
