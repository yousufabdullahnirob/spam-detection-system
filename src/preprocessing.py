import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def create_vectorizer(max_features=5000):
    return TfidfVectorizer(max_features=max_features)
