# File: preprocess_welfake.py

import re
import pandas as pd
import nltk #type: ignore
from nltk.corpus import stopwords #type: ignore
from nltk.stem import PorterStemmer #type: ignore

def preprocess_text(text: str) -> str:
    """
    1. Remove non‑alphabet characters
    2. Lowercase
    3. Tokenize
    4. Remove stopwords
    5. Stem
    """
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    tokens = text.split()
    filtered = [
        stemmer.stem(w) for w in tokens
        if w not in stop_words
    ]
    return ' '.join(filtered)

if __name__ == '__main__': # type: ignore
    # 1. Download NLTK assets (first run only)
    nltk.download('stopwords')

    # 2. Load raw WELFake CSV
    df = pd.read_csv('WELFake_Dataset.csv')  # adjust path/filename as needed

    # 3. Prepare stopwords & stemmer
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # 4. Apply preprocessing
    print("Preprocessing text…")
    df['processed_text'] = df['text'].astype(str).apply(preprocess_text)

    # 5. Export only what we need
    out = df[['processed_text', 'label']]
    out.to_csv('welfake_clean.csv', index=False)
    print(f"Saved cleaned data to welfake_clean.csv ({len(out)} rows)")