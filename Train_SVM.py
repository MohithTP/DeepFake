# File: train_svm_welfake.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.model_selection import train_test_split, GridSearchCV #type: ignore
from sklearn.svm import LinearSVC #type: ignore
from sklearn.metrics import accuracy_score, classification_report #type: ignore

if __name__ == '__main__': # type: ignore
    # 1. Load preprocessed data
    df = pd.read_csv('welfake_clean.csv')

    # 2. Handle any missing text
    df['processed_text'] = df['processed_text'].fillna('')

    # 3. Vectorize with TF‑IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']  # 0 = real, 1 = fake

    # 4. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 5. Train Linear SVM
    print("Training LinearSVC…")
    svm = LinearSVC(random_state=42)
    svm.fit(X_train, y_train)

    # 6. Evaluate
    y_pred = svm.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))