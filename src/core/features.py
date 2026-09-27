from sklearn.feature_extraction.text import TfidfVectorizer

def build_tfidf(corpus: list):
    """
    corpus: list of preprocessed strings (one string per document)
    Returns: TF-IDF vectorizer and feature matrix
    """
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(corpus)
    return vectorizer, X

# Optional test
if __name__ == "__main__":
    corpus = [
        "agreement make 10 january 2024 abc ltd john doe party sign agreement",
        "contract between xyz corp and jane smith payment due 5 february 2024"
    ]
    vectorizer, X = build_tfidf(corpus)
    print("Feature names:", vectorizer.get_feature_names_out())
    print("TF-IDF matrix shape:", X.shape)
