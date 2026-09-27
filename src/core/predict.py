import os
import pickle
import sys

# Add core directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocess import clean_text, tokenize_text, remove_stopwords, lemmatize_tokens

# Define paths relative to project root
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "legal_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "..", "vectorizer.pkl")

# Load model and vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

def preprocess_document(text: str) -> str:
    """Full preprocessing: clean, tokenize, remove stopwords, lemmatize"""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

def predict_document(text: str):
    """Predicts document type based on trained model"""
    processed_text = preprocess_document(text)
    vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized)
    return prediction[0]

# Optional test
if __name__ == "__main__":
    sample_text = "This contract is made between ABC Corp and XYZ Ltd for payment terms and delivery obligations."
    predicted_class = predict_document(sample_text)
    print("Predicted Document Type:", predicted_class)
