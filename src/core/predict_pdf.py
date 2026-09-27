import os
import pickle
import fitz  # PyMuPDF
from preprocess import clean_text, tokenize_text, remove_stopwords, lemmatize_tokens

# Paths for saved model and vectorizer
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "..", "legal_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "..", "..", "vectorizer.pkl")

# Load model and vectorizer
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(VECTORIZER_PATH, "rb") as f:
    vectorizer = pickle.load(f)

def preprocess_document(text: str) -> str:
    """Clean, tokenize, remove stopwords, lemmatize, return as string"""
    cleaned = clean_text(text)
    tokens = tokenize_text(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize_tokens(tokens)
    return " ".join(tokens)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from PDF using PyMuPDF"""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def predict_pdf(pdf_path: str) -> str:
    text = extract_text_from_pdf(pdf_path)
    processed_text = preprocess_document(text)
    vectorized = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized)
    return prediction[0]

# Optional test
if __name__ == "__main__":
    pdf_file = input("Enter path to PDF file: ").strip()
    if os.path.exists(pdf_file):
        doc_type = predict_pdf(pdf_file)
        print(f"Predicted Document Type: {doc_type}")
    else:
        print("PDF file not found!")
