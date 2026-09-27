import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy

# Make sure NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy English model (installed via: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")

# Load English stopwords once
stop_words = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_text(text: str) -> list:
    return word_tokenize(text)

def remove_stopwords(tokens: list) -> list:
    return [token for token in tokens if token not in stop_words]

def lemmatize_tokens(tokens: list) -> list:
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

# Optional test
if __name__ == "__main__":
    sample_text = "This Agreement, made on 10 January 2024, is between ABC Ltd and John Doe. Parties are signing the agreements."
    cleaned = clean_text(sample_text)
    tokens = tokenize_text(cleaned)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = lemmatize_tokens(filtered_tokens)

    print("Cleaned Text:\n", cleaned)
    print("\nTokens:\n", tokens)
    print("\nTokens after Stopword Removal:\n", filtered_tokens)
    print("\nTokens after Lemmatization:\n", lemmatized_tokens)
