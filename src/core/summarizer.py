import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
from core import preprocess

# Make sure punkt is available
nltk.download('punkt', quiet=True)

def summarize_text(text, num_sentences=3):
    """Simple frequency-based text summarization"""
    sentences = sent_tokenize(text)
    cleaned = [preprocess.clean_text(s) for s in sentences]
    all_words = " ".join(cleaned).split()

    freq = Counter(all_words)
    sentence_scores = {}
    for sent in sentences:
        for word in preprocess.clean_text(sent).split():
            if word in freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + freq[word]

    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
    summary = " ".join(top_sentences)
    return summary
