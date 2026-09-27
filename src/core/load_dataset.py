# src/load_dataset.py
import os
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]  # project root
DATA_DIR = BASE / "data"

def load_dataset():
    texts, labels = [], []
    for class_folder in sorted(os.listdir(DATA_DIR)):
        folder = DATA_DIR / class_folder
        if folder.is_dir():
            for fname in folder.glob("*.txt"):
                with open(fname, "r", encoding="utf-8") as f:
                    texts.append(f.read())
                    labels.append(class_folder)
    return texts, labels

if __name__ == "__main__":
    X, y = load_dataset()
    print("Loaded", len(X), "samples from", len(set(y)), "classes")
