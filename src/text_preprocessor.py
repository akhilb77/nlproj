"""
text_preprocessor.py
--------------------
Normalisation, tokenisation, stop-word removal and stemming.
Uses only resources that need no download (Porter stemmer, sklearn stop words).
"""

import re
import unicodedata

from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()

# Domain words that appear in almost every question and carry no meaning for retrieval.
QUESTION_WORDS = {"article", "articles", "section", "sections", "provision", "provisions",
                  "deal", "deals", "dealt", "explain", "describe", "tell", "constitution",
                  "india", "indian", "say", "says", "mean", "means", "does", "provide", "provides"}
STOP_WORDS = set(ENGLISH_STOP_WORDS) | QUESTION_WORDS

# A token is a word, a number with optional clause brackets (19(1)(a)), or punctuation.
TOKEN_RE = re.compile(r"\d+[A-Za-z]{0,2}(?:\(\w{1,4}\))*|[A-Za-z]+(?:[-'][A-Za-z]+)*|[?.,;:!]")


def normalize(text: str) -> str:
    """Unicode-normalise, unify quotes/dashes, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    text = text.replace("—", " - ").replace("–", " - ")
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize(text))


def stem(word: str) -> str:
    return _stemmer.stem(word.lower())


def analyze(text: str) -> list[str]:
    """Tokens used by TF-IDF: lower-case, no stop words / punctuation, stemmed."""
    out = []
    for t in tokenize(text):
        low = t.lower()
        if low in STOP_WORDS or not low[0].isalnum():
            continue
        out.append(stem(low))
    return out
