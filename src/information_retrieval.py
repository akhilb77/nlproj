"""
information_retrieval.py
------------------------
Algorithm 4: TF-IDF + Cosine Similarity retrieval.

Each provision becomes one "document" made of its label, title (weighted x2),
Part / heading and text.  A query is vectorised with the same vectoriser and
compared with every provision using cosine similarity.  Provisions that the
user referenced explicitly (Article 21, Section 302, Part III) get a boost.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.regex_analyzer import LegalReference
from src.text_preprocessor import analyze

EXPLICIT_BOOST = 1.0   # added to the cosine score of a referenced provision
PART_BOOST = 0.15      # smaller boost for every Article inside a referenced Part


@dataclass
class Hit:
    record: dict
    score: float
    cosine: float
    boost: float


def document_text(r: dict) -> str:
    return " ".join([r["label"], r["title"], r["title"], r.get("part", ""), r.get("heading", ""), r["text"]])


def part_number(part: str) -> str:
    """'Part III - Fundamental Rights' -> 'III'"""
    return part.split(" - ")[0].replace("Part", "").strip().upper() if part else ""


class TfidfRetriever:
    def __init__(self, records: list[dict]):
        self.records = records
        self.vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(document_text(r) for r in records)

    def matches(self, r: dict, ref: LegalReference) -> bool:
        if ref.provision_type == "Preamble":
            return r["provision_type"] == "Preamble"
        return r["provision_type"] == ref.provision_type and r["provision_number"].upper() == ref.number

    def search(self, query: str, refs: list[LegalReference] | None = None, top_k: int = 5,
               documents: list[str] | None = None) -> list[Hit]:
        refs = refs or []
        q = self.vectorizer.transform([query])
        cos = cosine_similarity(q, self.matrix).ravel()
        boost = np.zeros(len(self.records))
        for i, r in enumerate(self.records):
            for ref in refs:
                if ref.provision_type == "Part":
                    if part_number(r.get("part", "")) == ref.number:
                        boost[i] += PART_BOOST
                elif self.matches(r, ref):
                    boost[i] += EXPLICIT_BOOST
        scores = cos + boost
        if documents:
            allowed = np.array([r["document_name"] in documents for r in self.records])
            scores = np.where(allowed, scores, -1.0)
        order = np.argsort(-scores)[:top_k]
        return [Hit(self.records[i], float(scores[i]), float(cos[i]), float(boost[i]))
                for i in order if scores[i] > 0]

    def find_reference(self, ref: LegalReference, documents: list[str] | None = None) -> dict | None:
        for r in self.records:
            if self.matches(r, ref) and (not documents or r["document_name"] in documents):
                return r
        return None
