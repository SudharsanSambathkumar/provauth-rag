"""Vector index over document chunks.

Embeddings are Latent Semantic Analysis vectors (TF-IDF followed by truncated SVD,
L2-normalized). They are fully reproducible offline. The encoder is *frozen* after it
is fitted once on the original corpus, so changing restricted documents later (as the
counterfactual tests do) never moves the vectors of documents a user may read; that
mirrors a pretrained embedding model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from .kb import KnowledgeBase


@dataclass(frozen=True)
class Chunk:
    id: str
    doc_id: str
    evidence_ids: tuple[str, ...]
    text: str


def make_chunks(kb: KnowledgeBase, sentences_per_chunk: int = 4) -> list[Chunk]:
    by_doc: dict[str, list] = {}
    for e in kb.evidence.values():
        by_doc.setdefault(e.doc_id, []).append(e)
    chunks = []
    for doc_id, items in by_doc.items():
        title = kb.documents[doc_id].title
        for i in range(0, len(items), sentences_per_chunk):
            part = items[i : i + sentences_per_chunk]
            text = title + ". " + " ".join(e.text for e in part)
            chunks.append(Chunk(f"{doc_id}#{i // sentences_per_chunk}", doc_id, tuple(e.id for e in part), text))
    return chunks


class Encoder:
    def __init__(self, dim: int = 128, seed: int = 0):
        self.dim, self.seed = dim, seed
        self.model = None

    def fit(self, texts: list[str]) -> Encoder:
        dim = min(self.dim, len(texts) - 1)
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, token_pattern=r"(?u)\b[\w-]+\b"),
            TruncatedSVD(n_components=dim, random_state=self.seed),
            Normalizer(copy=False),
        )
        self.model.fit(texts)
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.transform(texts).astype(np.float32)


class VectorIndex:
    def __init__(self, kb: KnowledgeBase, encoder: Encoder, sentences_per_chunk: int = 4):
        self.kb = kb
        self.chunks = make_chunks(kb, sentences_per_chunk)
        self.encoder = encoder
        self.vectors = encoder.encode([c.text for c in self.chunks])

    def search(self, query: str, k: int, readable=None) -> list[tuple[Chunk, float]]:
        """Top-k chunks by cosine similarity; ``readable(chunk)`` filters before ranking."""
        scores = self.vectors @ self.encoder.encode([query])[0]
        out = []
        for i in np.argsort(-scores, kind="stable"):
            c = self.chunks[int(i)]
            if readable is not None and not readable(c):
                continue
            out.append((c, float(scores[i])))
            if len(out) == k:
                break
        return out
