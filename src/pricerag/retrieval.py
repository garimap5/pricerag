"""Dense retrieval: TF-IDF -> LSA embeddings -> FAISS inner-product index.

The embedder is deliberately pluggable (`embed_fn`). The default LSA embedder is label-free
and has no downloads; swap in a sentence-transformers model by passing
`SentenceTransformerEmbedder("all-MiniLM-L6-v2")` (needs `pip install sentence-transformers`).
"""
from __future__ import annotations

import faiss
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# FAISS + XGBoost/sklearn load two OpenMP runtimes on macOS and segfault when both go
# multi-threaded. Exact search on <1M vectors is fast single-threaded, so pin it.
faiss.omp_set_num_threads(1)


class LSAEmbedder:
    def __init__(self, dim: int = 96, seed: int = 0):
        self.dim, self.seed = dim, seed
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                                     lowercase=True)
        self.svd = TruncatedSVD(n_components=dim, random_state=seed)

    def fit(self, texts):
        self.svd.fit(self.tfidf.fit_transform(texts))
        return self

    def transform(self, texts) -> np.ndarray:
        z = self.svd.transform(self.tfidf.transform(texts))
        return normalize(z).astype("float32")


class SentenceTransformerEmbedder:  # optional backend
    def __init__(self, name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(name)

    def fit(self, texts):
        return self

    def transform(self, texts) -> np.ndarray:
        return self.model.encode(list(texts), normalize_embeddings=True).astype("float32")


class VectorIndex:
    """Exact cosine search (vectors are L2-normalised so inner product == cosine)."""

    def __init__(self, emb: np.ndarray):
        self.emb = np.ascontiguousarray(emb, dtype="float32")
        self.index = faiss.IndexFlatIP(self.emb.shape[1])
        self.index.add(self.emb)

    def search(self, q: np.ndarray, k: int):
        k = min(k, self.index.ntotal)
        sims, idx = self.index.search(np.ascontiguousarray(q, dtype="float32"), k)
        return sims, idx


def oof_neighbors(emb: np.ndarray, k: int, n_folds: int = 5, seed: int = 0, groups=None):
    """Out-of-fold neighbours for training rows.

    A training row must never retrieve itself (or anything from its own fold), otherwise the
    ranker learns "the top neighbour's price is the answer" and collapses at test time.
    If `groups` is given, folds are group-aware (no sibling leakage).
    Returns (sims, idx, fold) with idx in *global* training-row coordinates.
    """
    n = len(emb)
    rng = np.random.default_rng(seed)
    if groups is None:
        fold = rng.integers(0, n_folds, n)
    else:  # whole groups go to one fold -> a row never retrieves its own family
        g, inv = np.unique(np.asarray(groups), return_inverse=True)
        fold = rng.integers(0, n_folds, len(g))[inv]
    sims = np.zeros((n, k), "float32")
    idx = np.zeros((n, k), "int64")
    for f in range(n_folds):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        s, i = VectorIndex(emb[tr]).search(emb[te], k)
        sims[te], idx[te] = s, tr[i]
    return sims, idx, fold
