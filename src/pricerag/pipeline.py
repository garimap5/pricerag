"""PriceRAG: retrieval-augmented *residual* price estimator with calibrated intervals.

Design (and why):
  stage 1  text-only Ridge on TF-IDF -> log-price estimate p0 (explains what the words explain).
  retrieve k nearest historical products; for each, its *residual* r = log(price) - p0_oof
           (how far its real price sat from what its text implies: line premium, brand halo ...).
  stage 2  small XGBoost learns the correction from [p0, similarity-weighted neighbour residuals,
           neighbour spread, top similarity].  final = p0 + correction.
Retrieving residuals instead of raw prices removes the confound that siblings differ in specs.
Training-time neighbours and stage-1 predictions are out-of-fold, so no row sees its own label.
Split-conformal calibration on a held-out set turns the point estimate into a 90% interval.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from .retrieval import LSAEmbedder, VectorIndex, oof_neighbors


@dataclass
class Config:
    k: int = 8
    dim: int = 96
    temp: float = 0.05          # softmax temperature over neighbour similarities
    alpha: float = 0.10         # 1 - target coverage
    ridge_alpha: float = 1.0
    seed: int = 0
    xgb: dict = field(default_factory=lambda: dict(
        n_estimators=250, max_depth=3, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.9, min_child_weight=5, n_jobs=4))


def _weights(sims, temp):
    w = np.exp((sims - sims.max(1, keepdims=True)) / temp)
    return w / w.sum(1, keepdims=True)


def residual_features(sims, nbr_resid, temp):
    w = _weights(sims, temp)
    mean = (w * nbr_resid).sum(1)
    std = np.sqrt((w * (nbr_resid - mean[:, None]) ** 2).sum(1))
    return np.column_stack([mean, std, np.median(nbr_resid, 1), nbr_resid[:, 0], sims[:, 0], sims.mean(1)])


def knn_estimate(sims, nbr_logp, temp):
    """Pure retrieval baseline: similarity-weighted mean of neighbour log-prices."""
    return (_weights(sims, temp) * nbr_logp).sum(1)


class PriceRAG:
    def __init__(self, cfg: Config | None = None, use_retrieval: bool = True):
        self.cfg, self.use_retrieval = cfg or Config(), use_retrieval

    def _stage1(self, texts):
        return self.ridge.predict(self.tfidf.transform(texts))

    def fit(self, texts, prices, calib_texts, calib_prices, groups=None):
        """groups: optional product-family ids. Pass them when queries are expected to come from
        families absent from the index, so training-time retrieval mimics test conditions."""
        c = self.cfg
        self.train_texts = list(texts)
        self.train_prices = np.asarray(prices, float)
        self.train_logp = np.log1p(self.train_prices)
        self.embedder = LSAEmbedder(c.dim, c.seed).fit(texts)
        self.train_emb = self.embedder.transform(texts)
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
        X = self.tfidf.fit_transform(texts)
        self.ridge = Ridge(alpha=c.ridge_alpha).fit(X, self.train_logp)

        if self.use_retrieval:
            sims, idx, fold = oof_neighbors(self.train_emb, c.k, seed=c.seed, groups=groups)
            p0_oof = np.zeros(len(texts))                      # out-of-fold stage-1 predictions
            for f in np.unique(fold):
                tr, te = fold != f, fold == f
                p0_oof[te] = Ridge(alpha=c.ridge_alpha).fit(X[tr], self.train_logp[tr]).predict(X[te])
            self.train_resid = self.train_logp - p0_oof
            feats = np.column_stack([p0_oof, residual_features(sims, self.train_resid[idx], c.temp)])
            self.corrector = XGBRegressor(random_state=c.seed, **c.xgb).fit(feats, self.train_resid)
        self._build_index()
        pred = self._predict_log(calib_texts)[0]
        res = np.abs(np.log1p(np.asarray(calib_prices, float)) - pred)
        n = len(res)
        q = min(1.0, np.ceil((n + 1) * (1 - c.alpha)) / n)
        self.q_hat = float(np.quantile(res, q))             # split-conformal radius (log space)
        return self

    def _build_index(self):
        self.index = VectorIndex(self.train_emb)

    def _predict_log(self, texts):
        p0 = self._stage1(texts)
        if not self.use_retrieval:
            return p0, None, None
        sims, idx = self.index.search(self.embedder.transform(texts), self.cfg.k)
        feats = np.column_stack([p0, residual_features(sims, self.train_resid[idx], self.cfg.temp)])
        return p0 + self.corrector.predict(feats), sims, idx

    def predict(self, texts) -> dict:
        lp = self._predict_log(texts)[0]
        return {"price": np.expm1(lp), "low": np.expm1(lp - self.q_hat), "high": np.expm1(lp + self.q_hat)}

    def predict_with_evidence(self, text: str, n_evidence: int = 3) -> dict:
        t0 = time.perf_counter()
        lp, sims, idx = self._predict_log([text])
        out = {"price": float(np.expm1(lp[0])), "low": float(np.expm1(lp[0] - self.q_hat)),
               "high": float(np.expm1(lp[0] + self.q_hat)), "neighbors": []}
        if idx is not None:
            for s, i in zip(sims[0][:n_evidence], idx[0][:n_evidence]):
                out["neighbors"].append({"similarity": round(float(s), 3), "price": float(self.train_prices[i]),
                                         "description": self.train_texts[i][:200]})
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    def neighbors(self, texts, k=None):
        """Raw neighbours (for the optional LLM estimator and the kNN baseline)."""
        return self.index.search(self.embedder.transform(texts), k or self.cfg.k)

    # FAISS index is rebuilt on load rather than pickled
    def save(self, path):
        idx, self.index = self.index, None
        joblib.dump(self, path)
        self.index = idx

    @staticmethod
    def load(path) -> "PriceRAG":
        m = joblib.load(path)
        m._build_index()
        return m
