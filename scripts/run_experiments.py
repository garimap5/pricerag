"""Reproduce every number in the README.

    python scripts/run_experiments.py [--csv my_catalog.csv] [--llm 50]

Compares, on the SAME split: global median, Ridge-on-TF-IDF, XGBoost on embeddings, kNN-RAG
(retrieval only), and Residual-RAG (text model + retrieved residual correction). Runs two protocols:
  random   : siblings of a test product may be in train (typical marketplace re-listing)
  by_family: test product lines are never seen in train (hard, honest)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pricerag import Config, PriceRAG  # noqa: E402
from pricerag.data import load_csv, make_synthetic_catalog, split  # noqa: E402
from pricerag.llm import anthropic_complete, llm_estimate  # noqa: E402
from pricerag.metrics import bootstrap_ci, interval_report, regression_report  # noqa: E402
from pricerag.pipeline import knn_estimate  # noqa: E402
from pricerag.retrieval import LSAEmbedder  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "results"


def run_protocol(df, by_family, seed=0):
    tr, ca, te = split(df, by_family=by_family, seed=seed)
    y = te.price.values
    res, preds = {}, {}

    preds["Global median"] = np.full(len(te), tr.price.median())

    tf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    ridge = Ridge(alpha=1.0).fit(tf.fit_transform(tr.description), np.log1p(tr.price))
    preds["Ridge (TF-IDF)"] = np.expm1(ridge.predict(tf.transform(te.description)))

    emb = LSAEmbedder(96).fit(tr.description)
    xgb = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=.8, n_jobs=4, random_state=0)
    xgb.fit(emb.transform(tr.description), np.log1p(tr.price))
    preds["XGBoost (embeddings)"] = np.expm1(xgb.predict(emb.transform(te.description)))

    g = tr.family_id.values if by_family else None
    rag = PriceRAG(Config()).fit(tr.description, tr.price, ca.description, ca.price, groups=g)
    sims, idx = rag.neighbors(te.description)
    preds["kNN-RAG (retrieval only)"] = np.expm1(knn_estimate(sims, rag.train_logp[idx], rag.cfg.temp))
    out = rag.predict(te.description)
    preds["Residual-RAG (ours)"] = out["price"]

    for name, p in preds.items():
        res[name] = regression_report(y, p)
        res[name]["MAE_CI95"] = bootstrap_ci(y, p)
    d = np.abs(preds["Ridge (TF-IDF)"] - y) - np.abs(preds["Residual-RAG (ours)"] - y)  # >0 => RAG better
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    res["_paired_MAE_gain_vs_Ridge"] = {"mean": float(d.mean()), "CI95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    res["_interval_Residual-RAG"] = interval_report(y, out["low"], out["high"])
    res["_interval_Residual-RAG"]["target_coverage"] = 1 - rag.cfg.alpha

    # k ablation
    abl = {}
    for k in (1, 3, 8, 16):
        m = PriceRAG(Config(k=k)).fit(tr.description, tr.price, ca.description, ca.price, groups=g)
        abl[k] = regression_report(y, m.predict(te.description)["price"])["MAE"]
    res["_k_ablation_MAE"] = abl

    # error by category
    res["_MAE_by_category"] = {c: float(np.mean(np.abs(out["price"][te.category == c] - y[te.category == c])))
                               for c in sorted(te.category.unique())}
    return res, rag, (te, out), preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--llm", type=int, default=0, help="evaluate the LLM estimator on N test items")
    a = ap.parse_args()
    df = load_csv(a.csv) if a.csv else make_synthetic_catalog()
    print(f"{len(df)} products, {df.family_id.nunique()} families")

    results = {}
    for name, bf in (("random", False), ("by_family", True)):
        results[name], rag, (te, out), _ = run_protocol(df, bf)
        print(f"\n=== split: {name} ===")
        for k, v in results[name].items():
            if not k.startswith("_"):
                print(f"{k:28s} MAE {v['MAE']:8.2f}  RMSE {v['RMSE']:8.2f}  RMSLE {v['RMSLE']:.3f}  "
                      f"within20% {v['within_20pct']:.3f}")
        print("paired MAE gain vs Ridge:", results[name]["_paired_MAE_gain_vs_Ridge"])
        print("interval:", results[name]["_interval_Residual-RAG"])
        print("k ablation (MAE):", results[name]["_k_ablation_MAE"])
        if name == "random":
            OUT.mkdir(exist_ok=True)
            os.makedirs("artifacts", exist_ok=True)
            rag.save("artifacts/pricerag.joblib")
            lo, hi = out["low"], out["high"]
            fig, ax = plt.subplots(figsize=(5.5, 5))
            o = np.argsort(te.price.values)[::6]
            ax.fill_between(range(len(o)), lo[o], hi[o], alpha=.25, label="90% conformal interval")
            ax.plot(range(len(o)), te.price.values[o], "k.", ms=3, label="true price")
            ax.plot(range(len(o)), out["price"][o], "r-", lw=.8, label="prediction")
            ax.set_yscale("log"); ax.set_xlabel("test products (sorted by price)"); ax.set_ylabel("price (log)")
            ax.legend(); fig.tight_layout(); fig.savefig(OUT / "intervals.png", dpi=140)

    # when does retrieval help? vary the latent line-level premium (invisible in the text)
    sens = {}
    if not a.csv:
        for fs in (0.0, 0.08, 0.2, 0.35):
            d = make_synthetic_catalog(fam_sigma=fs)
            tr_, ca_, te_ = split(d, seed=0)
            b = PriceRAG(Config(), use_retrieval=False).fit(tr_.description, tr_.price, ca_.description, ca_.price)
            r = PriceRAG(Config()).fit(tr_.description, tr_.price, ca_.description, ca_.price)
            sens[fs] = {"TextOnly_MAE": regression_report(te_.price, b.predict(te_.description)["price"])["MAE"],
                        "ResidualRAG_MAE": regression_report(te_.price, r.predict(te_.description)["price"])["MAE"]}
            sens[fs]["RAG_gain_pct"] = 100 * (1 - sens[fs]["ResidualRAG_MAE"] / sens[fs]["TextOnly_MAE"])
        results["sensitivity_latent_family_premium"] = sens
        print("\nSensitivity (random split):")
        for fs, v in sens.items():
            print(f"  latent premium sd={fs:<5} text-only MAE {v['TextOnly_MAE']:.1f}  RAG MAE {v['ResidualRAG_MAE']:.1f}  gain {v['RAG_gain_pct']:+.1f}%")

    # model comparison chart
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, name in zip(axs, results):
        items = [(k, v["MAE"]) for k, v in results[name].items() if not k.startswith("_")]
        ax.barh([i[0] for i in items], [i[1] for i in items], color="#1f4e79")
        ax.set_title(f"MAE — {name} split"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(OUT / "model_comparison.png", dpi=140)

    if a.llm and os.getenv("ANTHROPIC_API_KEY"):
        complete = anthropic_complete()
        sub = te.sample(a.llm, random_state=0)
        p = [llm_estimate(rag, t, complete) for t in sub.description]
        ok = [(y_, p_) for y_, p_ in zip(sub.price, p) if p_]
        results["llm_estimator"] = {"n": len(ok), **regression_report(*map(np.array, zip(*ok)))}
        print("LLM estimator:", results["llm_estimator"])
    elif a.llm:
        print("--llm requested but ANTHROPIC_API_KEY not set; skipping")

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    print("\nwrote", OUT / "results.json")


if __name__ == "__main__":
    main()
