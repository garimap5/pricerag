import numpy as np
import pytest
from fastapi.testclient import TestClient

from pricerag import Config, PriceRAG
from pricerag.api import create_app
from pricerag.data import make_synthetic_catalog, split
from pricerag.llm import build_prompt, parse_price
from pricerag.metrics import interval_report
from pricerag.retrieval import LSAEmbedder, oof_neighbors


@pytest.fixture(scope="module")
def fitted():
    df = make_synthetic_catalog(n_families=500, seed=1)
    tr, ca, te = split(df, seed=1)
    cfg = Config(xgb=dict(n_estimators=120, max_depth=4, learning_rate=0.1, n_jobs=2))
    return PriceRAG(cfg).fit(tr.description, tr.price, ca.description, ca.price), te


def test_split_by_family_has_no_family_overlap():
    df = make_synthetic_catalog(200, seed=2)
    tr, ca, te = split(df, by_family=True)
    assert not set(tr.family_id) & set(te.family_id)
    assert not set(ca.family_id) & set(te.family_id)


def test_oof_neighbors_never_return_own_fold():
    df = make_synthetic_catalog(100, seed=3)
    emb = LSAEmbedder(32).fit(df.description).transform(df.description)
    sims, idx, fold = oof_neighbors(emb, k=5)
    assert (fold[idx] != fold[:, None]).all()          # no same-fold neighbour
    assert not (idx == np.arange(len(emb))[:, None]).any()  # never itself


def test_conformal_coverage_close_to_target(fitted):
    model, te = fitted
    out = model.predict(te.description)
    cov = interval_report(te.price, out["low"], out["high"])["coverage"]
    assert cov >= 0.82  # target 0.90; allow finite-sample slack on a small test set


def test_predictions_positive_and_ordered(fitted):
    model, te = fitted
    out = model.predict(te.description)
    assert (out["price"] > 0).all() and (out["low"] <= out["price"]).all() and (out["price"] <= out["high"]).all()


def test_save_load_roundtrip(fitted, tmp_path):
    model, te = fitted
    model.save(tmp_path / "m.joblib")
    again = PriceRAG.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(model.predict(te.description[:20])["price"],
                               again.predict(te.description[:20])["price"], rtol=1e-5)


def test_api_contract(fitted):
    model, _ = fitted
    c = TestClient(create_app(model))
    assert c.get("/health").json()["model_loaded"] is True
    r = c.post("/predict", json={"description": "Sony over-ear wireless headphones with active noise cancelling, 30 hours battery"})
    assert r.status_code == 200
    body = r.json()
    assert body["low"] <= body["price"] <= body["high"] and len(body["neighbors"]) == 3
    assert "x-latency-ms" in r.headers
    assert c.post("/predict", json={"description": "short"}).status_code == 422  # input validation
    assert c.get("/metrics").json()["latency_ms_p95"] >= 0


def test_llm_prompt_and_parser():
    p = build_prompt("q item", ["a", "b"], [10.0, 20.0])
    assert "1. a -> $10.00" in p and "q item" in p
    assert parse_price("reasoning...\nPRICE: $1,234.50") == 1234.5
    assert parse_price("no number here") is None
