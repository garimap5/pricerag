# PriceRAG

Predict a product's price from its text description. The model finds similar past products and uses
them to correct a simple text model. It also gives a 90% price range and shows the similar products it used.

**Note on data:** the data is synthetic (6,000 products, 6 categories). I made it so that results can be
reproduced without any download. To use real data, run
`python scripts/run_experiments.py --csv your.csv` (columns: `description`, `price`).
Numbers below are for the synthetic data only.

## Results (test set, 1,200 products)

| Model | MAE | RMSE | RMSLE | within 20% |
|---|---|---|---|---|
| Global median | 420.5 | 678.5 | 1.136 | 8.6% |
| Ridge on TF-IDF (text only) | 87.1 | 177.7 | 0.180 | 73.3% |
| XGBoost on embeddings | 99.5 | 189.8 | 0.248 | 62.4% |
| kNN retrieval only | 118.4 | 222.9 | 0.323 | 53.2% |
| **PriceRAG (this repo)** | **80.6** | **154.5** | **0.172** | **75.2%** |

- The MAE gain over the text-only model is 6.5 (95% CI 1.4 to 11.5, paired bootstrap).
- On a harder split where test product lines are never seen in training, the gain is 4.7 (CI 0.6 to 8.7).
- The 90% price range covers 88.1% of test products (92.6% on the harder split).

## What did not work at first

1. My first version gave XGBoost the raw prices of similar products. It did not beat plain XGBoost, and it
   was worse on unseen product lines.
2. Two reasons: similar products have different specs, so their raw prices mislead; and during training a
   product could see its own siblings, which is not true at test time.
3. Fix: retrieve the *residuals* of the text model (how wrong it was on each similar product), and build
   training neighbours out-of-fold and by product family.
4. I expected retrieval to help most when there is a hidden price premium. The test showed the opposite:
   the gain was biggest with no hidden premium. So here retrieval mostly fixes the mistakes of a linear model.

## How it works

```
text -> TF-IDF -> Ridge -> base price (log) ---------------------+
  |                                                               +--> XGBoost fixes the error --> price +/- range
  +--> TF-IDF -> SVD -> FAISS: k nearest products -> their errors +
```

| File | What it does |
|---|---|
| `src/pricerag/data.py` | synthetic catalogue and train/test split |
| `src/pricerag/retrieval.py` | embeddings, FAISS index, out-of-fold neighbours |
| `src/pricerag/pipeline.py` | the model and the conformal price range |
| `src/pricerag/api.py` | FastAPI service (`/predict`, `/health`, `/metrics`) |
| `src/pricerag/llm.py` | optional LLM price estimate using the similar products (not in the results above) |
| `scripts/run_experiments.py` | runs every experiment in this README |
| `tests/` | 7 tests (no leakage, range coverage, API, save/load) |

## Run

```bash
pip install -r requirements.txt
pytest -q
python scripts/run_experiments.py
PRICERAG_MODEL=artifacts/pricerag.joblib PYTHONPATH=src uvicorn pricerag.api:app
```

## Limits
- Synthetic data. Real descriptions are messier.
- Embeddings are TF-IDF + SVD, not a neural model (`SentenceTransformerEmbedder` can replace it).
- No LLM fine-tuning (QLoRA) in this repo.
- One random seed. The price range is correct on average, not for each product.
