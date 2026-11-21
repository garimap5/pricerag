"""FastAPI service: validated input, evidence-carrying output, latency monitoring."""
from __future__ import annotations

import logging
import os
import time
from collections import deque

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from .pipeline import PriceRAG

log = logging.getLogger("pricerag")


class PredictRequest(BaseModel):
    description: str = Field(..., min_length=10, max_length=2000)


class Neighbor(BaseModel):
    similarity: float
    price: float
    description: str


class PredictResponse(BaseModel):
    price: float
    low: float
    high: float
    neighbors: list[Neighbor]
    latency_ms: float


def create_app(model: PriceRAG | None = None) -> FastAPI:
    app = FastAPI(title="PriceRAG", version="1.0")
    app.state.model = model
    app.state.lat = deque(maxlen=2000)
    app.state.errors = 0

    @app.on_event("startup")
    def _load():
        if app.state.model is None:
            path = os.getenv("PRICERAG_MODEL", "artifacts/pricerag.joblib")
            app.state.model = PriceRAG.load(path)
            log.info("loaded model from %s", path)

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        app.state.lat.append(ms)
        resp.headers["X-Latency-ms"] = f"{ms:.2f}"
        log.info("%s %s %s %.1fms", request.method, request.url.path, resp.status_code, ms)
        return resp

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": app.state.model is not None}

    @app.get("/metrics")
    def metrics():
        lat = np.array(app.state.lat) if app.state.lat else np.array([0.0])
        return {"requests": len(app.state.lat), "errors": app.state.errors,
                "latency_ms_p50": float(np.percentile(lat, 50)),
                "latency_ms_p95": float(np.percentile(lat, 95))}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        if app.state.model is None:
            app.state.errors += 1
            raise HTTPException(503, "model not loaded")
        return app.state.model.predict_with_evidence(req.description)

    return app


app = create_app() if os.getenv("PRICERAG_MODEL") else None  # `uvicorn pricerag.api:app` with env set
