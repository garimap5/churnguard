from __future__ import annotations

import logging
import os
import time
from collections import deque

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Path, Request
from pydantic import BaseModel

from .agent import RetentionCopilot

log = logging.getLogger("churnguard")


class ScoreResponse(BaseModel):
    sub_id: str
    churn_risk: float
    uplift_pp: float
    net_value_inr: float
    top_drivers: list[str]


def create_app(bundle: dict | None = None) -> FastAPI:
    app = FastAPI(title="ChurnGuard", version="1.0")
    app.state.lat = deque(maxlen=2000)
    app.state.copilot = None

    def _set(b):
        app.state.copilot = RetentionCopilot(b["models"], b["features"], b["subs"])

    if bundle:
        _set(bundle)

    @app.on_event("startup")
    def _load():
        if app.state.copilot is None:
            _set(joblib.load(os.getenv("CHURNGUARD_BUNDLE", "artifacts/churnguard.joblib")))

    @app.middleware("http")
    async def _t(request: Request, call_next):
        t0 = time.perf_counter()
        r = await call_next(request)
        ms = (time.perf_counter() - t0) * 1000
        app.state.lat.append(ms)
        r.headers["X-Latency-ms"] = f"{ms:.2f}"
        log.info("%s %s %s %.1fms", request.method, request.url.path, r.status_code, ms)
        return r

    def _cp():
        if app.state.copilot is None:
            raise HTTPException(503, "model not loaded")
        return app.state.copilot

    def _check(cp, sub_id):
        if sub_id not in cp.X.index:
            raise HTTPException(404, f"unknown subscriber {sub_id}")

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": app.state.copilot is not None}

    @app.get("/metrics")
    def metrics():
        lat = np.array(app.state.lat) if app.state.lat else np.array([0.0])
        return {"requests": len(app.state.lat), "latency_ms_p50": float(np.percentile(lat, 50)),
                "latency_ms_p95": float(np.percentile(lat, 95))}

    @app.get("/score/{sub_id}", response_model=ScoreResponse)
    def score(sub_id: str = Path(..., min_length=3, max_length=20)):
        cp = _cp(); _check(cp, sub_id)
        s = cp.score(sub_id)
        return ScoreResponse(sub_id=sub_id, churn_risk=s["risk"], uplift_pp=100 * s["uplift"], net_value_inr=s["net_value"],
                             top_drivers=[n for n, _ in s["drivers"]])

    @app.get("/copilot/{sub_id}")
    def copilot(sub_id: str):
        cp = _cp(); _check(cp, sub_id)
        return cp.decide(sub_id).dict()

    return app


app = create_app() if os.getenv("CHURNGUARD_BUNDLE") else None
