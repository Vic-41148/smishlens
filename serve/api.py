"""Thin FastAPI wrapper. No logic here: everything lives in ``smishlens.predict``.

Operational rules: bind to localhost (or put it behind auth and rate limiting),
never log message bodies, and pin ``model_version`` in every response so
downstream users can detect a model change.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from functools import lru_cache

from fastapi import FastAPI
from pydantic import BaseModel, Field

from smishlens.predict import load_model, predict_one

app = FastAPI(title="smishlens", version="1.0")


@lru_cache(maxsize=1)
def bundle():
    return load_model(os.getenv("ARTIFACT_DIR", "artifacts"))


class Req(BaseModel):
    text: str = Field(..., max_length=5000)
    explain: bool = False  # LLM sentence only if the server sets EXPLAINER=local|api


@app.get("/health")
def health():
    return {"ok": True, "model_version": bundle().meta.get("model_version")}


@app.post("/classify")
def classify(r: Req):
    mode = os.getenv("EXPLAINER", "off") if r.explain else "off"
    return asdict(predict_one(r.text, bundle(), explainer=mode))  # content is never logged
