# src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict
from .analysis import (
    analyze_sentiment,
    extract_keywords,
    readability_score,
    detect_pii,
    extract_entities,
    summarize_text,
    full_analysis,
)
app = FastAPI(title="AI Text Analytics API", version="0.1.0")


class TextRequest(BaseModel):
    text: str


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze/sentiment")
def sentiment_endpoint(req: TextRequest):
    return analyze_sentiment(req.text)


@app.post("/analyze/keywords")
def keywords_endpoint(req: TextRequest):
    return extract_keywords(req.text)


@app.post("/analyze/readability")
def readability_endpoint(req: TextRequest):
    return readability_score(req.text)


@app.post("/analyze/pii")
def pii_endpoint(req: TextRequest):
    return detect_pii(req.text)


@app.post("/analyze/entities")
def entities_endpoint(req: TextRequest):
    return extract_entities(req.text)


@app.post("/analyze/summary")
def summary_endpoint(req: TextRequest):
    return summarize_text(req.text)


@app.post("/analyze/full")
def full_endpoint(req: TextRequest):
    return full_analysis(req.text)
