from fastapi import FastAPI, Query

from src.analysis import (
    analyze_sentiment,
    extract_keywords,
    detect_pii,
    extract_entities,
    summarize_text,
)

app = FastAPI(title="AI Text Analytics API")

# -----------------
# Health Check
# -----------------
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "API is running!"}

# -----------------
# Sentiment Analysis
# -----------------
@app.post("/analyze/sentiment")
async def sentiment_endpoint(text: str):
    return {"sentiment": analyze_sentiment(text)}

# -----------------
# Keyword Extraction
# -----------------
@app.post("/analyze/keywords")
async def keywords_endpoint(text: str):
    return {"keywords": extract_keywords(text)}

# -----------------
# Readability


# -----------------
# PII Detection
# -----------------
@app.post("/analyze/pii")
async def pii_endpoint(text: str):
    return {"pii_entities": detect_pii(text)}

# -----------------
# Multilingual Analysis
# -----------------


# -----------------
# Named Entity Recognition (NER)
# -----------------
@app.post("/analyze/summarize")
def summarize_endpoint(text: str = Query(...)):
    """
    Composite summarization endpoint:
      - summary: { "summary": "..." }
      - sentiment: dict
      - keywords: list[str]
      - pii: dict (found PII items, if any)
      - entities: list[dict]
    """
    # summary
    summary_result = summarize_text(text)

    # sentiment
    sentiment_result = analyze_sentiment(text)

    # keywords
    keywords_result = extract_keywords(text)
    keyword_list = (
        keywords_result.get("keywords") if isinstance(keywords_result, dict) else keywords_result
    )

    # pii detection
    pii_result = detect_pii(text)

    # named entity recognition
    entities_result = extract_entities(text)

    return {
        "summary": summary_result,
        "sentiment": sentiment_result,
        "keywords": keyword_list or [],
        "pii": pii_result or {},
        "entities": entities_result or [],   # <- now included
    }



@app.post("/analyze/entities")
def entities_endpoint(text: str = Query(...)):
    """
    Extract named entities from text.
    Example: "Barack Obama was the 44th President of the USA."
    """
    entities_result = extract_entities(text)
    return {"entities": entities_result or []}


