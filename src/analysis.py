# src/analysis.py
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from collections import Counter
from typing import List, Dict, Any
import nltk
import spacy
from presidio_analyzer import AnalyzerEngine

from .utils import detect_language, translate_to_en

# Initialize analyzer tools
VADER = SentimentIntensityAnalyzer()
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

PII_ANALYZER = AnalyzerEngine()


# --------------------------
# Sentiment
# --------------------------
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """
    Use both TextBlob (polarity) and VADER (compound) for robust scoring.
    Return combined dict.
    """
    tb = TextBlob(text)
    tb_polarity = tb.sentiment.polarity
    vader_scores = VADER.polarity_scores(text)
    # simple label logic mixing both
    avg_score = (tb_polarity + vader_scores["compound"]) / 2.0
    if avg_score > 0.2:
        label = "positive"
    elif avg_score < -0.2:
        label = "negative"
    else:
        label = "neutral"
    return {
        "textblob_polarity": round(tb_polarity, 4),
        "vader": {k: round(v, 4) for k, v in vader_scores.items()},
        "combined_score": round(avg_score, 4),
        "label": label,
    }


# --------------------------
# Keywords (simple freq-based)
# --------------------------
_STOPWORDS = set(nltk.corpus.stopwords.words("english")) if "stopwords" in nltk.corpus.__dict__ else set()

def extract_keywords(text: str, top_n: int = 8) -> Dict[str, List[str]]:
    tokens = [t.lower() for t in re.findall(r"\b[a-zA-Z']+\b", text)]
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
    most_common = Counter(tokens).most_common(top_n)
    return {"keywords": [w for w, _ in most_common]}


# --------------------------
# Readability (Flesch Reading Ease)
# --------------------------
def _count_syllables(word: str) -> int:
    # approximate syllable counting
    word = word.lower()
    matches = re.findall(r"[aeiouy]+", word)
    count = max(1, len(matches))
    if word.endswith("e"):
        # rough adjustment
        count = max(1, count - 1)
    return count

def readability_score(text: str) -> Dict[str, float]:
    sentences = nltk.sent_tokenize(text)
    words = re.findall(r"\b[a-zA-Z']+\b", text)
    syllables = sum(_count_syllables(w) for w in words)
    num_sentences = max(1, len(sentences))
    num_words = max(1, len(words))
    asl = num_words / num_sentences  # avg sentence length
    asw = syllables / num_words      # avg syllables per word
    flesch = 206.835 - 1.015 * asl - 84.6 * asw
    return {"flesch_reading_ease": round(flesch, 2)}


# --------------------------
# Summarization (extractive by sentence scoring)
# --------------------------
def summarize_text(text: str, max_sentences: int = 3) -> Dict[str, Any]:
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return {"summary": " ".join(sentences)}
    # score sentences by word frequency
    words = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", text)]
    freqs = Counter(w for w in words if w not in _STOPWORDS)
    sent_scores = []
    for s in sentences:
        s_words = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", s)]
        score = sum(freqs.get(w, 0) for w in s_words)
        sent_scores.append((score, s))
    # pick top sentences
    sent_scores.sort(reverse=True, key=lambda x: x[0])
    top = [s for _, s in sent_scores[:max_sentences]]
    # preserve original order
    ordered = [s for s in sentences if s in top]
    return {"summary": " ".join(ordered)}


# --------------------------
# NER using SpaCy
# --------------------------
def extract_entities(text: str) -> Dict[str, List[Dict[str, Any]]]:
    doc = nlp(text)
    ents = {}
    for ent in doc.ents:
        ents.setdefault(ent.label_, []).append({"text": ent.text, "start": ent.start_char, "end": ent.end_char})
    return {"entities": ents}


# --------------------------
# PII detection via Presidio
# --------------------------
def detect_pii(text: str) -> Dict[str, List[Dict[str, Any]]]:
    results = PII_ANALYZER.analyze(text=text, language="en")
    entities = []
    for r in results:
        entities.append({
            "entity_type": r.entity_type,
            "start": r.start,
            "end": r.end,
            "text": text[r.start:r.end],
            "score": round(r.score, 3)
        })
    return {"pii_entities": entities}


# --------------------------
# Composite full analysis
# --------------------------
def full_analysis(text: str) -> Dict[str, Any]:
    # detect language
    from .utils import detect_language, translate_to_en
    lang = detect_language(text)

    if lang != "en":
        trans = translate_to_en(text)
        analysis_text = trans["translated"]
        translation_meta = {"original_language": trans.get("src", lang), "translated": trans["translated"]}
    else:
        analysis_text = text
        translation_meta = {"original_language": "en", "translated": text}

    sentiment = analyze_sentiment(analysis_text)
    keywords = extract_keywords(analysis_text)
    readability = readability_score(analysis_text)
    summary = summarize_text(analysis_text)
    entities = extract_entities(analysis_text)
    pii = detect_pii(analysis_text)

    return {
        "language": lang,
        "translation": translation_meta,
        "sentiment": sentiment,
        "keywords": keywords,
        "readability": readability,
        "summary": summary,
        "entities": entities,
        "pii": pii,
    }
