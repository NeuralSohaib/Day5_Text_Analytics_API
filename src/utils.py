# src/utils.py
import nltk
import spacy
from langdetect import detect, DetectorFactory
from googletrans import Translator
import logging

DetectorFactory.seed = 0
logger = logging.getLogger(__name__)

# Ensure NLTK punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Ensure spaCy en model is available (small)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Translator (googletrans)
translator = Translator()


def detect_language(text: str) -> str:
    """Return short language code like 'en', 'ur', 'fr'."""
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        logger.warning("Language detection failed: %s", e)
        return "und"


def translate_to_en(text: str) -> dict:
    """
    Translate text to English using googletrans.
    Returns dict: { 'translated': str, 'src': 'xx' }
    If translation fails, returns original text as 'translated'.
    """
    try:
        res = translator.translate(text, dest="en")
        return {"translated": res.text, "src": res.src}
    except Exception as e:
        logger.warning("Translation failed: %s", e)
        return {"translated": text, "src": "und"}
