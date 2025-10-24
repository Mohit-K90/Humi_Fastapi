# app/config.py
from pathlib import Path
import os
from dotenv import load_dotenv
import yaml

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"
CHROMA_DIR = BASE_DIR / "chroma_db"

CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

# Model config and safety yaml paths
MODEL_CONFIG_PATH = CONFIG_DIR / "model_config.yaml"
SAFETY_KEYWORDS_PATH = CONFIG_DIR / "safety_keywords.yaml"

# Helpers to read YAMLs with fallback defaults
def load_model_config():
    default = {
        "model": "gemini-1.5-flash",
        "temperature": 0.2,
        "max_output_tokens": 1000,
        "top_p": 1.0
    }
    try:
        if MODEL_CONFIG_PATH.exists():
            with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return {**default, **(data or {})}
    except Exception:
        pass
    return default

def load_safety_keywords():
    default = {
        "self_harm": [
            "kill myself",
            "want to die",
            "suicide",
            "end my life",
            "hang myself",
            "cut myself",
            "ending it all"
        ]
    }
    try:
        if SAFETY_KEYWORDS_PATH.exists():
            with open(SAFETY_KEYWORDS_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return {**default, **(data or {})}
    except Exception:
        pass
    return default
