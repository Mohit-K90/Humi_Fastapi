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
        "model": "gemini-2.5-flash",
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
            # direct suicide intent / ideation
            "kill myself",
            "killing myself",
            "want to die",
            "wanna die",
            "want to end my life",
            "end my life",
            "end it all",
            "ending it all",
            "plan to kill myself",
            "planning to kill myself",
            "thinking about suicide",
            "thinking of ending it all",
            "thinking about dying",
            "committing suicide",
            "commit suicide",
            "attempt suicide",
            "attempting suicide",
            "take my own life",
            "taking my own life",
            "take their own life",
            "take his own life",
            "take her own life",
            "end my suffering",
            "time to end it",
            "this is the end",
            "decided to end it",
            "ending everything",
            "done living",
            "goodbye world",
            "goodbye cruel world",
            "won’t be here much longer",
            "won’t wake up tomorrow",
            "if i die tonight",
            "if i’m gone tomorrow",
            "last day alive",
            "i’m done with life",
            "no reason to live",
            "life isn’t worth it",
            "life not worth living",
            "can’t go on",
            "can’t do this anymore",
            "done with everything",
            "sleep forever",
            "never wake up",
            "wish i don’t wake up",
            "just want peace forever",
            "want everything to stop",
            "want to disappear forever",
            "want to vanish forever",
            "want to stop existing",
            "won’t make it",
            "say goodbye forever",
            "final goodbye",

            # self-harm behavior
            "cut myself",
            "cutting myself",
            "cut my wrists",
            "cutting again",
            "hurt myself",
            "hurting myself",
            "hurt myself again",
            "back to cutting",
            "can’t stop cutting",
            "bleed out",
            "slit my wrist",
            "slit my wrists",
            "overdose",
            "took too many pills",
            "take too many pills",
            "jump off a bridge",
            "jump off a building",
            "shoot myself",
            "stab myself",
            "hang myself",
            "hang it up for good",
            "self harm",
            "self-harm",
            "selfharm",
            "hurt myself on purpose",
            "destroy myself",
            "ending me",
            "end it myself"
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


