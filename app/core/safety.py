# app/core/safety.py
from app.config import load_safety_keywords
from app.core.llm import invoke as llm_invoke

SAFETY_KEYWORDS = load_safety_keywords()

def safety_check_node(user_text: str, use_llm_confirmation: bool = False) -> bool:
    """
    Returns True if the message appears high-risk.
    - Basic rule-based keyword check (fast).
    - Optionally, asks the LLM to confirm intent (disabled by default).
    """
    text = user_text.lower()
    for kw in SAFETY_KEYWORDS.get("self_harm", []):
        if kw in text:
            return True

    # Optionally confirm with LLM (rarely necessary)
    if use_llm_confirmation:
        try:
            resp = llm_invoke(f"Does the following message indicate current intent to self-harm? Answer YES or NO.\nMessage: {user_text}")
            ans = getattr(resp, "content", "") or ""
            return ans.strip().lower().startswith("yes")
        except Exception:
            return False
    return False
