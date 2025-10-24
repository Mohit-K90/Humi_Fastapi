# app/core/llm.py
"""
LLM wrapper. Tries to initialize a LangChain-compatible Gemini wrapper (via langchain_google_genai).
Falls back to a safe placeholder (echo).
"""

from app.config import GOOGLE_API_KEY, load_model_config
MODEL_CFG = load_model_config()

llm = None
_llm_available = False

try:
    # Attempt to import the LangChain Google GenAI wrapper
    # the exact class name / constructor may vary by installed package version.
    from langchain_google_genai import GoogleGenerativeAI
    # Instantiate with model_name/value according to the installed package API
    llm = GoogleGenerativeAI(
        model=MODEL_CFG.get("model"),
        temperature=float(MODEL_CFG.get("temperature", 0.2)),
        max_output_tokens=int(MODEL_CFG.get("max_output_tokens", 512)),
        top_p=float(MODEL_CFG.get("top_p", 1.0)),
        api_key=GOOGLE_API_KEY
    )
    _llm_available = True
except Exception:
    # If import fails, set llm to None and rely on placeholder behavior elsewhere.
    llm = None
    _llm_available = False


def invoke(prompt: str):
    """
    Unified invoke that returns a simple object with `.content`.
    If real LLM available, delegates to its invoke/generate method.
    Otherwise returns a simple echo placeholder.
    """
    class Resp:
        def __init__(self, content: str):
            self.content = content

    if _llm_available and llm is not None:
        try:
            # try the common invocation methods; adapt to the installed wrapper
            if hasattr(llm, "invoke"):
                return llm.invoke(prompt)
            if hasattr(llm, "generate"):
                # produce a simple interface-compatible object
                gen = llm.generate([{"input": prompt}])
                # some wrappers return structured output; try to extract
                try:
                    text = gen.generations[0][0].text
                except Exception:
                    text = str(gen)
                return Resp(text)
            # fallback call
            out = llm(prompt)
            return Resp(str(out))
        except Exception as e:
            return Resp(f"[LLM call failed: {e}]")
    else:
        # placeholder safe echo with short reply
        safe_reply = "I'm here to help. (LLM not configured; this is placeholder output.)"
        return Resp(safe_reply)
