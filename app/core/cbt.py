# app/core/cbt.py
from app.core.llm import invoke as llm_invoke
from app.core.safety import safety_check_node
from langchain_core.chat_history import HumanMessage, AIMessage

def _build_persona_prompt():
    return (
        "You are a compassionate, non-judgmental CBT coach giving short, empathetic, practical micro-interventions. Speak naturally, using simple, encouraging language, and keep responses brief and TTS-friendly with contractions. If the user expresses high-risk thoughts, respond immediately with safety-focused support. When relevant, gently acknowledge feelings, suggest possible thought patterns, offer a small practical CBT tool, and invite reflection. Avoid headings, bullet points, or scripted responses—keep it natural and human."

    )

def cbt_node(user_text: str, retrieved_context=None, session_history=None):
    """
    Build the full prompt using session + relevant RAG docs and call the LLM (via llm_invoke).
    If safety-check triggers, returns helpline message string.
    """
    # 0. Quick rule-based safety
    if safety_check_node(user_text):
        return (
            "⚠️ HELPLINE ALERT: It seems like you may be thinking about self-harm. "
            "Please reach out immediately to a trained professional or helpline. You are not alone."
        )

    persona = _build_persona_prompt()
    full_prompt = persona + "\n"

    # RAG: include only RAG docs that share keywords with user_text
    if retrieved_context:
        if isinstance(retrieved_context, list):
            user_keywords = set(user_text.lower().split())
            relevant_docs = []
            for doc in retrieved_context:
                try:
                    doc_text = getattr(doc, "page_content", "") or ""
                except Exception:
                    doc_text = str(doc)
                doc_keywords = set(doc_text.lower().split())
                if user_keywords.intersection(doc_keywords):
                    relevant_docs.append(doc_text)
            if relevant_docs:
                context_text = "\n".join(relevant_docs[:6])
                full_prompt += f"\n--- Relevant Past Themes (RAG) ---\n{context_text}\n"

    # Session memory: last few messages if relevant
    if session_history:
        conv = ""
        recent_msgs = session_history.messages[-6:]  # last 6
        user_kw = set(user_text.lower().split())
        for m in recent_msgs:
            if isinstance(m, HumanMessage) and user_kw.intersection(set(m.content.lower().split())):
                conv += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):
                conv += f"AI: {m.content}\n"
        if conv:
            full_prompt += f"\n--- Recent Conversation ---\n{conv}\n"

    full_prompt += f"\nUser Input:\n{user_text}\n\nGenerate the assistant response following the structure above:"

    # Call LLM wrapper
    resp = llm_invoke(full_prompt)
    return getattr(resp, "content", str(resp))


