# app/core/pipeline.py
from app.core.safety import safety_check_node
from app.core.cbt import cbt_node
from app.core.rag import rag_retrieval_node
from app.utils.storage import save_user_summary
from datetime import datetime
from langchain_core.chat_history import InMemoryChatMessageHistory, HumanMessage, AIMessage

def mental_health_pipeline(user_id: str, user_text: str, session_history: InMemoryChatMessageHistory):
    """
    Performs:
    1) safety check
    2) RAG retrieval
    3) CBT response generation
    4) save to session & Chroma
    Returns: ai_response (str), retrieved_context (list), risk_flag (bool)
    """
    # 1. Safety first (rule-based)
    risk_flag = safety_check_node(user_text)
    if risk_flag:
        helpline_msg = (
            "It seems you might be at risk. Please contact local helpline immediately.\n"
            "India: KIRAN 1800-599-0019 | Vandrevala 91-9999-666-555"
        )
        # update session & storage
        try:
            session_history.add_user_message(user_text)
            session_history.add_ai_message(helpline_msg)
        except Exception:
            pass
        save_user_summary(user_id, f"RISK_FLAG: {user_text}", tags={"risk": "high"})
        return helpline_msg, [], True

    # 2. RAG retrieval
    retrieved_context = rag_retrieval_node(user_id, user_text)

    # 3. Generate response
    ai_response = cbt_node(user_text, retrieved_context, session_history)

    # 4. Update session history and persist summary
    try:
        session_history.add_user_message(user_text)
        session_history.add_ai_message(ai_response)
    except Exception:
        pass

    # Save a summary line to Chroma for auditing / retrieval
    try:
        summary = f"{datetime.utcnow().isoformat()} | Q: {user_text[:300]} | A: {ai_response[:300]}"
    except Exception:
        summary = f"Q: {user_text[:300]} | A: {ai_response[:300]}"

    save_user_summary(user_id, summary, tags={"risk": "none"})

    return ai_response, retrieved_context or [], False
