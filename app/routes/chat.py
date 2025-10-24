# app/routes/chat.py
from fastapi import APIRouter, HTTPException
from app.models import ChatRequest, ChatResponse
from app.core.pipeline import mental_health_pipeline
from app.core.memory import restore_session
from typing import Dict

router = APIRouter()

# In-memory session store for demo (user_id -> InMemoryChatMessageHistory)
_sessions: Dict[str, object] = {}


@router.post("/", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    try:
        # restore or create session history
        if req.user_id not in _sessions:
            _sessions[req.user_id] = restore_session()
        session_history = _sessions[req.user_id]

        ai_response, retrieved_context, risk_flag = mental_health_pipeline(
            req.user_id, req.message, session_history
        )

        # Return the response and optionally the retrieved_context metadata for debugging
        return ChatResponse(
            ai_response=ai_response,
            retrieved_context=[
                {"snippet": d.page_content[:300], "metadata": getattr(d, "metadata", {})}
                for d in (retrieved_context or [])
            ] if retrieved_context else None,
            risk_flag=risk_flag
        )
    except Exception as e:
        # keep responses stable
        raise HTTPException(status_code=500, detail=str(e))
