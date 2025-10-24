# app/models.py
from pydantic import BaseModel

class ChatRequest(BaseModel):
    user_id: str
    message: str

class ChatResponse(BaseModel):
    ai_response: str
    retrieved_context: list | None = None
    risk_flag: bool | None = None
