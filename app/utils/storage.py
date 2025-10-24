# app/utils/storage.py
from datetime import datetime
from langchain_core.documents import Document
from typing import Optional
from app.core.rag import chroma_client

def save_user_summary(user_id: str, text: str, tags: Optional[dict] = None):
    """
    Save summary into Chroma collection with metadata.
    Returns IDs or None on failure.
    """
    if chroma_client is None:
        return None

    if tags is None:
        tags = {}
    tags = {**tags, "user_id": user_id, "timestamp": datetime.now().isoformat()}

    try:
        doc = Document(page_content=text, metadata=tags)
        ids = chroma_client.add_documents([doc])
        return ids
    except Exception:
        return None
