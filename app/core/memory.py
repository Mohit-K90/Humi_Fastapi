# app/core/memory.py
from langchain_core.chat_history import InMemoryChatMessageHistory

def restore_session():
    """
    Return a fresh InMemoryChatMessageHistory instance.
    In production you might load from a persistent store (Redis, DB).
    """
    return InMemoryChatMessageHistory()
