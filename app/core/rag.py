# app/core/rag.py
from app.config import CHROMA_DIR, GOOGLE_API_KEY
from app.core.llm import _llm_available  # to ensure import order
from typing import List

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_chroma import Chroma
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", api_key=GOOGLE_API_KEY)
    chroma_client = Chroma(
        collection_name="user_summaries",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    _chroma_available = True
except Exception:
    chroma_client = None
    _chroma_available = False

def rag_retrieval_node(user_id: str, user_text: str, k=3) -> List:
    """
    Return top-k Document objects or empty list.
    """
    if not _chroma_available or chroma_client is None:
        return []
    try:
        # Some Chroma wrappers accept a filter parameter
        try:
            results = chroma_client.similarity_search(user_text, k=k, filter={"user_id": user_id})
        except TypeError:
            results = chroma_client.similarity_search(user_text, k=k)
        return results or []
    except Exception:
        return []
