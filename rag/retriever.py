# rag/retriever.py
import os
from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings

def load_vectorstore(index_dir: str | None = None) -> FAISS:
    index_dir = index_dir or os.getenv("INDEX_DIR", "vectorstore/faiss_index")
    emb = get_embeddings()
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)

def search(db: FAISS, query: str, k: int = 5):
    return db.similarity_search(query, k=k)

def search_mmr(db: FAISS, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.3):
    return db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
