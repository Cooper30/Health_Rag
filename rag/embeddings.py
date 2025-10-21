# rag/embeddings.py
import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    # normalize_embeddings=True => cosine benzerliği için iyi pratik
    return HuggingFaceEmbeddings(model_name=model_name,
                                 encode_kwargs={"normalize_embeddings": True})
