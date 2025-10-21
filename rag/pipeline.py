# rag/pipeline.py
import os
from .retriever import load_vectorstore, search, search_mmr
from .prompts import build_prompt

USE_GEMINI = False
try:
    import google.generativeai as genai
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False

def rag_answer(query: str, k: int = 5, use_mmr: bool = False,
               fetch_k: int = 20, lambda_mult: float = 0.3):
    """
    Retrieval + (ops.) Gemini ile cevap.
    Geriye (answer, docs) döner.
    """
    db = load_vectorstore()
    docs = (search_mmr(db, query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
            if use_mmr else search(db, query, k=k))

    answer = None
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    if api_key and USE_GEMINI:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            prompt = build_prompt(query, docs)
            resp = model.generate_content(prompt)
            answer = (resp.text or "").strip()
        except Exception as e:
            answer = f"(Gemini hatası: {e})"

    return answer, docs
