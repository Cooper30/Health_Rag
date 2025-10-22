#  rag/pipeline.py
# Amaç:
# - Kullanıcı sorgularını FAISS vektör veritabanı üzerinden aramak (retrieval)
# - İsteğe bağlı olarak Gemini LLM kullanarak son kullanıcıya doğal dilde yanıt üretmek
# - Bu katman, arama + üretim (retrieval + generation) işlemlerini birleştiren "pipeline" mantığını uygular.

import os
from .retriever import load_vectorstore, search, search_mmr
from .prompts import build_prompt

# Gemini API kullanımı isteğe bağlıdır
USE_GEMINI = False
try:
    import google.generativeai as genai
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False


def rag_answer(query: str,
               k: int = 5,
               use_mmr: bool = False,
               fetch_k: int = 20,
               lambda_mult: float = 0.3):
    """
    RAG pipeline fonksiyonu.
    Retrieval + (opsiyonel) Gemini yanıt üretimi yapar.

    Args:
        query (str): Kullanıcının sorduğu soru.
        k (int): Top-k dönecek belge sayısı.
        use_mmr (bool): True → MMR araması (çeşitliliği artırır),
                        False → klasik benzerlik araması.
        fetch_k (int): MMR aramasında aday alınacak belge sayısı.
        lambda_mult (float): MMR için dengeleme parametresi (0~1).

    Returns:
        tuple[str | None, list]:
            answer (str | None): Gemini'den gelen cevap (veya None)
            docs (list): FAISS aramasından dönen doküman listesi
    """
    # 🧭 FAISS vektör veritabanını yükle
    db = load_vectorstore()

    # 🔍 Kullanıcı sorgusuna göre arama yap
    # MMR aktifse çeşitliliği artırılmış bir sonuç döner
    docs = (
        search_mmr(db, query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
        if use_mmr else
        search(db, query, k=k)
    )

    answer = None
    api_key = os.getenv("GOOGLE_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    # 🧠 Gemini API varsa prompt üzerinden yanıt oluştur
    if api_key and USE_GEMINI:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)

            # Context + Soru → Prompt
            prompt = build_prompt(query, docs)

            # Yanıt oluştur
            resp = model.generate_content(prompt)
            answer = (resp.text or "").strip()
        except Exception as e:
            # Hata durumunda mesaj döndür
            answer = f"(Gemini hatası: {e})"

    # 📨 Yanıt (veya None) + doküman listesi döndürülür
    return answer, docs
