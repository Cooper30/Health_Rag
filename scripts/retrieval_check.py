#  scripts/retrieval_check.py
# Amaç:
# - FAISS indeksinden belirli bir sorgu için en benzer dokümanları çekip terminalde görüntülemek
# - RAG pipeline'ının "retrieval" kısmını LLM (Gemini) kullanmadan bağımsız test etmek
# - Top-k dokümanları hızlıca gözden geçirerek indeksin doğruluğunu kontrol etmek
#
# Kullanım (terminal):
#   python scripts/retrieval_check.py "<soru>" [k]
#
# Örnek:
#   python scripts/retrieval_check.py "what is hypertension?" 5
#
# Not:
# - Bu betik yalnızca FAISS arama kısmını test eder.
# - LLM veya prompt üretimi yapılmaz.

import sys
from rag.retriever import load_vectorstore, search


def main():
    """
    Kullanıcıdan alınan bir sorgu için FAISS veritabanında benzerlik araması yapar
    ve sonuçları terminalde listeler.

    Komut satırı argümanları:
        sys.argv[1] -> Sorgu metni (zorunlu)
        sys.argv[2] -> Döndürülecek doküman sayısı (isteğe bağlı, varsayılan=5)
    """
    if len(sys.argv) < 2:
        print('Kullanım: python scripts/retrieval_check.py "<soru>" [k]')
        return

    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # 🧭 FAISS vektör veritabanını yükle
    db = load_vectorstore()

    # 🔍 Top-k benzer dokümanı getir
    docs = search(db, query, k)

    # 📄 Dokümanları terminale yazdır
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.metadata.get('source', '-')}")
        print(d.page_content[:400].strip(), "\n")


if __name__ == "__main__":
    main()
