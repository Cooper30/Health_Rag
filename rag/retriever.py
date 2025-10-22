#  rag/retriever.py
# Amaç:
# - FAISS vektör veritabanını yüklemek (load_vectorstore)
# - Kullanıcı sorgusuna benzer en yakın metin parçalarını aramak (search)
# - MMR (Maximal Marginal Relevance) yöntemiyle daha çeşitli sonuçlar döndürmek (search_mmr)
#
# Bu dosya, RAG (Retrieval-Augmented Generation) pipeline'ının “retrieval” katmanını temsil eder.

import os
from langchain_community.vectorstores import FAISS
from .embeddings import get_embeddings


def load_vectorstore(index_dir: str | None = None) -> FAISS:
    """
    FAISS vektör veritabanını diskteki klasörden yükler.

    Args:
        index_dir (str | None): FAISS index klasörü yolu. (Varsayılan: .env veya 'vectorstore/faiss_index')

    Returns:
        FAISS: LangChain FAISS vektör veritabanı nesnesi.
    """
    index_dir = index_dir or os.getenv("INDEX_DIR", "vectorstore/faiss_index")
    emb = get_embeddings()
    return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)


def search(db: FAISS, query: str, k: int = 5):
    """
    Kullanıcı sorgusu için klasik benzerlik araması (cosine similarity) yapar.

    Args:
        db (FAISS): FAISS vektör veritabanı nesnesi.
        query (str): Kullanıcı sorgusu.
        k (int): Döndürülecek en yakın belge sayısı.

    Returns:
        list[Document]: Benzerlik sırasına göre dönen dokümanlar.
    """
    return db.similarity_search(query, k=k)


def search_mmr(db: FAISS, query: str, k: int = 5, fetch_k: int = 20, lambda_mult: float = 0.3):
    """
    Kullanıcı sorgusu için MMR (Maximal Marginal Relevance) tabanlı arama yapar.
    MMR, hem benzerliği hem de çeşitliliği gözeterek sonuç döndürür.

    Args:
        db (FAISS): FAISS vektör veritabanı.
        query (str): Kullanıcı sorgusu.
        k (int): Döndürülecek sonuç sayısı.
        fetch_k (int): Aday havuzundaki maksimum sonuç sayısı.
        lambda_mult (float): Çeşitlilik dengeleme katsayısı (0~1 arası).
                             1'e yaklaştıkça benzerliğe, 0'a yaklaştıkça çeşitliliğe ağırlık verilir.

    Returns:
        list[Document]: MMR sıralamasına göre dönen dokümanlar.
    """
    return db.max_marginal_relevance_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
