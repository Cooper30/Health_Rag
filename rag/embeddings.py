# rag/embeddings.py
# Amaç:
# - Projede embedding (vektör) hesaplamalarında kullanılacak HuggingFace modelini merkezi olarak tanımlamak.
# - Bu sayede diğer dosyalarda embedding modeli kodu tekrarlanmadan import edilip kullanılabilir.
# - normalize_embeddings=True parametresi, cosine benzerliği tabanlı aramalarda daha iyi sonuç alınmasını sağlar.

import os
from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    HuggingFace embedding modelini başlatır ve döndürür.
    Ortam değişkenlerinden model adı okunur; yoksa varsayılan model kullanılır:
      - Varsayılan model: "intfloat/multilingual-e5-base"
      - normalize_embeddings=True → cosine benzerliği için iyi pratik

    Returns:
        HuggingFaceEmbeddings: LangChain uyumlu embedding nesnesi
    """
    model_name = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True}
    )
