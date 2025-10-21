# tests/test_retrieval.py
# pytest ile çalıştır: pytest -q
import os, pytest
from rag.retriever import load_vectorstore, search

@pytest.mark.skipif(not os.path.exists("vectorstore/faiss_index"),
                    reason="Index yok; önce build_index.py ile oluşturun.")
def test_retrieval_returns_docs():
    db = load_vectorstore()
    docs = search(db, "What is diabetes?", k=3)
    assert len(docs) > 0
    assert any(d.page_content for d in docs)
