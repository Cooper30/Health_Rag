# scripts/retrieval_check.py
# Kullanım: python scripts/retrieval_check.py "what is hypertension?" 5
import sys
from rag.retriever import load_vectorstore, search

def main():
    if len(sys.argv) < 2:
        print('Kullanım: python scripts/retrieval_check.py "<soru>" [k]')
        return
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    db = load_vectorstore()
    docs = search(db, query, k)
    for i, d in enumerate(docs, 1):
        print(f"[{i}] {d.metadata.get('source', '-')}")
        print(d.page_content[:400].strip(), "\n")

if __name__ == "__main__":
    main()
