# scripts/index_stats.py
# Kullanım: python scripts/index_stats.py
import json, os
from pathlib import Path

def main():
    index_dir = os.getenv("INDEX_DIR", "vectorstore/faiss_index")
    meta = Path(index_dir) / "meta.json"
    if not meta.exists():
        print(f"[HATA] {meta} bulunamadı. Önce indeks oluşturun.")
        return
    data = json.loads(meta.read_text("utf-8"))
    print(f"Belge/Parça sayısı: {len(data)}")
    sources = {}
    for c in data:
        src = c.get("source", "-")
        sources[src] = sources.get(src, 0) + 1
    print("\nKaynak başına parça sayısı (ilk 10):")
    for k, v in list(sorted(sources.items(), key=lambda x: x[1], reverse=True))[:10]:
        print(f"{v:5d}  {k}")

if __name__ == "__main__":
    main()
