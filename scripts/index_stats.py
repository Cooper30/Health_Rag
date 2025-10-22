# scripts/index_stats.py
# Amaç:
# - FAISS indeksine ait meta.json dosyasını okuyarak istatistiksel bilgi vermek
# - Toplam parça (chunk) sayısını ve kaynak bazlı dağılımı göstermek
# - Hızlı kontrol ve debugging için CLI'dan çalıştırılabilir bir yardımcı betik
#
# Kullanım (terminal):
#   python scripts/index_stats.py
#
# Not:
#   meta.json dosyası, build_index.py tarafından indeks oluşturma sırasında kaydedilir.

import json, os
from pathlib import Path


def main():
    """
    FAISS index klasöründeki meta.json dosyasını okur ve:
    - Toplam belge/parça sayısını ekrana yazdırır
    - Kaynak başına parça sayısını sıralı şekilde gösterir (ilk 10 kaynak)

    Ortam değişkenleri:
        INDEX_DIR (str): FAISS index klasörü (varsayılan: "vectorstore/faiss_index")
    """
    index_dir = os.getenv("INDEX_DIR", "vectorstore/faiss_index")
    meta = Path(index_dir) / "meta.json"

    # meta.json dosyası yoksa uyarı ver
    if not meta.exists():
        print(f"[HATA] {meta} bulunamadı. Önce indeks oluşturun.")
        return

    # JSON dosyasını oku
    data = json.loads(meta.read_text("utf-8"))
    print(f"Belge/Parça sayısı: {len(data)}")

    # Kaynak bazlı sayım
    sources = {}
    for c in data:
        src = c.get("source", "-")
        sources[src] = sources.get(src, 0) + 1

    # İlk 10 kaynağı ekrana yaz
    print("\nKaynak başına parça sayısı (ilk 10):")
    for k, v in list(sorted(sources.items(), key=lambda x: x[1], reverse=True))[:10]:
        print(f"{v:5d}  {k}")


if __name__ == "__main__":
    main()
