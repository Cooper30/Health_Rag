#  scripts/peek_dataset.py
# Amaç:
# - Veri setinin ilk birkaç satırını hızlıca görüntülemek (CSV dosyaları için)
# - Delimiter (ayraç) tahmini yapmak
# - Veri önizleme ile veri kalitesini/formatını hızlı kontrol etmek
#
# Kullanım (terminal):
#   python scripts/peek_dataset.py <dosya_yolu> [n]
#
# Örnek:
#   python scripts/peek_dataset.py data/OtherQA.csv 5
#
# Not:
# - Bu betik, özellikle embedding/indexleme öncesi veri formatını doğrulamak için kullanılır.

import sys, csv
from pathlib import Path


def main():
    """
    Komut satırından verilen bir CSV dosyasının ilk n satırını ekrana yazdırır.
    - Otomatik delimiter (ayraç) tahmini yapılır.
    - Satır sayısı, başlıklar ve örnek satırlar yazdırılır.

    Komut satırı argümanları:
        sys.argv[1] -> Dosya yolu (zorunlu)
        sys.argv[2] -> Önizlenecek satır sayısı (isteğe bağlı, varsayılan=5)
    """
    if len(sys.argv) < 2:
        print("Kullanım: python scripts/peek_dataset.py <dosya> [n]")
        return

    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # 🧭 Delimiter (ayraç) tahmini
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
            delim = dialect.delimiter
        except Exception:
            delim = ";"

    # 📊 Dosyayı oku
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    # 📢 Bilgi yazdırma
    print(f"Delimiter: '{delim}'  |  Satır sayısı: {len(rows)}")
    print("Başlıklar:", rows[0] if rows else [])
    for i, r in enumerate(rows[1:n+1], 1):
        print(f"{i:02d}:", r)


if __name__ == "__main__":
    main()
