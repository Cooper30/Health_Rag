# scripts/peek_dataset.py
# Kullanım: python scripts/peek_dataset.py data/OtherQA.csv 5
import sys, csv
from pathlib import Path

def main():
    if len(sys.argv) < 2:
        print("Kullanım: python scripts/peek_dataset.py <dosya> [n]")
        return
    path = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    # Delimiter tahmini
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
            delim = dialect.delimiter
        except Exception:
            delim = ";"

    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter=delim)
        rows = list(reader)

    print(f"Delimiter: '{delim}'  |  Satır sayısı: {len(rows)}")
    print("Başlıklar:", rows[0] if rows else [])
    for i, r in enumerate(rows[1:n+1], 1):
        print(f"{i:02d}:", r)

if __name__ == "__main__":
    main()
