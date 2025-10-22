#  rag/build_index.py
# Amaç:
# - CSV/JSON/TXT/MD kaynaklarından metinleri toplar
# - Metinleri parçalara (chunk) böler
# - HuggingFace embedding modeliyle vektörlere dönüştürür
# - FAISS vektör veritabanı oluşturup diske kaydeder
#
# Notlar:
# - CSV için; 'text' sütunu varsa doğrudan kullanılır.
#   Yoksa Question/Answer (ve Türkçe eşdeğerleri) şemasından metin derlenir.
# - Dosya okuma, dil ve ayraç (delimiter) farklarına dayanıklı olacak şekilde yazılmıştır.

import os, json, argparse, csv
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

# Metin bölme (split) için LC'nin yeni/eski iki modülüne de uyum
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # fallback

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# .env yükle (EMBEDDING_MODEL, CHUNK_SIZE vb.)
load_dotenv()

EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


def s(x: Any) -> str:
    """Her türlü girdiyi güvenli bir şekilde 'strip'lenmiş string'e çevirir."""
    if isinstance(x, str):
        return x.strip()
    if x is None:
        return ""
    return str(x).strip()


def get_ci(d: Dict, *keys: str) -> str:
    """
    Sözlükte anahtarları 'case-insensitive' arar.
    Örn: get_ci(row, "Question", "question", "soru")
    """
    lower = {str(k).strip().lower(): v for k, v in d.items()}
    for k in keys:
        kk = k.strip().lower()
        if kk in lower:
            return s(lower[kk])
    return ""


def extract_text_from_row(row: Dict) -> str:
    """
    Tek bir satırdan indekslenecek metni üretir.
    - 'text' sütunu varsa doğrudan onu kullanır.
    - Aksi halde Question/Answer/Explanation şemasından blok metin kurar.
    """
    text = get_ci(row, "text")
    if text:
        return text

    # QA şeması (TR/EN anahtarlar)
    q = get_ci(row, "question", "soru", "prompt", "Question")
    a = get_ci(row, "answer", "cevap", "label", "correct", "cop", "Answer")
    expl = get_ci(row, "explanation", "context", "rationale", "exp")

    # Fallback: Answer yoksa, ikinci kolon aday olarak denenir
    if not a:
        vals = list(row.values())
        if len(vals) >= 2:
            cand = s(vals[1])
            if cand and cand != q:
                a = cand

    block = []
    if q:
        block.append(f"Soru: {q}")
    if a:
        block.append(f"Doğru Cevap: {a}")
    if expl:
        block.append(f"Açıklama: {expl}")
    return "\n\n".join(block)


def read_csv_robust(path: Path) -> List[Dict]:
    """
    CSV'yi 'robust' şekilde okur:
    - Delimiter Sniffer → ardından ';' ',' '\t' fallback
    - Encoding denemeleri: utf-8, utf-8-sig, cp1254, latin-1
    - Başlıklar (header) strip edilir
    Dönüş: [{'text': ..., 'source': '...'}, ...]
    """
    # Delimiter adaylarını hazırla (Sniffer + fallback)
    delims: List[str] = []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            sample = f.read(4096)
            if sample:
                dialect = csv.Sniffer().sniff(sample, delimiters=";, \t")
                delims.append(dialect.delimiter)
    except Exception:
        pass
    for d in [";", ",", "\t"]:
        if d not in delims:
            delims.append(d)

    encodings = ["utf-8", "utf-8-sig", "cp1254", "latin-1"]

    # Tüm kombinasyonları deneyerek ilk başarılı okuma sonucunu döndür
    for enc in encodings:
        for delim in delims:
            try:
                with open(path, newline="", encoding=enc, errors="ignore") as f:
                    reader = csv.reader(f, delimiter=delim)
                    rows = list(reader)
                if not rows:
                    continue

                header = [(h or "").strip() for h in rows[0]]
                data_rows = rows[1:]
                out: List[Dict] = []
                for r in data_rows:
                    row = {
                        header[i].strip(): (r[i] if i < len(header) and i < len(r) else "")
                        for i in range(len(header))
                    }
                    text = extract_text_from_row(row)
                    if text:
                        out.append({"text": text, "source": str(path)})
                if out:
                    return out  # Başarılı okuma
            except Exception:
                continue

    print(f"[WARN] {path.name} CSV okunamadı (delimiter/encoding algılanamadı).")
    return []


def read_any(path: Path) -> List[Dict]:
    """
    Verilen dosyayı uzantısına göre okur:
    - .csv  → read_csv_robust
    - .json → QA şemasına göre
    - .txt/.md → düz metin
    Dönüş: [{'text': ..., 'source': '...'}, ...]
    """
    out: List[Dict] = []
    suf = path.suffix.lower()
    try:
        if suf == ".csv":
            return read_csv_robust(path)
        elif suf == ".json":
            data = json.loads(path.read_text("utf-8"))
            rows = data if isinstance(data, list) else [data]
            for row in rows:
                if isinstance(row, dict):
                    text = extract_text_from_row(row)
                    if text:
                        out.append({"text": text, "source": str(path)})
        elif suf in {".txt", ".md"}:
            out.append({"text": path.read_text("utf-8", errors="ignore"), "source": str(path)})
        else:
            print(f"[WARN] Desteklenmeyen uzantı: {path.name}")
    except Exception as e:
        print(f"[WARN] {path} okunamadı: {e}")
    return out


def collect_documents(input_path: str) -> List[Dict]:
    """
    Bir klasör (recursive) veya tek dosya yolundan tüm dokümanları toplar.
    Dönüş: [{'text': ..., 'source': '...'}, ...]
    """
    p = Path(input_path)
    if p.is_dir():
        files = (
            list(p.rglob("*.csv"))
            + list(p.rglob("*.json"))
            + list(p.rglob("*.txt"))
            + list(p.rglob("*.md"))
        )
    elif p.exists():
        files = [p]
    else:
        raise FileNotFoundError(input_path)

    docs: List[Dict] = []
    for fp in files:
        docs.extend(read_any(fp))
    return docs


def chunk_docs(docs: List[Dict]) -> List[Dict]:
    """
    Dokümanları RecursiveCharacterTextSplitter ile CHUNK_SIZE/CHUNK_OVERLAP parametrelerine
    göre parçalara böler. Her parça için kaynak (source) bilgisi taşınır.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks: List[Dict] = []
    for d in docs:
        for ch in splitter.split_text(d["text"]):
            chunks.append({"text": ch, "source": d["source"]})
    return chunks


def build_faiss(chunks: List[Dict], out_dir: str):
    """
    Verilen chunk'lar için:
    - HuggingFaceEmbeddings ile vektör üretir
    - FAISS index oluşturur ve 'out_dir' içine kaydeder
    - meta.json içine indekslenen chunk özetlerini yazar (debug/analiz amaçlı)
    """
    emb = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    vs = FAISS.from_texts(
        [c["text"] for c in chunks],
        emb,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(out_dir)

    # İndekslenen parçaların özet meta bilgisi
    with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    """
    CLI kullanım:
      python rag/build_index.py --input ./data/OtherQA.csv --out_dir ./vectorstore/faiss_index
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Klasör veya dosya yolu (CSV/JSON/TXT/MD)")
    ap.add_argument("--out_dir", default="vectorstore/faiss_index", help="FAISS index çıkış klasörü")
    args = ap.parse_args()

    print(f"Embedding modeli: {EMB_MODEL}")
    print(f"Chunk size/overlap: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"Girdi: {args.input}  →  Çıkış: {args.out_dir}")

    # 1) Dokümanları topla
    docs = collect_documents(args.input)
    print(f"Okunan belge sayısı: {len(docs)}")
    if not docs:
        print('[HATA] Hiç belge/tekst bulunamadı. CSV başlıklarının "Question/Answer" (veya eşdeğerleri) olduğundan emin olun.')
        return

    # 2) Parçalara böl
    chunks = chunk_docs(docs)
    print(f"Oluşturulan parça sayısı: {len(chunks)}")

    # 3) FAISS indeksini inşa et ve kaydet
    build_faiss(chunks, args.out_dir)
    print(f"\n✅ FAISS indeks '{args.out_dir}' içine kaydedildi.")


if __name__ == "__main__":
    main()
