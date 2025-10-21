# rag/build_index.py
import os, json, argparse, csv
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # fallback

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMB_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


def s(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    if x is None:
        return ""
    return str(x).strip()


def get_ci(d: Dict, *keys: str) -> str:
    """Sözlükte anahtarları case-insensitive ve strip'lenmiş olarak ara."""
    lower = {str(k).strip().lower(): v for k, v in d.items()}
    for k in keys:
        kk = k.strip().lower()
        if kk in lower:
            return s(lower[kk])
    return ""


def extract_text_from_row(row: Dict) -> str:
    """Satırdan indekslenecek metni üret: 'text' varsa doğrudan al; yoksa QA şeması kur."""
    text = get_ci(row, "text")
    if text:
        return text

    # QA şeması (TR/EN anahtarlar)
    q = get_ci(row, "question", "soru", "prompt", "Question")
    a = get_ci(row, "answer", "cevap", "label", "correct", "cop", "Answer")
    expl = get_ci(row, "explanation", "context", "rationale", "exp")

    # Fallback: Answer bulunamazsa ikinci kolonu aday yap
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
    CSV'yi güvenle oku:
    - Delimiter: Sniffer, sonra fallback olarak ; , \t
    - Encoding: utf-8, utf-8-sig, cp1254, latin-1
    - Başlıklar strip edilir
    """
    # Delimiter adayları
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
                    row = {header[i].strip(): (r[i] if i < len(header) and i < len(r) else "") for i in range(len(header))}
                    text = extract_text_from_row(row)
                    if text:
                        out.append({"text": text, "source": str(path)})
                if out:
                    return out  # başarılı okuma
            except Exception:
                continue

    print(f"[WARN] {path.name} CSV okunamadı (delimiter/encoding algılanamadı).")
    return []


def read_any(path: Path) -> List[Dict]:
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: List[Dict] = []
    for d in docs:
        for ch in splitter.split_text(d["text"]):
            chunks.append({"text": ch, "source": d["source"]})
    return chunks


def build_faiss(chunks: List[Dict], out_dir: str):
    emb = HuggingFaceEmbeddings(model_name=EMB_MODEL, encode_kwargs={"normalize_embeddings": True})
    vs = FAISS.from_texts(
        [c["text"] for c in chunks],
        emb,
        metadatas=[{"source": c["source"]} for c in chunks],
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(out_dir)
    with open(Path(out_dir) / "meta.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Klasör veya dosya yolu (CSV/JSON/TXT/MD)")
    ap.add_argument("--out_dir", default="vectorstore/faiss_index", help="FAISS index çıkış klasörü")
    args = ap.parse_args()

    print(f"Embedding modeli: {EMB_MODEL}")
    print(f"Chunk size/overlap: {CHUNK_SIZE}/{CHUNK_OVERLAP}")
    print(f"Girdi: {args.input}  →  Çıkış: {args.out_dir}")

    docs = collect_documents(args.input)
    print(f"Okunan belge sayısı: {len(docs)}")
    if not docs:
        print('[HATA] Hiç belge/tekst bulunamadı. CSV başlıklarının "Question/Answer" (veya eşdeğerleri) olduğundan emin olun.')
        return

    chunks = chunk_docs(docs)
    print(f"Oluşturulan parça sayısı: {len(chunks)}")

    build_faiss(chunks, args.out_dir)
    print(f"\n✅ FAISS indeks '{args.out_dir}' içine kaydedildi.")


if __name__ == "__main__":
    main()
