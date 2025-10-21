# app.py — Streamlit RAG (FAISS)
import os
import streamlit as st
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# (Opsiyonel) Gemini ile cevap üretimi
USE_GEMINI = False
try:
    import google.generativeai as genai
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False

load_dotenv()

# -------- Config --------
INDEX_DIR = os.getenv("INDEX_DIR", "vectorstore/faiss_index")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
TOP_K_DEFAULT = int(os.getenv("TOP_K", 5))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

if GOOGLE_API_KEY and USE_GEMINI:
    genai.configure(api_key=GOOGLE_API_KEY)

st.set_page_config(page_title="🩺 Sağlık Chatbotu (RAG · FAISS)", page_icon="💬", layout="wide")
st.title("🩺 Sağlık Chatbotu — RAG (FAISS)")
st.caption("Soru → FAISS top-k → (ops.) Gemini ile yanıt | Kaynakları gösterir.")

@st.cache_resource(show_spinner=False)
def load_index():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

def build_prompt(query: str, docs):
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    sys = ("Sen bir RAG asistanısın. Sadece verilen CONTEXT'e dayanarak yanıt ver. "
           "Emin değilsen 'Bilmiyorum' de. Yanıtta kullandığın kaynak numaralarını köşeli parantezle belirt (örn. [1], [2]).")
    user = f"Soru: {query}\n\nCONTEXT:\n{context}\n\nYanıt:"
    return f"{sys}\n\n{user}"

# -------- Sidebar --------
with st.sidebar:
    st.subheader("Ayarlar")
    st.write("Embedding modeli:")
    st.code(EMBED_MODEL)
    st.write("Index klasörü:")
    st.code(INDEX_DIR)

    top_k = st.slider("Top-k (kaç parça çekilsin?)", min_value=1, max_value=10, value=TOP_K_DEFAULT, step=1)

    if GOOGLE_API_KEY and USE_GEMINI:
        st.success(f"Gemini etkin: {GEMINI_MODEL}")
    else:
        st.warning("Gemini devre dışı (GOOGLE_API_KEY ekleyin).")

# -------- Load index --------
try:
    db = load_index()
    st.success("FAISS indeksi yüklendi.")
except Exception as e:
    st.error(f"FAISS yüklenemedi: {e}")
    st.stop()

# -------- Chat UI --------
query = st.text_input("Sorunuzu yazın 👇", placeholder="Örn. Diyabet belirtileri nelerdir?")
ask = st.button("Gönder", type="primary")

if ask and query.strip():
    with st.spinner("Aranıyor..."):
        docs = db.similarity_search(query, k=top_k)

    # Cevap
    st.subheader("💬 Yanıt")
    if GOOGLE_API_KEY and USE_GEMINI:
        prompt = build_prompt(query, docs)
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            answer = resp.text.strip() if getattr(resp, "text", None) else "(Boş yanıt)"
        except Exception as e:
            answer = f"(Gemini hatası: {e})"
        st.write(answer)
    else:
        st.info("LLM kapalı. Aşağıda ilgili context parçaları listelendi.")

    # Kaynaklar
    st.subheader("🔎 Kaynak Parçalar")
    for i, d in enumerate(docs, 1):
        st.markdown(f"**[{i}]** {d.metadata.get('source', '-')}")
        st.write(d.page_content[:800] + ("…" if len(d.page_content) > 800 else ""))
