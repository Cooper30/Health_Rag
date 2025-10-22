#  app.py — Streamlit RAG (FAISS)
# Bu dosya, kullanıcıdan alınan soruları FAISS vektör veritabanı üzerinden aratarak
# en benzer bilgi parçalarını getirir ve isteğe bağlı olarak Gemini LLM kullanarak
# yanıt oluşturur. Arayüz Streamlit ile oluşturulmuştur.

import os
import streamlit as st
from dotenv import load_dotenv

# FAISS vektör veritabanı için LangChain paketleri
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# (Opsiyonel) Gemini API importu
# Eğer google.generativeai paketi yüklüyse Gemini LLM desteği otomatik açılır
USE_GEMINI = False
try:
    import google.generativeai as genai
    USE_GEMINI = True
except Exception:
    USE_GEMINI = False

# Ortam değişkenlerini .env dosyasından yükle
load_dotenv()

# -------- ⚙️ Config Ayarları --------
# INDEX_DIR        : FAISS index dosyalarının bulunduğu klasör
# EMBED_MODEL      : HuggingFace embedding modeli
# TOP_K_DEFAULT    : Varsayılan Top-K değeri (kaç parça çekileceği)
# GOOGLE_API_KEY   : Gemini API anahtarı
# GEMINI_MODEL     : Kullanılacak Gemini modeli (ör. gemini-2.5-pro)
INDEX_DIR = os.getenv("INDEX_DIR", "vectorstore/faiss_index")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
TOP_K_DEFAULT = int(os.getenv("TOP_K", 5))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Eğer Google API key varsa Gemini LLM yapılandırılır
if GOOGLE_API_KEY and USE_GEMINI:
    genai.configure(api_key=GOOGLE_API_KEY)

# -------- 🧭 Streamlit Sayfa Ayarları --------
st.set_page_config(
    page_title="🩺 Sağlık Chatbotu (RAG · FAISS)",
    page_icon="💬",
    layout="wide"
)

# Başlık ve açıklama
st.title("🩺 Sağlık Chatbotu — RAG (FAISS)")
st.caption("Soru → FAISS top-k → (ops.) Gemini ile yanıt | Kaynakları gösterir.")

# -------- 🧠 FAISS Index Yükleme --------
@st.cache_resource(show_spinner=False)
def load_index():
    """
    HuggingFace embedding modelini yükler ve FAISS index klasöründen
    önceden oluşturulmuş vektör indeksini döndürür.
    """
    emb = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(INDEX_DIR, emb, allow_dangerous_deserialization=True)

# Kullanıcı sorusunu Gemini'ye göndermek için prompt oluşturur
def build_prompt(query: str, docs):
    """
    Soru ve en yakın context parçalarına dayanarak Gemini için prompt üretir.
    Args:
        query (str): Kullanıcının sorduğu soru
        docs (List[Document]): FAISS'den dönen benzer dokümanlar
    Returns:
        str: LLM'e gönderilecek formatlı prompt
    """
    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    sys = (
        "Sen bir RAG asistanısın. Sadece verilen CONTEXT'e dayanarak yanıt ver. "
        "Emin değilsen 'Bilmiyorum' de. Yanıtta kullandığın kaynak numaralarını "
        "köşeli parantezle belirt (örn. [1], [2])."
    )
    user = f"Soru: {query}\n\nCONTEXT:\n{context}\n\nYanıt:"
    return f"{sys}\n\n{user}"

# -------- 🧭 Sidebar Ayarları --------
with st.sidebar:
    st.subheader("Ayarlar")

    # Kullanılan embedding modeli
    st.write("Embedding modeli:")
    st.code(EMBED_MODEL)

    # FAISS index klasörü
    st.write("Index klasörü:")
    st.code(INDEX_DIR)

    # Top-K değeri için slider
    top_k = st.slider(
        "Top-k (kaç parça çekilsin?)",
        min_value=1,
        max_value=10,
        value=TOP_K_DEFAULT,
        step=1
    )

    # Gemini etkin mi kontrolü
    if GOOGLE_API_KEY and USE_GEMINI:
        st.success(f"Gemini etkin: {GEMINI_MODEL}")
    else:
        st.warning("Gemini devre dışı (GOOGLE_API_KEY ekleyin).")

# -------- 📥 FAISS Index'i Belleğe Al --------
try:
    db = load_index()
    st.success("FAISS indeksi yüklendi.")
except Exception as e:
    st.error(f"FAISS yüklenemedi: {e}")
    st.stop()

# -------- 💬 Chat Arayüzü --------
# Kullanıcıdan metin girişi al
query = st.text_input("Sorunuzu yazın 👇", placeholder="Örn. Diyabet belirtileri nelerdir?")
ask = st.button("Gönder", type="primary")

# -------- 🧠 Sorgu İşleme --------
if ask and query.strip():
    with st.spinner("Aranıyor..."):
        # FAISS veritabanında benzerlik araması yap
        docs = db.similarity_search(query, k=top_k)

    # Yanıt alanı
    st.subheader("💬 Yanıt")

    # Eğer Gemini açıksa LLM cevabı üret
    if GOOGLE_API_KEY and USE_GEMINI:
        prompt = build_prompt(query, docs)
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            answer = resp.text.strip() if getattr(resp, "text", None) else "(Boş yanıt)"
        except Exception as e:
            answer = f"(Gemini hatası: {e})"
        st.write(answer)

    # Gemini kapalıysa context parçaları gösterilir
    else:
        st.info("LLM kapalı. Aşağıda ilgili context parçaları listelendi.")

    # -------- 📚 Kaynak Parçaları --------
    st.subheader("🔎 Kaynak Parçalar")
    for i, d in enumerate(docs, 1):
        st.markdown(f"**[{i}]** {d.metadata.get('source', '-')}")
        st.write(d.page_content[:800] + ("…" if len(d.page_content) > 800 else ""))
