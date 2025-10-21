# 🩺 Health RAG Chatbot

**Health RAG Chatbot**, sağlık alanındaki soru–cevap verilerini kullanarak kullanıcıların tıbbi terimleri, hastalık belirtilerini ve genel sağlık konularını hızlıca anlamasına yardımcı olmak için geliştirilmiş bir **Retrieval-Augmented Generation (RAG)** tabanlı yapay zekâ uygulamasıdır.  

📊 FAISS vektör veritabanı ve 🧠 Gemini 2.5 Pro LLM kullanarak, sorgulara doğru, kaynaklı ve doğal dilde yanıtlar üretir.

---

## 🧠 Uygulama Mimarisi

Health RAG Chatbot şu RAG hattını uygular:

1. **Veri Alma (Dataset)** – *Kaggle* üzerinden indirilen CSV tabanlı Soru–Cevap veri seti projeye eklenir (`data/OtherQA.csv`).  
2. **Chunking (Parçalara Ayırma)** – Metinler belirli boyutlarda parçalara ayrılır (ör. `CHUNK_SIZE=512`, `CHUNK_OVERLAP=64`).  
3. **Embedding Oluşturma** – Parçalar, **HuggingFace `intfloat/multilingual-e5-base`** modeliyle semantik vektörlere dönüştürülür.  
4. **Vektör Depolama** – Embedding’ler **FAISS** veritabanında saklanır.  
5. **Retrieval (Erişim)** – Kullanıcı sorusu embedding’e çevrilir ve FAISS’ten en alakalı **top‑k** parçalar getirilir.  
6. **Yanıt Üretimi (ops.)** – Getirilen context ile **Gemini 2.5 Pro** modelinden nihai yanıt üretilir; kaynak numaraları gösterilir.

> Not: “Embedding” adımı, veri kaynağından bağımsızdır. Kaggle’dan gelen CSV verileri de önce parçalara ayrılır, sonra embedding’e çevrilir.

---

## 🌐 Deploy Link (isteğe bağlı)

> Örnek: [https://health-rag.onrender.com](https://health-rag.onrender.com) *(henüz deploy edilmediyse bu alan boş kalabilir)*

---

## 📊 Dataset

Uygulama, proje klasöründeki `data/OtherQA.csv` dosyasını kullanır.  
- Veri türü: Soru–Cevap (Question, Answer)  
- Örnek alanlar:
  ```
  Question, Answer
  What is diabetes?, Diabetes is a chronic disease characterized by high blood sugar levels...
  ```
- Kullanıcı ayrıca yeni veri dosyaları ekleyerek kendi FAISS indeksini oluşturabilir.

---

## ✨ Özellikler & Kullanım Alanları

- 🩺 **Tıbbi Bilgi Sorgulama:** Temel sağlık konuları hakkında doğal dilde soru-cevap.
- 🧠 **RAG + Gemini Entegrasyonu:** Doğrulanabilir ve kaynak gösteren cevaplar.
- 🌍 **İki Dilli Destek:** İngilizce ve Türkçe soruları anlayabilir.
- 📂 **Esnek Veri Yapısı:** Kendi veri setlerini kolayca ekleme.
- ⚡ **Hızlı Cevaplama:** FAISS aramasıyla anında sonuç.
- 💬 **Streamlit Arayüzü:** Kullanıcı dostu etkileşimli web arayüzü.

**İdeal Kullanıcılar:**
- Sağlık çalışanları ve öğrenciler 🧑‍⚕️  
- Tıbbi terimlere hızlı erişmek isteyen araştırmacılar  
- Chatbot ve RAG mimarilerini öğrenmek isteyen geliştiriciler 👨‍💻

---

## 🧰 Kullanılan Teknolojiler

- **Backend:** Python, Streamlit  
- **Frontend:** Streamlit bileşenleri  
- **GenAI:** Gemini 2.5 Pro, HuggingFace E5 Embeddings, LangChain  
- **Vector Database:** FAISS  
- **Environment:** dotenv

---

## 🧭 Lokal Kurulum Adımları

### 1. Depoyu Klonla
```bash
git clone https://github.com/kullanici/health-rag.git
cd health-rag
```

### 2. Sanal Ortam Oluştur ve Aktifleştir
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate
```

### 3. Gereksinimleri Yükle
```bash
pip install -r requirements.txt
```

### 4. Ortam Değişkeni (.env) Dosyasını Ayarla
```bash
copy .env.example .env
```

`.env` içeriği 👇
```ini
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-pro
EMBEDDING_MODEL=intfloat/multilingual-e5-base
INDEX_DIR=vectorstore/faiss_index
CHUNK_SIZE=512
CHUNK_OVERLAP=64
TOP_K=5
```

### 5. FAISS İndeksi Oluştur
```bash
python rag/build_index.py --input ./data/OtherQA.csv --out_dir ./vectorstore/faiss_index
```

### 6. Uygulamayı Çalıştır
```bash
streamlit run app.py
```
👉 [http://localhost:8501](http://localhost:8501)

---

## 📂 Proje Yapısı

```
health-rag/
├─ app.py                        # Streamlit arayüzü
├─ requirements.txt
├─ .env.example
│
├─ rag/
│  ├─ build_index.py
│  ├─ embeddings.py
│  ├─ retriever.py
│  ├─ prompts.py
│  ├─ pipeline.py
│  └─ utils.py
│
├─ scripts/
│  ├─ index_stats.py
│  ├─ peek_dataset.py
│  └─ retrieval_check.py
│
├─ tests/
│  └─ test_retrieval.py
│
├─ data/
│  └─ OtherQA.csv
│
└─ vectorstore/
   └─ faiss_index/
```

---

## 🧪 CLI Test Komutları

```bash
# İndeks istatistiklerini görüntüle
python scripts/index_stats.py

# Veri setinden örnek kayıtları görüntüle
python scripts/peek_dataset.py data/OtherQA.csv 5

# Retrieval testini yap
python scripts/retrieval_check.py "What is diabetes?" 5
```

---

## 📞 İletişim

Eğer proje hakkında soruların varsa:  
- 📧 E-mail: batusahin6060@gmail.com  
- 💻 GitHub: https://github.com/Cooper30 
- 🌐 LinkedIn: https://www.linkedin.com/in/batuhan-sahin-444684247/

---

## 📝 Lisans

MIT © 2025  
Bu proje **Akbank GenAI Bootcamp** kapsamında geliştirilmiştir.
