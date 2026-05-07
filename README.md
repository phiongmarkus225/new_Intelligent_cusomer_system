# 🤖 Intelligent Customer Assistant

Sistem **Intelligent Customer Assistant** berbasis arsitektur **RAG (Retrieval-Augmented Generation)** menggunakan LangChain, PostgreSQL (PgVector), Streamlit, dan Groq (LLaMA3).

---

## 📁 Struktur Proyek

```
assigment_Intelligent_CUstomer_assistant/
├── app.py                 # Streamlit Chatbot UI (Part 3)
├── search_engine.py       # Semantic Search Engine (Part 2)
├── setup_db.py            # Database setup & data ingestion (Part 1)
├── generate_data.py       # Knowledge Base generator (50 entries)
├── benchmark.py           # Search performance benchmarking (Part 2)
├── data/
│   └── knowledge_base.csv # 50 entri knowledge base
├── docker-compose.yml     # PostgreSQL + PgVector container
├── requirements.txt       # Python dependencies
├── .env                   # API keys & database config
├── .gitignore
└── README.md              # Dokumentasi ini
```

---

## Part 1: Environment Setup & Knowledge Base Preparation

### 1.1 Development Environment Configuration

| Komponen | Detail |
|----------|--------|
| **IDE** | VS Code |
| **Runtime** | Python 3.12 + venv |
| **Database** | PostgreSQL 16 + PgVector (Docker) |
| **LLM Provider** | Groq API (LLaMA3-8B-8192) |
| **Embedding Model** | sentence-transformers/all-MiniLM-L6-v2 |
| **UI Framework** | Streamlit |
| **Orchestrator** | LangChain |

**Konfigurasi:**
- Semua konfigurasi (API keys, database URL) dipusatkan di file `.env`
- Docker Compose menyediakan PostgreSQL + PgVector secara otomatis
- Virtual environment (`venv`) mengisolasi dependensi proyek

### 1.2 Database Schema Design

**Schema PgVector (langchain_pg_embedding):**

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `id` | VARCHAR | UUID unik untuk setiap dokumen |
| `collection_id` | UUID | FK ke collection |
| `embedding` | VECTOR(384) | Vector embedding 384 dimensi |
| `document` | TEXT | Teks gabungan (Question + Answer) |
| `cmetadata` | JSONB | Metadata terstruktur (category, tag, question, answer) |

**Indexing Strategy — HNSW (Hierarchical Navigable Small World):**

| Index | Operator | Kegunaan |
|-------|----------|----------|
| `knowledge_base_hnsw_cosine_idx` | `vector_cosine_ops` | Primary search (similarity) |
| `knowledge_base_hnsw_l2_idx` | `vector_l2_ops` | Benchmarking (Euclidean distance) |
| `knowledge_base_hnsw_ip_idx` | `vector_ip_ops` | Benchmarking (Inner Product) |

**Alasan pemilihan HNSW:**
1. **Kecepatan query** — O(log n) vs O(n) untuk brute-force
2. **Tidak butuh training** — Berbeda dengan IVFFlat yang memerlukan `VACUUM` setelah insert
3. **Recall tinggi** — Dengan parameter `m=16, ef_construction=64`, recall mendekati 99%
4. **Mendukung banyak distance metrics** — cosine, L2, inner product dalam satu setup

**Metadata (JSONB):**
- `category`: Kategori utama (FAQ, Policy, Troubleshooting, Contact Information)
- `tag`: Sub-kategori (Account, Payment, Shipping, dll.)
- Digunakan untuk **filtered search** langsung di level database

### 1.3 Knowledge Base Data Preparation

**50 entri** knowledge base yang mencakup 4 kategori:

| Kategori | Jumlah | Contoh Tag |
|----------|--------|------------|
| FAQ | 14 | Account, Payment, Shipping, Product, Promo, Order |
| Policy | 11 | Return, Refund, Privacy, Terms, Warranty, Affiliate |
| Troubleshooting | 14 | Login, App, Payment, Web, Promo, Account, Product, Review |
| Contact Information | 11 | CS, Social, Office, Business, Career, Press, Store, Feedback |

**Proses Data:**
1. **generate_data.py** — Menyiapkan 50 entri data dalam format terstruktur
2. **Preprocessing** — Normalisasi whitespace, pembersihan karakter non-printable
3. **Validasi** — Cek duplikat, kelengkapan kolom, konsistensi kategori
4. **Batch ingestion** — Data di-embed dan disimpan dalam batch (10 per batch) untuk efisiensi

---

## Part 2: Vector Embeddings & Semantic Search

### 2.1 Embedding System Development

**Model: `sentence-transformers/all-MiniLM-L6-v2`**

| Parameter | Nilai |
|-----------|-------|
| Dimensi output | 384 |
| Ukuran model | ~22 MB |
| Bahasa | Multilingual |
| Max sequence length | 256 tokens |

**Alasan pemilihan:**
- **Lightweight** — Cukup ringan untuk berjalan lokal tanpa GPU
- **Cepat** — Inference time <50ms per query
- **Akurat** — Performa tinggi pada Semantic Textual Similarity benchmarks
- **Multilingual** — Mendukung Bahasa Indonesia

**Preprocessing sebelum embedding:**
```python
def preprocess_text(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)          # Collapse whitespace
    text = re.sub(r"[^\w\s\?\!\.\,\-\/\@\'\:\;\(\)]", "", text)  # Clean chars
    return text
```

### 2.2 Vector Database Integration

**Fitur PGVector yang diimplementasi:**
- ✅ Similarity search dengan cosine distance (primary)
- ✅ L2 Euclidean distance (benchmarking)
- ✅ Inner Product distance (benchmarking)
- ✅ HNSW indexing untuk ketiga metrics
- ✅ Metadata filtering (category, tag) via JSONB

### 2.3 Semantic Search Engine (`search_engine.py`)

**Pipeline pencarian:**

```
User Query → Preprocessing → Embedding → Vector Search → Ranking → Results
                                              ↓
                                    + Exact Match Search
                                              ↓
                                       Hybrid Merge
```

**Fitur:**
1. **Semantic Search** — Cosine similarity pada vector embeddings
2. **Exact Match** — SQL ILIKE untuk keyword matching
3. **Hybrid Search** — Weighted combination (70% semantic + 30% exact)
4. **Similarity Threshold** — Score < 0.3 dianggap tidak relevan
5. **Metadata Filtering** — Filter berdasarkan category/tag
6. **Search Analytics** — Logging setiap query ke file dan DataFrame

### 2.4 Benchmarking (`benchmark.py`)

Benchmark membandingkan 3 aspek:

| Benchmark | Deskripsi |
|-----------|-----------|
| **Distance Metrics** | Cosine vs L2 vs Inner Product (latency + accuracy) |
| **Search Methods** | Semantic vs Exact Match vs Hybrid (latency + category recall) |
| **Filtered Search** | Unfiltered vs Category-filtered (latency + relevance score) |

Jalankan: `python benchmark.py`

---

## Part 3: RAG System & Streamlit Interface

### 3.1 RAG Pipeline Implementation

```
User Input → Retriever (PGVector) → Top-K Documents → Prompt Template → Groq LLM → Response
```

**Komponen:**
- **Retriever**: `PGVector.as_retriever(k=3)` dengan optional metadata filter
- **LLM**: Groq `llama3-8b-8192` (temperature=0.2 untuk konsistensi)
- **Prompt**: Custom system prompt khusus customer service Indonesia
- **Chain**: LangChain `create_retrieval_chain` + `create_stuff_documents_chain`

**Prompt customer service:**
```
Anda adalah Intelligent Customer Assistant yang ramah, profesional, dan membantu.
- Jawab dalam Bahasa Indonesia yang sopan
- Gunakan HANYA informasi dari konteks
- Jika tidak tahu, arahkan ke Customer Service
- Jangan membuat-buat jawaban
```

### 3.2 Streamlit User Interface

**Fitur UI:**
- 💬 **Chat Interface** — Format percakapan interaktif
- 🔍 **Real-time Response** — Loading spinner saat proses berjalan
- 📚 **Source References** — Menampilkan sumber dokumen yang digunakan
- 🏷️ **Sidebar Filters** — Filter pencarian berdasarkan kategori dan tag
- 📊 **Session Analytics** — Total query dan rata-rata response time
- 💡 **Quick Actions** — Tombol pertanyaan populer untuk pengguna baru
- 🎨 **Premium Dark Theme** — Gradient header, glassmorphism cards, micro-animations
- ⚙️ **Connection Status** — Indikator status DB, LLM, dan Embeddings

---

## 🚀 Cara Menjalankan

### 1. Jalankan PostgreSQL

```bash
docker-compose up -d
```

### 2. Konfigurasi API Key

Edit file `.env`:
```
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
DATABASE_URL=postgresql://postgres:password@localhost:5432/customer_assistant
```

> Dapatkan API key gratis di [console.groq.com](https://console.groq.com)

### 3. Install Dependencies

```bash
python -m venv venv
.\venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 4. Siapkan Database

```bash
python generate_data.py     # Buat knowledge base CSV (50 entri)
python setup_db.py          # Setup DB, embed data, buat index
```

### 5. Jalankan Aplikasi

```bash
streamlit run app.py
```

### 6. (Opsional) Jalankan Benchmark

```bash
python benchmark.py
```

---

## 📋 Checklist Assignment

### Part 1: Environment Setup & Knowledge Base (25 poin)
- [x] Development environment (VS Code, venv, dependencies)
- [x] API key & database connection via `.env`
- [x] PostgreSQL + PgVector via Docker
- [x] Verifikasi koneksi semua komponen (sidebar status)
- [x] Database schema design (PGVector + JSONB metadata)
- [x] HNSW indexing strategy (3 distance metrics)
- [x] Metadata filtering (category, tag)
- [x] Dokumentasi schema & indexing
- [x] 50 entri knowledge base (FAQ, Policy, Troubleshooting, Contact)
- [x] Preprocessing & cleaning data
- [x] Kategorisasi & tagging
- [x] Validasi kualitas data
- [x] Dokumentasi proses ingestion

### Part 2: Vector Embeddings & Semantic Search
- [x] Embedding model selection + justifikasi (all-MiniLM-L6-v2)
- [x] Batch processing embedding generation
- [x] Text preprocessing & normalization
- [x] PGVector integration + similarity search
- [x] Multiple distance metrics (cosine, L2, inner product)
- [x] Vector indexing optimization (HNSW)
- [x] Category/metadata filtering
- [x] Benchmarking performa pencarian
- [x] Query processing pipeline
- [x] Similarity threshold (0.3)
- [x] Relevance ranking
- [x] Hybrid search (semantic + exact match)
- [x] Search logging & analytics

### Part 3: RAG System & Streamlit Interface (25 poin)
- [x] LangChain + Groq API integration
- [x] Context retrieval dari vector database
- [x] Context + prompt → LLM
- [x] Custom prompt customer service
- [x] Streamlit chatbot interface
- [x] Chat format yang user-friendly
- [x] Real-time response (spinner)
- [x] Loading indicator
