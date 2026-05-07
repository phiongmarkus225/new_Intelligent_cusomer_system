"""
Intelligent Customer Assistant — Streamlit Chatbot
====================================================
A full-featured RAG-based customer service chatbot with:
- Semantic search with PGVector
- LLM response via Groq (LLaMA3)
- Sidebar category/tag filtering
- Streamed response (real-time)
- Source reference display
- Chat history management
- Search analytics dashboard
- Premium dark-themed UI
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from search_engine import SemanticSearchEngine

# ──────────────────────────────────────────────────────────
# Environment & Page Config
# ──────────────────────────────────────────────────────────
load_dotenv()

# Support both .env (local) and Streamlit Secrets (cloud)
def get_secret(key, default=None):
    """Get secret from Streamlit secrets (cloud) or .env (local)."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError, AttributeError):
        return os.getenv(key, default)

DATABASE_URL = get_secret("DATABASE_URL")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

st.set_page_config(
    page_title="Intelligent Customer Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Premium Dark Theme CSS
# ──────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Header Gradient */
.header-title {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0;
    line-height: 1.2;
}

.header-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 4px;
    font-weight: 300;
}

/* Chat Messages */
.stChatMessage {
    border-radius: 12px !important;
    margin-bottom: 8px !important;
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.15);
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* Source Reference Cards */
.source-card {
    background: linear-gradient(135deg, #1e293b 0%, #1a2332 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    transition: border-color 0.2s ease;
}
.source-card:hover {
    border-color: #667eea;
}
.source-tag {
    display: inline-block;
    background: linear-gradient(135deg, #667eea33, #764ba233);
    color: #a5b4fc;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.3px;
    margin-right: 6px;
}
.source-score {
    display: inline-block;
    background: #065f4620;
    color: #34d399;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
section[data-testid="stSidebar"] .stMarkdown p {
    color: #cbd5e1;
}

/* Divider */
.gradient-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, #667eea, #764ba2, transparent);
    border: none;
    margin: 16px 0;
}

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.status-connected {
    background: #065f4630;
    color: #34d399;
    border: 1px solid #065f4650;
}
.status-error {
    background: #7f1d1d30;
    color: #f87171;
    border: 1px solid #7f1d1d50;
}

/* Analytics Table */
.analytics-container {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# Session State Initialization
# ──────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_logs" not in st.session_state:
    st.session_state.search_logs = []
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_response_time" not in st.session_state:
    st.session_state.total_response_time = 0.0


# ──────────────────────────────────────────────────────────
# Cached Resource Initialization
# ──────────────────────────────────────────────────────────
@st.cache_resource
def init_embeddings():
    """Load the embedding model once."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def init_vectorstore(_embeddings):
    """Connect to PGVector store."""
    return PGVector(
        embeddings=_embeddings,
        collection_name="customer_knowledge_base",
        connection=DATABASE_URL,
        use_jsonb=True,
    )


@st.cache_resource
def init_llm():
    """Initialize Groq LLM (LLaMA3-8B)."""
    return ChatGroq(
        temperature=0.2,
        model_name="llama-3.1-8b-instant",
        api_key=GROQ_API_KEY,
    )


@st.cache_resource
def init_search_engine():
    """Initialize the semantic search engine for analytics."""
    return SemanticSearchEngine()


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store, llm, k=3, category_filter=None, tag_filter=None):
    """Build the RAG chain with optional metadata filters using LCEL."""
    search_kwargs = {"k": k}
    filter_dict = {}
    if category_filter and category_filter != "Semua":
        filter_dict["category"] = category_filter
    if tag_filter and tag_filter != "Semua":
        filter_dict["tag"] = tag_filter
    if filter_dict:
        search_kwargs["filter"] = filter_dict

    retriever = vector_store.as_retriever(search_kwargs=search_kwargs)

    system_prompt = (
        "Anda adalah Intelligent Customer Assistant yang ramah, profesional, dan membantu.\n"
        "Tugas Anda adalah menjawab pertanyaan pelanggan dengan akurat berdasarkan konteks yang diberikan.\n\n"
        "ATURAN PENTING:\n"
        "1. Jawab dalam Bahasa Indonesia yang sopan dan profesional.\n"
        "2. Gunakan HANYA informasi dari konteks yang diberikan.\n"
        "3. Jika informasi tidak tersedia dalam konteks, katakan: "
        "'Mohon maaf, saya belum memiliki informasi mengenai hal tersebut. "
        "Silakan hubungi Customer Service kami di 1500-123 atau email support@perusahaan.com untuk bantuan lebih lanjut.'\n"
        "4. Jangan pernah membuat-buat jawaban.\n"
        "5. Berikan jawaban yang ringkas namun lengkap.\n"
        "6. Jika relevan, tawarkan bantuan tambahan.\n\n"
        "Konteks:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # LCEL chain: retrieve docs, format context, send to LLM
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: retriever.invoke(x["input"]),
        )
    )
    answer_chain = prompt | llm | StrOutputParser()
    return retriever, answer_chain


# ──────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────
def render_sidebar():
    """Render the sidebar with filters, status, and analytics."""
    with st.sidebar:
        st.markdown('<p class="header-title" style="font-size:1.4rem;">⚙️ Pengaturan</p>', unsafe_allow_html=True)
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Connection Status
        st.markdown("##### 🔗 Status Koneksi")
        try:
            embeddings = init_embeddings()
            vs = init_vectorstore(embeddings)
            llm = init_llm()
            st.markdown('<span class="status-badge status-connected">✓ Database Connected</span>', unsafe_allow_html=True)
            st.markdown('<span class="status-badge status-connected">✓ LLM Ready</span>', unsafe_allow_html=True)
            st.markdown('<span class="status-badge status-connected">✓ Embeddings Loaded</span>', unsafe_allow_html=True)
            connection_ok = True
        except Exception as e:
            st.markdown(f'<span class="status-badge status-error">✗ Error: {str(e)[:50]}</span>', unsafe_allow_html=True)
            connection_ok = False

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Search Filters
        st.markdown("##### 🏷️ Filter Pencarian")

        categories = ["Semua"]
        tags = ["Semua"]

        if connection_ok:
            try:
                search_eng = init_search_engine()
                categories += search_eng.get_available_categories()
                selected_category = st.selectbox("Kategori", categories, key="filter_category")
                if selected_category != "Semua":
                    tags += search_eng.get_available_tags(selected_category)
                else:
                    tags += search_eng.get_available_tags()
                selected_tag = st.selectbox("Tag", tags, key="filter_tag")
            except Exception:
                selected_category = "Semua"
                selected_tag = "Semua"
                st.selectbox("Kategori", categories, key="filter_category")
                st.selectbox("Tag", tags, key="filter_tag")
        else:
            selected_category = "Semua"
            selected_tag = "Semua"

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Search Parameters
        st.markdown("##### 🔧 Parameter")
        top_k = st.slider("Jumlah Dokumen Referensi (K)", 1, 10, 3, key="top_k")
        show_sources = st.toggle("Tampilkan Sumber Referensi", value=True, key="show_sources")
        search_method = st.radio(
            "Metode Pencarian",
            ["Semantic Search", "Hybrid Search"],
            key="search_method",
            help="Semantic = vector similarity. Hybrid = semantic + exact match.",
        )

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Chat Controls
        st.markdown("##### 💬 Kontrol Chat")
        if st.button("🗑️ Hapus Riwayat Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.search_logs = []
            st.session_state.total_queries = 0
            st.session_state.total_response_time = 0.0
            st.rerun()

        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

        # Session Analytics
        st.markdown("##### 📊 Statistik Sesi")
        total_q = st.session_state.total_queries
        avg_time = (
            st.session_state.total_response_time / total_q if total_q > 0 else 0
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{total_q}</div>
                    <div class="metric-label">Total Query</div>
                </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""<div class="metric-card">
                    <div class="metric-value">{avg_time:.1f}s</div>
                    <div class="metric-label">Avg Response</div>
                </div>""",
                unsafe_allow_html=True,
            )

    return connection_ok


# ──────────────────────────────────────────────────────────
# Main Chat Interface
# ──────────────────────────────────────────────────────────
def main():
    # ── Header ──
    st.markdown('<p class="header-title">🤖 Intelligent Customer Assistant</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="header-subtitle">'
        'Asisten virtual berbasis AI untuk menjawab pertanyaan Anda secara cepat dan akurat '
        '• Powered by RAG + LLaMA3 + PGVector'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    # ── Sidebar ──
    connection_ok = render_sidebar()

    if not connection_ok:
        st.error("⚠️ Sistem tidak dapat terhubung ke database atau LLM. Silakan periksa konfigurasi.")
        st.info("Pastikan:\n1. Docker PostgreSQL sedang berjalan (`docker-compose up -d`)\n2. `setup_db.py` sudah dijalankan\n3. API key Groq sudah diisi di `.env`")
        return

    # ── Initialize RAG components ──
    try:
        embeddings = init_embeddings()
        vector_store = init_vectorstore(embeddings)
        llm = init_llm()
        search_eng = init_search_engine()

        selected_category = st.session_state.get("filter_category", "Semua")
        selected_tag = st.session_state.get("filter_tag", "Semua")
        top_k = st.session_state.get("top_k", 3)

        retriever, answer_chain = build_rag_chain(
            vector_store, llm, k=top_k,
            category_filter=selected_category,
            tag_filter=selected_tag,
        )
    except Exception as e:
        st.error(f"Gagal inisialisasi komponen: {e}")
        return

    # ── Display Chat History ──
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🤖"):
            st.markdown(message["content"])

            # Show sources if available
            if message["role"] == "assistant" and message.get("sources") and st.session_state.get("show_sources", True):
                with st.expander("📚 Sumber Referensi", expanded=False):
                    for src in message["sources"]:
                        cat = src.get("category", "")
                        tag = src.get("tag", "")
                        question = src.get("question", "")
                        st.markdown(
                            f"""<div class="source-card">
                                <span class="source-tag">{cat}</span>
                                <span class="source-tag">{tag}</span>
                                <br><strong style="color:#e2e8f0;font-size:0.85rem;">{question}</strong>
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # ── Chat Input ──
    if prompt := st.chat_input("Ketik pertanyaan Anda di sini... (contoh: Bagaimana cara mereset password?)"):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🤖"):
            start_time = time.time()

            with st.spinner("🔍 Mencari informasi relevan..."):
                try:
                    # Step 1: Retrieve relevant documents
                    context_docs = retriever.invoke(prompt)
                    
                    # Step 2: Format context and invoke LLM
                    context_text = format_docs(context_docs)
                    answer = answer_chain.invoke({"context": context_text, "input": prompt})

                    elapsed = time.time() - start_time

                    # Display answer
                    st.markdown(answer)

                    # Collect source references
                    sources = []
                    for doc in context_docs:
                        sources.append({
                            "category": doc.metadata.get("category", ""),
                            "tag": doc.metadata.get("tag", ""),
                            "question": doc.metadata.get("question", ""),
                            "answer": doc.metadata.get("answer", ""),
                        })

                    # Show sources
                    show_sources = st.session_state.get("show_sources", True)
                    if show_sources and sources:
                        with st.expander("📚 Sumber Referensi", expanded=False):
                            for src in sources:
                                st.markdown(
                                    f"""<div class="source-card">
                                        <span class="source-tag">{src['category']}</span>
                                        <span class="source-tag">{src['tag']}</span>
                                        <br><strong style="color:#e2e8f0;font-size:0.85rem;">{src['question']}</strong>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

                    # Show response time
                    st.caption(f"⚡ Response time: {elapsed:.2f}s")

                    # Save message with sources
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

                    # Update analytics
                    st.session_state.total_queries += 1
                    st.session_state.total_response_time += elapsed
                    st.session_state.search_logs.append({
                        "query": prompt,
                        "response_time": elapsed,
                        "sources_count": len(sources),
                        "category_filter": selected_category,
                        "tag_filter": selected_tag,
                    })

                except Exception as e:
                    error_msg = (
                        "Mohon maaf, terjadi kesalahan pada sistem. "
                        f"Detail: {str(e)[:200]}\n\n"
                        "Silakan coba lagi atau hubungi Customer Service kami di 1500-123."
                    )
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "sources": [],
                    })

    # ── Welcome message if no chat history ──
    if not st.session_state.messages:
        st.markdown("---")
        st.markdown("### 👋 Selamat datang!")
        st.markdown(
            "Saya siap membantu Anda dengan pertanyaan seputar **akun**, **pembayaran**, "
            "**pengiriman**, **kebijakan**, **troubleshooting**, dan informasi lainnya."
        )

        # Quick action buttons
        st.markdown("##### 💡 Pertanyaan Populer:")
        cols = st.columns(2)
        quick_questions = [
            "Bagaimana cara mereset password?",
            "Metode pembayaran apa saja yang tersedia?",
            "Berapa lama waktu pengiriman?",
            "Bagaimana cara menghubungi CS?",
        ]
        for i, q in enumerate(quick_questions):
            with cols[i % 2]:
                if st.button(f"💬 {q}", key=f"quick_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.rerun()


if __name__ == "__main__":
    main()
