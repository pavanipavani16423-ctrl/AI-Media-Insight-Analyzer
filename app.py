"""
AI Media Insight Analyzer - Streamlit Frontend
===============================================
Main entry point for the application. Provides a clean UI for:
- File upload (PDF/TXT)
- AI-generated insights (summary, sentiment, topics)
- RAG-based Q&A chat interface
"""

import streamlit as st
import os
import time
from backend.utils import extract_text_from_file, validate_api_key
from backend.embeddings import EmbeddingManager
from backend.rag_pipeline import RAGPipeline

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Media Insight Analyzer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .main-header {
        font-family: 'Syne', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        background: linear-gradient(135deg, #0f4c75, #1b262c, #e94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }

    .sub-header {
        font-family: 'DM Sans', sans-serif;
        font-weight: 300;
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
        font-style: italic;
    }

    .insight-card {
        background: linear-gradient(145deg, #ffffff, #f8faff);
        border: 1px solid #e5e9f2;
        border-left: 4px solid #0f4c75;
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 2px 12px rgba(15, 76, 117, 0.06);
    }

    .insight-card h4 {
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        color: #0f4c75;
        margin-bottom: 0.6rem;
        font-size: 0.85rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .insight-card p {
        color: #374151;
        line-height: 1.7;
        font-size: 0.95rem;
        margin: 0;
    }

    .sentiment-positive { border-left-color: #10b981; }
    .sentiment-positive h4 { color: #059669; }

    .sentiment-negative { border-left-color: #ef4444; }
    .sentiment-negative h4 { color: #dc2626; }

    .sentiment-neutral { border-left-color: #f59e0b; }
    .sentiment-neutral h4 { color: #d97706; }

    .topic-pill {
        display: inline-block;
        background: linear-gradient(135deg, #0f4c75, #1b6ca8);
        color: white;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        margin: 0.25rem;
        font-size: 0.82rem;
        font-weight: 500;
        font-family: 'DM Sans', sans-serif;
    }

    .chat-user {
        background: linear-gradient(135deg, #0f4c75, #1b6ca8);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        margin-left: 20%;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    .chat-ai {
        background: #f0f4f8;
        color: #1f2937;
        border-radius: 18px 18px 18px 4px;
        padding: 0.9rem 1.2rem;
        margin: 0.5rem 0;
        margin-right: 10%;
        font-size: 0.93rem;
        line-height: 1.7;
        border: 1px solid #e2e8f0;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: #ecfdf5;
        color: #065f46;
        border: 1px solid #a7f3d0;
        border-radius: 20px;
        padding: 0.3rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 500;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #e5e9f2;
        margin: 2rem 0;
    }

    .stButton > button {
        font-family: 'Syne', sans-serif;
        font-weight: 600;
        letter-spacing: 0.03em;
    }

    .upload-hint {
        color: #9ca3af;
        font-size: 0.82rem;
        font-style: italic;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Initialization ─────────────────────────────────────────────
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "chat_history": [],
        "rag_pipeline": None,
        "insights_generated": False,
        "summary": "",
        "sentiment": "",
        "sentiment_label": "neutral",
        "topics": [],
        "document_loaded": False,
        "file_name": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ─── Sidebar: Configuration ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "Google Gemini API Key",
        type="password",
        placeholder="AIza...",
        help="Get your key at https://aistudio.google.com/app/apikey",
    )

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        if validate_api_key(api_key):
            st.markdown('<span class="status-badge">✅ API Key Valid</span>', unsafe_allow_html=True)
        else:
            st.error("⚠️ Invalid API key format")
    else:
        st.info("🔑 Enter your Gemini API key to begin")

    st.markdown("---")

    # Model info
    st.markdown("### 🤖 Model Info")
    st.markdown("""
    - **LLM:** Gemini 2.5 Flash
    - **Embeddings:** Gemini Embeddings
    - **Vector DB:** ChromaDB
    - **Framework:** LangChain
    """)

    st.markdown("---")

    # Settings
    st.markdown("### 🛠️ Settings")
    chunk_size = st.slider("Chunk Size (tokens)", 300, 1000, 600, 50)
    chunk_overlap = st.slider("Chunk Overlap", 50, 200, 100, 25)
    top_k = st.slider("Top-K Retrieval", 2, 8, 4)

    st.markdown("---")

    # Reset button
    if st.button("🗑️ Reset Session", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🎙️ AI Media Insight Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a transcript · Unlock AI-powered summaries, sentiment, topics & RAG-based Q&A</p>', unsafe_allow_html=True)

# ─── Tab Layout ───────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "💬 Ask Questions (RAG)", "ℹ️ How It Works"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Upload & Analyze
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("#### 📁 Upload Your Transcript")
        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["txt", "pdf"],
            help="Supports news articles, YouTube transcripts, podcast scripts",
        )
        st.markdown('<p class="upload-hint">Supported: .txt, .pdf · Max ~50,000 words</p>', unsafe_allow_html=True)

        if uploaded_file:
            st.success(f"✅ **{uploaded_file.name}** uploaded ({uploaded_file.size / 1024:.1f} KB)")

            # Process button
            analyze_btn = st.button("🚀 Analyze Document", type="primary", use_container_width=True)

            if analyze_btn:
                if not api_key:
                    st.error("⚠️ Please enter your Gemini API key in the sidebar first.")
                else:
                    with st.spinner("🔍 Extracting text..."):
                        raw_text = extract_text_from_file(uploaded_file)

                    if not raw_text or len(raw_text.strip()) < 50:
                        st.error("❌ Could not extract meaningful text from the file.")
                    else:
                        st.info(f"📄 Extracted **{len(raw_text.split()):,} words** from document")

                        # Build RAG pipeline
                        with st.spinner("🧩 Chunking & embedding document..."):
                            try:
                                embed_manager = EmbeddingManager(api_key=api_key)
                                rag = RAGPipeline(
                                    api_key=api_key,
                                    embedding_manager=embed_manager,
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap,
                                    top_k=top_k,
                                )
                                rag.ingest_document(raw_text)
                                st.session_state["rag_pipeline"] = rag
                                st.session_state["document_loaded"] = True
                                st.session_state["file_name"] = uploaded_file.name
                            except Exception as e:
                                st.error(f"❌ Embedding error: {e}")
                                st.stop()

                        # Generate insights
                        with st.spinner("✨ Generating AI insights..."):
                            try:
                                st.session_state["summary"] = rag.generate_summary(raw_text)
                                sentiment_result = rag.analyze_sentiment(raw_text)
                                st.session_state["sentiment"] = sentiment_result["text"]
                                st.session_state["sentiment_label"] = sentiment_result["label"]
                                st.session_state["topics"] = rag.extract_topics(raw_text)
                                st.session_state["insights_generated"] = True
                                st.success("🎉 Analysis complete! See insights on the right →")
                            except Exception as e:
                                st.error(f"❌ Insight generation error: {e}")

    # ── Insights Panel ──────────────────────────────────────────────────────
    with col2:
        st.markdown("#### 🧠 AI Insights")

        if not st.session_state["insights_generated"]:
            st.markdown("""
            <div style="text-align:center; padding: 3rem 1rem; color: #9ca3af;">
                <div style="font-size:3rem">📊</div>
                <div style="font-family:'DM Sans'; font-size:0.95rem; margin-top:0.8rem;">
                    Upload and analyze a document<br>to see insights here
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # File info
            st.markdown(f'<span class="status-badge">📄 {st.session_state["file_name"]}</span>', unsafe_allow_html=True)
            st.markdown("")

            # Summary card
            st.markdown(f"""
            <div class="insight-card">
                <h4>📝 Summary</h4>
                <p>{st.session_state["summary"]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Sentiment card
            label = st.session_state["sentiment_label"].lower()
            sentiment_class = f"sentiment-{label}" if label in ["positive", "negative", "neutral"] else "sentiment-neutral"
            emoji_map = {"positive": "😊", "negative": "😟", "neutral": "😐", "mixed": "🤔"}
            emoji = emoji_map.get(label, "🤔")

            st.markdown(f"""
            <div class="insight-card {sentiment_class}">
                <h4>{emoji} Sentiment — {label.capitalize()}</h4>
                <p>{st.session_state["sentiment"]}</p>
            </div>
            """, unsafe_allow_html=True)

            # Topics card
            topics_html = "".join(
                f'<span class="topic-pill">🏷️ {t}</span>'
                for t in st.session_state["topics"]
            )
            st.markdown(f"""
            <div class="insight-card">
                <h4>🗂️ Key Topics</h4>
                <p>{topics_html}</p>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: RAG Q&A Chat
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### 💬 Ask Anything About Your Document")
    st.markdown('<p class="upload-hint">Powered by Retrieval-Augmented Generation (RAG) — answers are grounded in your document</p>', unsafe_allow_html=True)

    if not st.session_state["document_loaded"]:
        st.warning("⚠️ Please upload and analyze a document in the **Upload & Analyze** tab first.")
    else:
        # Render chat history
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state["chat_history"]:
                if msg["role"] == "user":
                    st.markdown(f'<div class="chat-user">🙋 {msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Input area
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_query = st.text_input(
                "Your question",
                placeholder="e.g. What is the main argument? Who are the key people mentioned?",
                label_visibility="collapsed",
                key="chat_input",
            )
        with col_btn:
            send_btn = st.button("Send ➤", type="primary", use_container_width=True)

        # Suggested questions
        st.markdown("**💡 Try asking:**")
        sugg_cols = st.columns(3)
        suggestions = [
            "What is the main topic?",
            "Summarize the key points",
            "What conclusions are drawn?",
        ]
        for i, sugg in enumerate(suggestions):
            if sugg_cols[i].button(sugg, key=f"sugg_{i}", use_container_width=True):
                user_query = sugg
                send_btn = True

        # Handle query
        if send_btn and user_query.strip():
            st.session_state["chat_history"].append({"role": "user", "content": user_query})

            with st.spinner("🔍 Retrieving relevant context & generating answer..."):
                try:
                    answer = st.session_state["rag_pipeline"].answer_question(user_query)
                    st.session_state["chat_history"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": f"⚠️ Error generating answer: {str(e)}"
                    })
            st.rerun()

        # Clear chat
        if st.session_state["chat_history"]:
            if st.button("🗑️ Clear Chat History"):
                st.session_state["chat_history"] = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: How It Works
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### 🏗️ System Architecture & Concepts")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("""
        ##### 🔄 How RAG Works Here

        **RAG = Retrieval-Augmented Generation**

        Instead of asking the LLM to memorize your document (impossible for long files),
        RAG retrieves the most *relevant chunks* at query time and feeds them as context.

        **Step-by-step:**
        1. 📄 **Ingest** — Your document is split into ~600-token chunks
        2. 🔢 **Embed** — Each chunk → a dense vector (Gemini Embeddings)
        3. 💾 **Store** — Vectors saved in ChromaDB (local vector DB)
        4. ❓ **Query** — Your question is also embedded
        5. 🔍 **Retrieve** — Top-K most similar chunks are fetched
        6. 🤖 **Generate** — Gemini 2.5 Flash answers using those chunks as context

        This means answers are always **grounded in your document**, not hallucinated.
        """)

        st.markdown("""
        ##### 🗄️ Why ChromaDB?

        - **Lightweight** — runs fully in-memory or on disk, no server needed
        - **Fast** — uses HNSW indexing for sub-millisecond similarity search
        - **LangChain-native** — first-class integration
        - **Open source** — no API costs for vector storage
        - **Scalable** — can persist to disk for production use

        For production at scale, you'd swap to **Pinecone**, **Weaviate**, or **Qdrant**.
        """)

    with col_b:
        st.markdown("""
        ##### 📦 Project Structure

        ```
        project/
        ├── app.py               ← Streamlit UI (this file)
        ├── backend/
        │   ├── rag_pipeline.py  ← Core RAG + Gemini LLM calls
        │   ├── embeddings.py    ← Gemini embedding + ChromaDB
        │   └── utils.py         ← File parsing helpers
        ├── requirements.txt
        └── README.md
        ```
        """)

        st.markdown("""
        ##### 🚀 Scalability Improvements

        | Current | Production Upgrade |
        |---|---|
        | ChromaDB in-memory | Pinecone / Weaviate cloud |
        | Single file upload | Batch ingestion pipeline |
        | Streamlit UI | FastAPI + React frontend |
        | Local API key | Secret Manager / Vault |
        | No auth | OAuth2 / JWT auth |
        | Single user | Multi-tenant architecture |
        | Sync processing | Async job queue (Celery) |
        | No caching | Redis semantic cache |
        """)

    st.markdown("---")
    st.markdown("""
    ##### 🧠 AI Functions Implemented

    | Function | Input | Output |
    |---|---|---|
    | `generate_summary()` | Full document text | Concise paragraph summary |
    | `analyze_sentiment()` | Full document text | Label + explanation |
    | `extract_topics()` | Full document text | List of key topics |
    | `answer_question()` | User query | RAG-grounded answer |
    """)