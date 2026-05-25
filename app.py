"""
RAG PDF Chatbot — Streamlit App
Author: Aleena Anam | github.com/anam-aleena

NO LangChain dependency for LLM — uses Gemini API directly.
HuggingFace embeddings + FAISS for retrieval.
This approach is more stable and faster.
"""

import streamlit as st
import time
import os
import tempfile
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    section[data-testid="stSidebar"] { background-color: #1E1E2E !important; min-width: 300px !important; }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stTextInput input { background-color: #2D2D3F !important; color: #FFFFFF !important; border: 1px solid #555 !important; }
    section[data-testid="stSidebar"] .stButton button { background-color: #7C3AED !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
    .main-title { font-size: 2.2rem; font-weight: 700; color: #7C3AED; margin-bottom: 4px; }
    .main-sub { font-size: 1rem; color: #888; margin-bottom: 20px; }
    .source-card { background: #1E293B; border-left: 4px solid #7C3AED; padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 6px 0; font-size: 0.85rem; color: #CBD5E1; }
    .badge-ready { background: #14532D; color: #86EFAC; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ────────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "docs_loaded": False,
        "chunks": [],
        "embeddings_matrix": None,
        "chat_history": [],
        "num_pages": 0,
        "api_key_valid": False,
        "api_key": "",
        "embed_model": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── CORE FUNCTIONS ───────────────────────────────────────────────────────────

def load_and_chunk_pdfs(uploaded_files):
    """Load PDFs and split into chunks."""
    from pypdf import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    all_text_chunks = []
    total_pages = 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80
    )

    temp_dir = tempfile.mkdtemp()
    for uf in uploaded_files:
        tp = Path(temp_dir) / uf.name
        tp.write_bytes(uf.read())
        reader = PdfReader(str(tp))
        total_pages += len(reader.pages)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                sub_chunks = splitter.split_text(text)
                for chunk in sub_chunks:
                    all_text_chunks.append({
                        "text": chunk,
                        "source": uf.name,
                        "page": page_num + 1
                    })

    return all_text_chunks, total_pages


@st.cache_resource(show_spinner=False)
def load_embed_model():
    """Load HuggingFace embedding model once and cache it."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("all-MiniLM-L6-v2")


def embed_chunks(chunks, model):
    """Create embeddings for all chunks."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=False)
    return embeddings


def retrieve_relevant_chunks(question, chunks, embeddings_matrix, model, top_k=4):
    """Find most relevant chunks for the question."""
    q_embedding = model.encode([question])
    # Cosine similarity
    norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
    norm_matrix = embeddings_matrix / (norms + 1e-10)
    q_norm = q_embedding / (np.linalg.norm(q_embedding) + 1e-10)
    similarities = norm_matrix @ q_norm.T
    top_indices = np.argsort(similarities.flatten())[::-1][:top_k]
    return [chunks[i] for i in top_indices]


def ask_gemini(question, relevant_chunks, chat_history, api_key):
    """Call Gemini API directly — no LangChain."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    # Build context from retrieved chunks
    context = "\n\n---\n\n".join([
        f"[Source: {c['source']}, Page {c['page']}]\n{c['text']}"
        for c in relevant_chunks
    ])

    # Build conversation history
    history_text = ""
    for msg in chat_history[-4:]:  # last 4 exchanges
        role = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    prompt = f"""You are a helpful AI assistant. Answer questions ONLY based on the provided document excerpts below.
If the answer is not in the documents, say: "I couldn't find that in the uploaded documents."
Be concise, clear, and accurate. Mention the source page when relevant.

DOCUMENT EXCERPTS:
{context}

CONVERSATION HISTORY:
{history_text}

USER QUESTION: {question}

ANSWER:"""

    # Try different model names in order
    model_names = [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash"
    ]

    last_error = None
    for model_name in model_names:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, None
        except Exception as e:
            last_error = e
            if "quota" in str(e).lower():
                return None, "quota"
            continue

    return None, str(last_error)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 RAG PDF Chatbot")
    st.markdown("---")

    st.markdown("### 🔑 Gemini API Key")
    st.markdown("Get free key 👉 [AI Studio](https://aistudio.google.com/app/apikey)")

    api_key_input = st.text_input(
        "API Key", type="password",
        placeholder="AIzaSy...", label_visibility="collapsed"
    )

    if api_key_input and api_key_input.startswith("AIza"):
        st.session_state.api_key = api_key_input
        st.session_state.api_key_valid = True
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.success("✅ API key set!")
    elif api_key_input:
        st.error("❌ Must start with 'AIza'")
    else:
        st.warning("⚠️ Enter your Gemini API key")

    st.markdown("---")
    st.markdown("### 📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        "PDFs", type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.session_state.api_key_valid:
        for f in uploaded_files:
            st.markdown(f"📄 {f.name}")

        if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
            prog = st.progress(0, text="Loading embedding model...")
            try:
                # Load embed model
                embed_model = load_embed_model()
                st.session_state.embed_model = embed_model
                prog.progress(25, text="📖 Reading PDFs...")

                # Load and chunk
                chunks, total_pages = load_and_chunk_pdfs(uploaded_files)
                prog.progress(55, text=f"🧠 Embedding {len(chunks)} chunks...")

                # Embed
                emb_matrix = embed_chunks(chunks, embed_model)
                prog.progress(95, text="✅ Almost done...")

                st.session_state.chunks = chunks
                st.session_state.embeddings_matrix = emb_matrix
                st.session_state.num_pages = total_pages
                st.session_state.docs_loaded = True
                st.session_state.chat_history = []

                prog.progress(100, text="✅ Ready!")
                time.sleep(0.3)
                prog.empty()
                st.success(f"✅ {total_pages} pages, {len(chunks)} chunks ready!")
                st.rerun()

            except Exception as e:
                prog.empty()
                st.error(f"Error: {str(e)}")

    elif uploaded_files and not st.session_state.api_key_valid:
        st.warning("⚠️ Enter API key first")

    if st.session_state.docs_loaded:
        st.markdown("---")
        st.markdown("### 📊 Stats")
        c1, c2 = st.columns(2)
        c1.metric("Pages", st.session_state.num_pages)
        c2.metric("Chunks", len(st.session_state.chunks))

    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown("---")
    st.markdown("""
**Steps:**
1. 🔑 Enter Gemini API key
2. 📄 Upload PDFs
3. 🚀 Process PDFs
4. 💬 Ask anything!

**Built by:** [Aleena Anam](https://github.com/anam-aleena)
    """)


# ─── MAIN AREA ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">📄 RAG PDF Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">Chat with your PDFs · Google Gemini AI · FAISS · HuggingFace Embeddings</p>', unsafe_allow_html=True)

if not st.session_state.api_key_valid:
    st.info("👈 **Step 1:** Enter your Gemini API key in the sidebar")
elif not st.session_state.docs_loaded:
    st.info("👈 **Step 2:** Upload PDFs and click **Process PDFs**")
else:
    st.markdown(f'<span class="badge-ready">✅ {st.session_state.num_pages} pages indexed — Ready to chat!</span>', unsafe_allow_html=True)

st.markdown("---")

# Chat history display
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s)", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-card"><strong>#{i}</strong> {src["source"]} — Page {src["page"]}<br><em>{src["excerpt"]}</em></div>',
                        unsafe_allow_html=True
                    )

# Chat input
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.docs_loaded
):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🔍 Searching documents..."):
            # Retrieve relevant chunks
            relevant = retrieve_relevant_chunks(
                prompt,
                st.session_state.chunks,
                st.session_state.embeddings_matrix,
                st.session_state.embed_model,
                top_k=4
            )

        with st.spinner("✍️ Generating answer with Gemini..."):
            answer, error = ask_gemini(
                prompt, relevant,
                st.session_state.chat_history,
                st.session_state.api_key
            )

        if answer:
            placeholder = st.empty()
            displayed = ""
            for word in answer.split():
                displayed += word + " "
                placeholder.markdown(displayed + "▌")
                time.sleep(0.015)
            placeholder.markdown(displayed)

            sources = [
                {
                    "source": r["source"],
                    "page": r["page"],
                    "excerpt": r["text"][:150] + "..."
                }
                for r in relevant
            ]
            with st.expander(f"📚 {len(sources)} source(s) used", expanded=False):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f'<div class="source-card"><strong>#{i}</strong> {src["source"]} — Page {src["page"]}<br><em>{src["excerpt"]}</em></div>',
                        unsafe_allow_html=True
                    )

            st.session_state.chat_history.append({
                "role": "assistant", "content": answer, "sources": sources
            })

        elif error == "quota":
            st.warning("⏳ API quota reached. Wait 1 minute and ask again.")
        else:
            st.error(f"Gemini error: {error}")
            st.info("💡 Make sure your API key has Gemini API enabled at aistudio.google.com")

# Example prompts
if st.session_state.docs_loaded and not st.session_state.chat_history:
    st.markdown("### 💡 Try asking:")
    examples = [
        "Summarise this document",
        "What are the key points?",
        "What is the main conclusion?",
        "List the main topics",
        "Explain [topic] from this document",
        "Give me a brief overview"
    ]
    cols = st.columns(3)
    for i, (col, ex) in enumerate(zip(cols * 2, examples)):
        with col:
            if st.button(f"💬 {ex}", key=f"ex{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": ex})
                st.rerun()
