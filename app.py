"""
RAG PDF Chatbot — Streamlit App
Author: Aleena Anam | github.com/anam-aleena

Chat with your PDF documents using Google Gemini + LangChain + FAISS.
Deploy free on Streamlit Cloud.
"""

import streamlit as st
import time
from pathlib import Path
import tempfile
import os

from src.rag_pipeline import (
    load_pdfs, chunk_documents, build_vectorstore,
    build_rag_chain, load_vectorstore, ask
)


# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1B4F72;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .source-box {
        background: #EBF5FB;
        border-left: 4px solid #2196F3;
        padding: 0.75rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #E0E0E0;
    }
    .stChatMessage { border-radius: 12px; }
    div[data-testid="stSidebarContent"] { background: #F5F7FA; }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ───────────────────────────────────────────────────────────

def init_session_state():
    defaults = {
        "chain":          None,
        "chat_history":   [],
        "docs_loaded":    False,
        "num_chunks":     0,
        "num_pages":      0,
        "uploaded_files": [],
        "api_key_set":    False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Setup")
    st.markdown("---")

    # API Key input
    st.markdown("### 🔑 Gemini API Key")
    api_key = st.text_input(
        "Enter your Google Gemini API key",
        type="password",
        placeholder="AIza...",
        help="Get your free key at: makersuite.google.com/app/apikey"
    )

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("✅ API key set")
    else:
        st.warning("⚠️ Enter your Gemini API key to start")
        st.markdown("[Get free API key →](https://makersuite.google.com/app/apikey)")

    st.markdown("---")

    # PDF Upload
    st.markdown("### 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Supports multiple PDFs — all will be indexed together"
    )

    if uploaded_files and api_key:
        if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
            with st.spinner("Processing PDFs..."):
                try:
                    # Save uploaded files to temp directory
                    temp_dir = tempfile.mkdtemp()
                    all_docs = []

                    for uploaded_file in uploaded_files:
                        temp_path = Path(temp_dir) / uploaded_file.name
                        temp_path.write_bytes(uploaded_file.read())
                        docs = load_pdfs(str(temp_path))
                        all_docs.extend(docs)

                    st.info(f"📖 Loaded {len(all_docs)} pages from {len(uploaded_files)} PDF(s)")

                    # Chunk
                    with st.spinner("Splitting into chunks..."):
                        chunks = chunk_documents(all_docs)

                    # Embed
                    with st.spinner("Creating embeddings with Gemini..."):
                        vectorstore = build_vectorstore(chunks, api_key)

                    # Build chain
                    chain = build_rag_chain(vectorstore, api_key)

                    # Update session state
                    st.session_state.chain        = chain
                    st.session_state.docs_loaded  = True
                    st.session_state.num_chunks   = len(chunks)
                    st.session_state.num_pages    = len(all_docs)
                    st.session_state.chat_history = []

                    st.success("✅ Ready to chat!")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Make sure your API key is valid and you have internet access.")

    st.markdown("---")

    # Stats
    if st.session_state.docs_loaded:
        st.markdown("### 📊 Index Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", st.session_state.num_pages)
        with col2:
            st.metric("Chunks", st.session_state.num_chunks)

    # Clear chat
    if st.session_state.chat_history:
        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.chain:
                st.session_state.chain.memory.clear()
            st.rerun()

    st.markdown("---")
    st.markdown("""
    ### ℹ️ How it works
    1. Enter your **Gemini API key** (free)
    2. Upload **PDF documents**
    3. Click **Process PDFs**
    4. **Ask anything** about your documents!

    **Built with:**
    - 🔗 LangChain
    - 🤖 Google Gemini 1.5 Flash
    - 🗄️ FAISS Vector Store
    - 🎈 Streamlit

    **Author:** [Aleena Anam](https://github.com/anam-aleena)
    """)


# ─── MAIN AREA ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">📄 RAG PDF Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Chat with your PDF documents using Google Gemini AI + LangChain + FAISS</p>',
            unsafe_allow_html=True)

# Status banner
if not st.session_state.api_key_set:
    st.info("👈 Enter your Gemini API key in the sidebar to get started. It's free!")
elif not st.session_state.docs_loaded:
    st.info("👈 Upload your PDF files in the sidebar and click **Process PDFs**")
else:
    st.success(f"✅ {st.session_state.num_pages} pages indexed across your documents. Ask me anything!")

st.markdown("---")

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"],
                         avatar="🧑" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])} references)", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Source {i}:</strong> {source['source']} — Page {source['page']}<br>
                        <em>"{source['excerpt']}"</em>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.docs_loaded
):
    # Show user message
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    st.session_state.chat_history.append({
        "role":    "user",
        "content": prompt
    })

    # Generate response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            try:
                result = ask(st.session_state.chain, prompt)
                answer  = result["answer"]
                sources = result["sources"]

                # Stream the answer word by word
                placeholder = st.empty()
                displayed = ""
                for word in answer.split():
                    displayed += word + " "
                    placeholder.markdown(displayed + "▌")
                    time.sleep(0.02)
                placeholder.markdown(displayed)

                # Show sources
                if sources:
                    with st.expander(f"📚 Sources ({len(sources)} references)", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {i}:</strong> {source['source']} — Page {source['page']}<br>
                                <em>"{source['excerpt']}"</em>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                if "quota" in str(e).lower():
                    st.info("💡 You may have hit the free tier limit. Wait a minute and try again.")


# ─── EXAMPLE QUESTIONS ───────────────────────────────────────────────────────

if st.session_state.docs_loaded and not st.session_state.chat_history:
    st.markdown("### 💡 Try asking:")
    col1, col2, col3 = st.columns(3)
    examples = [
        "Summarise the main points of this document",
        "What are the key findings?",
        "What does this document say about [topic]?",
        "List the main recommendations",
        "What is the conclusion?",
        "Compare the different sections"
    ]
    for i, (col, example) in enumerate(zip([col1, col2, col3, col1, col2, col3], examples)):
        with col:
            if st.button(f"💬 {example}", key=f"ex_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": example})
                st.rerun()
