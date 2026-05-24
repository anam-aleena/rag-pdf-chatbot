"""
RAG PDF Chatbot — Streamlit App
Author: Aleena Anam | github.com/anam-aleena
Chat with your PDF documents using Google Gemini + LangChain + FAISS
"""

import streamlit as st
import time
import os
import tempfile
from pathlib import Path

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Force sidebar text to be visible */
    section[data-testid="stSidebar"] {
        background-color: #1E1E2E !important;
        min-width: 300px !important;
    }
    section[data-testid="stSidebar"] * {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] .stTextInput input {
        background-color: #2D2D3F !important;
        color: #FFFFFF !important;
        border: 1px solid #555 !important;
    }
    section[data-testid="stSidebar"] .stButton button {
        background-color: #7C3AED !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] .stFileUploader {
        background-color: #2D2D3F !important;
        border-radius: 8px !important;
        padding: 8px !important;
    }
    /* Main area */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #7C3AED;
        margin-bottom: 4px;
    }
    .main-sub {
        font-size: 1rem;
        color: #888;
        margin-bottom: 20px;
    }
    .source-card {
        background: #1E293B;
        border-left: 4px solid #7C3AED;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #CBD5E1;
    }
    /* Status badges */
    .badge-ready {
        background: #14532D;
        color: #86EFAC;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    /* Chat styling */
    .stChatMessage {
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── SESSION STATE ───────────────────────────────────────────────────────────

def init_state():
    defaults = {
        "chain": None,
        "chat_history": [],
        "docs_loaded": False,
        "num_chunks": 0,
        "num_pages": 0,
        "api_key_valid": False,
        "api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ─── LAZY IMPORTS (only when needed) ─────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_pipeline_functions():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.prompts import PromptTemplate
    return (PyPDFLoader, RecursiveCharacterTextSplitter, FAISS,
            GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI,
            ConversationalRetrievalChain, ConversationBufferWindowMemory, PromptTemplate)


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🤖 RAG PDF Chatbot")
    st.markdown("---")

    # API Key
    st.markdown("### 🔑 Gemini API Key")
    st.markdown("Get your **free** key 👉 [Google AI Studio](https://aistudio.google.com/app/apikey)")

    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="AIzaSy...",
        label_visibility="collapsed"
    )

    if api_key_input:
        st.session_state.api_key = api_key_input
        st.session_state.api_key_valid = True
        os.environ["GOOGLE_API_KEY"] = api_key_input
        st.success("✅ API key ready!")
    else:
        st.warning("⚠️ Please enter your Gemini API key above to start.")

    st.markdown("---")

    # PDF Upload
    st.markdown("### 📄 Upload PDF Files")
    st.markdown("Upload one or more PDFs to chat with")

    uploaded_files = st.file_uploader(
        "Choose PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and st.session_state.api_key_valid:
        file_names = [f.name for f in uploaded_files]
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for name in file_names:
            st.markdown(f"📄 {name}")

        if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
            (PyPDFLoader, RecursiveCharacterTextSplitter, FAISS,
             GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI,
             ConversationalRetrievalChain, ConversationBufferWindowMemory,
             PromptTemplate) = get_pipeline_functions()

            progress = st.progress(0, text="Starting...")

            try:
                # Step 1 — Load PDFs
                progress.progress(20, text="📖 Loading PDFs...")
                all_docs = []
                temp_dir = tempfile.mkdtemp()

                for uploaded_file in uploaded_files:
                    temp_path = Path(temp_dir) / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.read())
                    loader = PyPDFLoader(str(temp_path))
                    all_docs.extend(loader.load())

                # Step 2 — Chunk
                progress.progress(40, text="✂️ Splitting into chunks...")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=800, chunk_overlap=100
                )
                chunks = splitter.split_documents(all_docs)

                # Step 3 — Embed
                progress.progress(65, text="🧠 Creating Gemini embeddings...")
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=st.session_state.api_key
                )
                vectorstore = FAISS.from_documents(chunks, embeddings)

                # Step 4 — Build chain
                progress.progress(85, text="⛓️ Building RAG chain...")

                PROMPT = PromptTemplate(
                    input_variables=["context", "chat_history", "question"],
                    template="""You are a helpful AI assistant. Answer questions based ONLY on the provided documents.
If the answer is not in the documents, say: "I couldn't find that in the uploaded documents."

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:"""
                )

                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=st.session_state.api_key,
                    temperature=0.2,
                    convert_system_message_to_human=True
                )

                memory = ConversationBufferWindowMemory(
                    k=5, memory_key="chat_history",
                    return_messages=True, output_key="answer"
                )

                chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(
                        search_type="mmr",
                        search_kwargs={"k": 4, "fetch_k": 8}
                    ),
                    memory=memory,
                    combine_docs_chain_kwargs={"prompt": PROMPT},
                    return_source_documents=True,
                    verbose=False
                )

                progress.progress(100, text="✅ Ready!")
                time.sleep(0.5)
                progress.empty()

                st.session_state.chain = chain
                st.session_state.docs_loaded = True
                st.session_state.num_chunks = len(chunks)
                st.session_state.num_pages = len(all_docs)
                st.session_state.chat_history = []
                st.success(f"✅ {len(all_docs)} pages indexed! Ask me anything.")
                st.rerun()

            except Exception as e:
                progress.empty()
                st.error(f"Error: {str(e)}")
                if "api" in str(e).lower() or "key" in str(e).lower():
                    st.info("💡 Check your API key is correct.")
                elif "quota" in str(e).lower():
                    st.info("💡 API quota reached. Wait 1 minute and try again.")

    elif uploaded_files and not st.session_state.api_key_valid:
        st.warning("⚠️ Enter your API key first, then process PDFs.")

    # Stats
    if st.session_state.docs_loaded:
        st.markdown("---")
        st.markdown("### 📊 Index Stats")
        col1, col2 = st.columns(2)
        col1.metric("Pages", st.session_state.num_pages)
        col2.metric("Chunks", st.session_state.num_chunks)

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
**How it works:**
1. 🔑 Enter Gemini API key
2. 📄 Upload PDF files
3. 🚀 Click Process PDFs
4. 💬 Ask anything!

**Built by:** [Aleena Anam](https://github.com/anam-aleena)
    """)


# ─── MAIN AREA ───────────────────────────────────────────────────────────────

st.markdown('<p class="main-title">📄 RAG PDF Chatbot</p>', unsafe_allow_html=True)
st.markdown('<p class="main-sub">Chat with your PDF documents using Google Gemini AI · LangChain · FAISS Vector Search</p>',
            unsafe_allow_html=True)

if not st.session_state.api_key_valid:
    st.info("👈 **Step 1:** Enter your free Gemini API key in the sidebar")
elif not st.session_state.docs_loaded:
    st.info("👈 **Step 2:** Upload your PDF files and click **Process PDFs**")
else:
    st.markdown(
        f'<span class="badge-ready">✅ {st.session_state.num_pages} pages indexed — Ready to chat!</span>',
        unsafe_allow_html=True
    )

st.markdown("---")

# Chat history display
for msg in st.session_state.chat_history:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) referenced", expanded=False):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source {i}:</strong> {src['source']} — Page {src['page']}<br>
                        <em>{src['excerpt']}</em>
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.docs_loaded
):
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                result  = st.session_state.chain({"question": prompt})
                answer  = result["answer"]
                sources = [
                    {
                        "page":    doc.metadata.get("page", 0) + 1,
                        "source":  Path(doc.metadata.get("source", "doc")).name,
                        "excerpt": doc.page_content[:180] + "..."
                    }
                    for doc in result.get("source_documents", [])
                ]

                # Stream response
                placeholder = st.empty()
                displayed = ""
                for word in answer.split():
                    displayed += word + " "
                    placeholder.markdown(displayed + "▌")
                    time.sleep(0.015)
                placeholder.markdown(displayed)

                if sources:
                    with st.expander(f"📚 {len(sources)} source(s) referenced", expanded=False):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>Source {i}:</strong> {src['source']} — Page {src['page']}<br>
                                <em>{src['excerpt']}</em>
                            </div>
                            """, unsafe_allow_html=True)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:
                err = str(e)
                st.error(f"Error: {err}")
                if "quota" in err.lower():
                    st.info("💡 API quota hit. Wait 1 minute and try again.")
                elif "api" in err.lower():
                    st.info("💡 API key issue. Check your key in the sidebar.")

# Example prompts when no chat yet
if st.session_state.docs_loaded and not st.session_state.chat_history:
    st.markdown("### 💡 Try asking:")
    examples = [
        "Summarise this document",
        "What are the key points?",
        "What is the main conclusion?",
        "List the main topics covered",
        "What does this say about [topic]?",
        "Give me a brief overview"
    ]
    cols = st.columns(3)
    for i, (col, ex) in enumerate(zip(cols * 2, examples)):
        with col:
            if st.button(f"💬 {ex}", key=f"ex{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": ex})
                st.rerun()
