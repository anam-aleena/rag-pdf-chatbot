"""
RAG PDF Chatbot — Core Pipeline
Author: Aleena Anam | github.com/anam-aleena

Pipeline: PDF Loading → Chunking → Gemini Embeddings → FAISS → Retrieval → Gemini Generation
"""

import os
from pathlib import Path
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document


# ─── CONFIG ──────────────────────────────────────────────────────────────────

CHUNK_SIZE      = 800
CHUNK_OVERLAP   = 100
RETRIEVAL_K     = 4
MEMORY_WINDOW   = 5
LLM_MODEL       = "gemini-1.5-flash-latest"
EMBEDDING_MODEL = "models/embedding-001"
VECTOR_STORE_PATH = "data/vectorstore"


# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided PDF documents.

Rules:
- Answer ONLY from the provided context. Do not use outside knowledge.
- If the answer is not in the context, say: "I couldn't find that information in the uploaded documents."
- Be concise, clear, and accurate.
- If relevant, mention which part of the document supports your answer.
- For follow-up questions, use the conversation history for context.

Context from documents:
{context}

Conversation history:
{chat_history}

Question: {question}

Answer:"""

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT
)


# ─── 1. PDF LOADING ──────────────────────────────────────────────────────────

def load_pdfs(pdf_path: str) -> list[Document]:
    """
    Load PDFs from a file path or directory.
    Supports single PDF or folder of PDFs.
    """
    path = Path(pdf_path)

    if path.is_file() and path.suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        docs = loader.load()
        print(f"[LOAD] Loaded {len(docs)} pages from {path.name}")

    elif path.is_dir():
        loader = DirectoryLoader(
            str(path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        docs = loader.load()
        print(f"[LOAD] Loaded {len(docs)} pages from {len(list(path.glob('*.pdf')))} PDFs")

    else:
        raise ValueError(f"Invalid path: {pdf_path}. Must be a PDF file or directory.")

    return docs


# ─── 2. TEXT CHUNKING ────────────────────────────────────────────────────────

def chunk_documents(docs: list[Document]) -> list[Document]:
    """
    Split documents into overlapping chunks for better retrieval.

    Design decisions:
    - chunk_size=800: captures enough context per chunk
    - chunk_overlap=100: prevents losing context at boundaries
    - RecursiveCharacterTextSplitter: respects paragraph/sentence boundaries
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    print(f"[CHUNK] Split into {len(chunks)} chunks "
          f"(avg {sum(len(c.page_content) for c in chunks)//len(chunks)} chars)")
    return chunks


# ─── 3. VECTOR STORE ─────────────────────────────────────────────────────────

def build_vectorstore(chunks: list[Document], api_key: str) -> FAISS:
    """
    Embed chunks using Gemini embeddings and store in FAISS.
    FAISS enables fast approximate nearest-neighbour similarity search.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    print(f"[EMBED] Vector store built with {len(chunks)} vectors → saved to {VECTOR_STORE_PATH}")
    return vectorstore


def load_vectorstore(api_key: str) -> Optional[FAISS]:
    """Load existing vector store from disk if available."""
    if Path(VECTOR_STORE_PATH).exists():
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=api_key
        )
        vectorstore = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        print(f"[EMBED] Loaded existing vector store from {VECTOR_STORE_PATH}")
        return vectorstore
    return None


# ─── 4. RAG CHAIN ────────────────────────────────────────────────────────────

def build_rag_chain(vectorstore: FAISS, api_key: str) -> ConversationalRetrievalChain:
    """
    Build the full RAG chain with:
    - Gemini 1.5 Flash LLM (free tier)
    - MMR retrieval for diverse, non-redundant results
    - Sliding window conversation memory
    - Custom system prompt
    """
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=api_key,
        temperature=0.2,
        convert_system_message_to_human=True
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",          # Maximum Marginal Relevance = diverse results
        search_kwargs={
            "k": RETRIEVAL_K,
            "fetch_k": 8,           # Fetch 8, return top 4 diverse ones
            "lambda_mult": 0.7      # Balance relevance vs diversity
        }
    )

    memory = ConversationBufferWindowMemory(
        k=MEMORY_WINDOW,
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        return_source_documents=True,
        verbose=False
    )

    print("[CHAIN] RAG chain built successfully")
    return chain


# ─── 5. FULL PIPELINE ────────────────────────────────────────────────────────

def build_pipeline(pdf_path: str, api_key: str) -> ConversationalRetrievalChain:
    """
    Full pipeline: load PDFs → chunk → embed → build chain.
    Returns a ready-to-use conversational RAG chain.
    """
    docs   = load_pdfs(pdf_path)
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks, api_key)
    chain  = build_rag_chain(vs, api_key)
    return chain


def ask(chain: ConversationalRetrievalChain, question: str) -> dict:
    """
    Ask a question to the RAG chain.
    Returns answer + source documents.
    """
    result = chain({"question": question})
    return {
        "answer":   result["answer"],
        "sources":  [
            {
                "page":    doc.metadata.get("page", "?") + 1,
                "source":  Path(doc.metadata.get("source", "?")).name,
                "excerpt": doc.page_content[:200] + "..."
            }
            for doc in result.get("source_documents", [])
        ]
    }
