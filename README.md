# RAG PDF Chatbot — LangChain + Google Gemini + FAISS

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.20-green)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-orange?logo=google&logoColor=white)](https://makersuite.google.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-blue)](https://faiss.ai)
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-brightgreen)](LICENSE)

> **Chat with any PDF document using Google Gemini AI — ask questions, get accurate answers with source citations.**  
> Upload your PDFs, ask anything, and get grounded responses with page references — no hallucinations.

🚀 **[Live Demo →](https://your-app-name.streamlit.app)** *(deploy your own in 5 minutes — free!)*

---

## What It Does

| Feature | Description |
|---|---|
| 📄 Multi-PDF support | Upload and chat with multiple PDFs simultaneously |
| 🔍 Semantic search | FAISS vector store finds relevant passages even without exact keywords |
| 🤖 Gemini 1.5 Flash | Fast, accurate responses using Google's free LLM |
| 💬 Conversation memory | Remembers last 5 exchanges for follow-up questions |
| 📚 Source citations | Every answer shows the exact page and excerpt it came from |
| 🌊 Streaming responses | Answers appear word-by-word like ChatGPT |
| 🆓 100% Free | Google Gemini free tier — no credit card needed |

---

## RAG Pipeline Architecture

```
User Question
     ↓
Gemini Embedding (question → vector)
     ↓
FAISS Similarity Search (MMR — diverse top-4 chunks retrieved)
     ↓
Context Assembly (retrieved chunks + conversation history)
     ↓
Gemini 1.5 Flash (generates grounded answer from context)
     ↓
Answer + Source Citations → User
```

**Key design decisions:**
- **MMR retrieval** (Maximum Marginal Relevance) — retrieves diverse results, avoids redundant chunks
- **Chunk size 800, overlap 100** — balances context preservation with retrieval precision
- **Sliding window memory (k=5)** — maintains conversation context without overflowing context window
- **Temperature 0.2** — keeps answers factual and consistent, not overly creative
- **Custom system prompt** — forces model to answer ONLY from documents, prevents hallucination

---

## Project Structure

```
rag-pdf-chatbot/
│
├── app.py                    # Streamlit UI — main application
├── src/
│   └── rag_pipeline.py       # Core RAG pipeline (load → chunk → embed → retrieve → generate)
├── data/
│   └── sample_pdfs/          # Add your PDF files here
├── .streamlit/
│   └── secrets.toml.example  # Streamlit Cloud secrets template
├── .env.example              # Environment variables template
├── requirements.txt
└── README.md
```

---

## Quick Start — Run Locally

### 1. Clone & install
```bash
git clone https://github.com/anam-aleena/rag-pdf-chatbot.git
cd rag-pdf-chatbot
pip install -r requirements.txt
```

### 2. Get your FREE Gemini API key
Go to **[makersuite.google.com/app/apikey](https://makersuite.google.com/app/apikey)** → Create API key → Copy it

### 3. Set up environment
```bash
cp .env.example .env
# Edit .env and add your API key:
# GOOGLE_API_KEY=AIza...
```

### 4. Run the app
```bash
streamlit run app.py
```
→ Opens at `http://localhost:8501`

### 5. Use the app
1. Enter your Gemini API key in the sidebar
2. Upload one or more PDF files
3. Click **Process PDFs**
4. Ask anything about your documents!

---

## Deploy FREE on Streamlit Cloud

1. **Fork this repo** on GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → New app
3. Select your forked repo → `app.py`
4. Go to **App Settings → Secrets** → add:
```toml
GOOGLE_API_KEY = "your_key_here"
```
5. Click **Deploy** — live in 2 minutes!

---

## Example Use Cases

| Use Case | Example Question |
|---|---|
| Research papers | "What methodology did the authors use?" |
| Legal documents | "What are the key terms and conditions?" |
| Financial reports | "What was the revenue growth year over year?" |
| Textbooks | "Explain the concept of [topic] from chapter 3" |
| Contracts | "What are the termination clauses?" |
| Technical docs | "How do I configure [feature]?" |

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini 1.5 Flash (free tier) |
| Embeddings | Google Generative AI Embeddings |
| Vector Store | FAISS (Facebook AI Similarity Search) |
| RAG Framework | LangChain 0.1.20 |
| PDF Loading | PyPDF + LangChain Document Loaders |
| Memory | ConversationBufferWindowMemory |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## Author

**Aleena Anam** — AI/ML Engineer & GenAI Developer  
📧 anamaleena0@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/aleena-anam-2056a4368) | [GitHub](https://github.com/anam-aleena)

---

## License

MIT License — free to use, modify, and distribute.
