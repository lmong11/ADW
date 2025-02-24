# 📄 Intelligent Document Processing with Ollama RAG

This repository provides an intelligent document workflow (ADW) system utilizing **Ollama** as the core LLM with **4-bit quantization** and **KV cache optimization** for efficient retrieval-augmented generation (RAG). The system enables **document parsing, retrieval, summarization, translation, visualization, and code generation** with a user-friendly **Streamlit** interface.

## 🚀 Features
- **4-bit Quantization & KV Cache**: Optimized for memory efficiency and faster inference
- **Ollama-based RAG**: Uses Ollama embeddings and chat-based retrieval
- **Document Parsing**: Extract text from PDF files using OCR
- **Vector Database**: FAISS-powered semantic search
- **Multi-functional AI Agent**: Summarization, translation, visualization, and code generation
- **User-friendly UI**: Interactive **Streamlit** web application

## 📂 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/lmong11/ADW.git
   cd your-repo
   ```
2. Create a virtual environment (optional but recommended):
   ```sh
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Ensure Ollama is installed and pull the DeepSeek-R1 model:
   ```sh
   ollama pull deepseek-r1
   ```

## 🛠️ Usage
Run the Streamlit application:
```sh
streamlit run adw.py
```

## 🧠 How It Works
1. **Upload a PDF file**: The system extracts text using OCR.
2. **Index the content**: FAISS creates a vectorized knowledge base.
3. **Query the system**: Users can search and retrieve relevant document passages.
4. **Generate insights**: AI Agent assists with summarization, translations, visualizations, and code generation.

## 📌 Technologies Used
- **Python**
- **Ollama** (for LLM inference)
- **LangChain** (for RAG and AI agent integration)
- **FAISS** (for vector database search)
- **PyMuPDF & Tesseract OCR** (for PDF text extraction)
- **Streamlit** (for UI)


## 📬 Contact
For questions, reach out to **your-email@example.com** or create an issue in the repository.

