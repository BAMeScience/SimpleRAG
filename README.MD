# RAG PDF and Document Retrieval Framework

This repository provides a comprehensive framework for extracting and retrieving information from PDFs and text documents. It combines **OCR capabilities** with **state-of-the-art language models** to process text or image-based PDFs, create semantic embeddings, and retrieve information efficiently using a variety of advanced techniques.

---

## Features

### 1. PDF Text and Image Extraction
- Utilizes **MiniCPM-Llama3-V-2_5**, an advanced model from Hugging Face, for OCR-based text extraction from image-based PDFs.
- Provides simple text extraction for text-based PDFs using the `PyPDF2` library.

Model Reference: [MiniCPM-Llama3-V-2_5 on Hugging Face](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5)

### 2. RAG (Retrieve, Augment, Generate) Workflow
- Built on **LangChain**, a robust framework for building information retrieval systems.
- Employs **Ollama** as a chat and embedding model for advanced natural language interaction and retrieval.

LangChain Reference: [LangChain Documentation](https://www.langchain.com)  
Ollama installation guide: [Ollama Installation and Usage Guide](https://github.com/ollama/ollama)

Ollama available models: [Ollama Installation and Usage Guide](https://ollama.com)


### 3. Multi-Vector and Semantic Retrieval
- Supports semantic chunking of text and multi-vector retrieval for enhanced accuracy.
- Leverages `Chroma` and `FAISS` for vector storage and similarity search.

### 4. Query Answering
-  **`main.py`** Allows users to pass a question (query) along with the pdf document path to retrieve answers based on context.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/balbakri1/SimpleRAG.git
   cd rag-retrieval
