# RAG Chatbot (Local – LLaMA 3 + MiniLM)

A **Retrieval-Augmented Generation (RAG)** chatbot that lets you **chat with your own documents locally** — powered by **LLaMA 3 (Ollama)** for generation and **MiniLM embeddings** for retrieval.  
No API keys. No cloud costs. 100 % offline.

---

## Features

-  Upload multiple documents (PDF, TXT, MD)  
-  Intelligent document chunking and vector indexing  
-  Interactive Streamlit-based chat interface  
-  Context-aware answers with cited sources  
-  Persistent local vector store via **ChromaDB**  
-  Completely local LLM + embedding models (no OpenAI)  

---

## Installation

### Clone the repository
```bash
git clone https://github.com/Paarth-Shukla/rag-chatbot.git
cd rag-chatbot
```

### Create and activate a virtual environment
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Install Ollama and pull LLaMA 3
1. Download **Ollama** → [https://ollama.com/download](https://ollama.com/download)  
2. After installation, pull the model:
   ```bash
   ollama pull llama3
   ```

 Ollama will handle running the local LLaMA 3 model automatically.

---

## Usage

Start the chatbot with:
```bash
python -m streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## How It Works

1. **Document Upload:** Upload PDF/TXT/MD files via sidebar  
2. **Chunking:** Text is split into overlapping sections  
3. **Embedding:** Each chunk is embedded using **MiniLM (Hugging Face)**  
4. **Indexing:** Embeddings are stored locally in **ChromaDB**  
5. **Retrieval:** The most relevant chunks are retrieved per query  
6. **Generation:** **LLaMA 3** (via Ollama) generates an answer using that context  

---

## Project Structure

```
rag-chatbot/
├── app.py                     # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py   # Loads & splits documents
│   ├── embeddings.py           # MiniLM embeddings (local)
│   ├── vector_store.py         # ChromaDB operations
│   └── rag_chain.py            # RAG pipeline (LLaMA 3)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Configuration

Adjust parameters in **`app.py`** as needed:

| Parameter | Description | Default |
|------------|--------------|----------|
| `chunk_size` | Size of text chunks | 1000 |
| `chunk_overlap` | Overlap between chunks | 200 |
| `k` | Number of retrieved docs | 4 |
| `temperature` | LLM creativity | 0.7 |

---

## Tech Stack

- [LangChain](https://github.com/langchain-ai/langchain)  
- [Ollama (LLaMA 3)](https://ollama.com/library/llama3)  
- [Hugging Face MiniLM Embeddings](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)  
- [ChromaDB](https://www.trychroma.com/)  
- [Streamlit](https://streamlit.io/)  

---

## License
MIT License © 2025 Paarth Shukla  

---

## Contributing
Pull requests are welcome!  
If you’d like to improve features or support more local models, open an issue first to discuss proposed changes.


