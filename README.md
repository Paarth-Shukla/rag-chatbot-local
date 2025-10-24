# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your documents using AI.

## Features

-  Upload multiple documents (PDF, TXT, MD)
-  Intelligent document chunking and indexing
-  Interactive chat interface
-  Context-aware responses with source citations
-  Persistent vector storage with ChromaDB
-  Easy to use Streamlit interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Paarth-Shukla/rag-chatbot.git
cd rag-chatbot
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## How It Works

1. **Document Upload**: Upload your documents through the sidebar
2. **Processing**: Documents are split into chunks and embedded
3. **Indexing**: Embeddings are stored in ChromaDB for fast retrieval
4. **Query**: Ask questions about your documents
5. **Retrieval**: Relevant chunks are retrieved using vector similarity
6. **Generation**: GPT generates answers based on retrieved context

## Project Structure

```
rag-chatbot/
├── app.py                 # Main Streamlit application
├── src/
│   ├── __init__.py
│   ├── document_processor.py  # Document loading and chunking
│   ├── embeddings.py          # Embedding generation
│   ├── vector_store.py        # ChromaDB operations
│   └── rag_chain.py           # RAG pipeline
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Configuration

You can modify the following parameters in `app.py`:
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)
- `k`: Number of relevant chunks to retrieve (default: 4)
- `temperature`: LLM temperature (default: 0.7)

## License

MIT License

## Contributing

Pull requests are welcome! For major changes, please open an issue first.
