import streamlit as st
import os
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor
from src.vector_store import VectorStore
from src.rag_chain import RAGChain

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False

def initialize_rag_system():
    '''Initialize the RAG system components'''
    try:
        st.session_state.vector_store = VectorStore()
        st.session_state.rag_chain = RAGChain(st.session_state.vector_store)
        return True
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return False

def process_documents(uploaded_files):
    '''Process uploaded documents'''
    processor = DocumentProcessor()
    
    with st.spinner("Processing documents..."):
        all_chunks = []
        for file in uploaded_files:
            # Save uploaded file temporarily
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Process document
            chunks = processor.process_document(temp_path)
            all_chunks.extend(chunks)
            
            # Clean up
            os.remove(temp_path)
        
        # Add to vector store
        if st.session_state.vector_store is None:
            if not initialize_rag_system():
                return False
        
        st.session_state.vector_store.add_documents(all_chunks)
        st.session_state.documents_processed = True
        
        return True

def main():
    st.title("🤖 RAG Chatbot")
    st.markdown("Chat with your documents using AI")
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload your documents",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or MD files"
        )
        
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                if process_documents(uploaded_files):
                    st.success(f"✅ Processed {len(uploaded_files)} document(s)")
                else:
                    st.error("❌ Error processing documents")
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        k_docs = st.slider("Documents to retrieve", 1, 10, 4)
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("Reset Vector Store"):
            if st.session_state.vector_store:
                st.session_state.vector_store.clear()
                st.session_state.documents_processed = False
                st.success("Vector store cleared")
        
        st.divider()
        
        # Info
        st.markdown("""
        ### How to use:
        1. Upload your documents
        2. Click 'Process Documents'
        3. Start asking questions!
        
        ### Supported formats:
        - PDF files
        - Text files (.txt)
        - Markdown files (.md)
        """)
    
    # Main chat interface
    if not st.session_state.documents_processed:
        st.info("👈 Upload and process documents to start chatting")
        return
    
    # Initialize RAG chain if needed
    if st.session_state.rag_chain is None:
        if not initialize_rag_system():
            return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source)
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag_chain.query(
                        prompt,
                        k=k_docs,
                        temperature=temperature
                    )
                    
                    st.markdown(response["answer"])
                    
                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"]
                    })
                    
                    # Show sources
                    with st.expander("📚 View Sources"):
                        for i, source in enumerate(response["sources"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text(source)
                            st.divider()
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    # Check for API key
    # if not os.getenv("OPENAI_API_KEY"):
    #     st.error("⚠️ OpenAI API key not found. Please set it in the .env file")
    #     st.stop()
    
    main()