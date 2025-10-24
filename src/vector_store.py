# from langchain_community.vectorstores import Chroma
# from langchain_community.embeddings import OpenAIEmbeddings
# import os
# from typing import List

# class VectorStore:
#     def __init__(self, persist_directory: str = "./chroma_db"):
#         self.persist_directory = persist_directory
#         api_key = os.getenv("OPENAI_API_KEY")
        
#         self.embeddings = OpenAIEmbeddings(
#             openai_api_key=api_key,
#             model="text-embedding-ada-002"
#         )
        
#         self.vector_store = Chroma(
#             persist_directory=persist_directory,
#             embedding_function=self.embeddings
#         )
    
#     def add_documents(self, documents: List):
#         """Add documents to the vector store"""
#         self.vector_store.add_documents(documents)
#         self.vector_store.persist()
    
#     def similarity_search(self, query: str, k: int = 4) -> List:
#         """Search for similar documents"""
#         return self.vector_store.similarity_search(query, k=k)
    
#     def clear(self):
#         """Clear the vector store"""
#         import shutil
#         if os.path.exists(self.persist_directory):
#             shutil.rmtree(self.persist_directory)
        
#         self.vector_store = Chroma(
#             persist_directory=self.persist_directory,
#             embedding_function=self.embeddings
#         )
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from typing import List
import shutil

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )

    def add_documents(self, documents: List):
        """Add documents to the vector store"""
        self.vector_store.add_documents(documents)
        self.vector_store.persist()

    def similarity_search(self, query: str, k: int = 4) -> List:
        """Search for similar documents"""
        return self.vector_store.similarity_search(query, k=k)

    def clear(self):
        """Clear the vector store"""
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)

        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
