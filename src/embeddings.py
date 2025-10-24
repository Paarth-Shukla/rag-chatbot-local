# from langchain_community.embeddings import OpenAIEmbeddings
# from typing import List
# import os

# class EmbeddingGenerator:
#     def __init__(self):
#         api_key = os.getenv("OPENAI_API_KEY")
#         if not api_key:
#             raise ValueError("OpenAI API key not found in environment variables")
        
#         self.embeddings = OpenAIEmbeddings(
#             openai_api_key=api_key,
#             model="text-embedding-ada-002"
#         )
    
#     def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
#         """Generate embeddings for a list of texts"""
#         return self.embeddings.embed_documents(texts)
    
#     def generate_query_embedding(self, query: str) -> List[float]:
#         """Generate embedding for a single query"""
#         return self.embeddings.embed_query(query)
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List

class EmbeddingGenerator:
    def __init__(self):
        # Local, free embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        return self.embeddings.embed_documents(texts)

    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        return self.embeddings.embed_query(query)
