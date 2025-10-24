# from langchain.chains import RetrievalQA
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# import os

# class RAGChain:
#     def __init__(self, vector_store, model_name: str = "gpt-3.5-turbo"):
#         api_key = os.getenv("OPENAI_API_KEY")
        
#         self.llm = ChatOpenAI(
#             openai_api_key=api_key,
#             model_name=model_name,
#             temperature=0.7
#         )
        
#         self.vector_store = vector_store
        
#         self.prompt_template = """Use the following pieces of context to answer the question at the end. 
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context:
# {context}

# Question: {question}

# Answer: """
        
#         self.PROMPT = PromptTemplate(
#             template=self.prompt_template,
#             input_variables=["context", "question"]
#         )
    
#     def query(self, question: str, k: int = 4, temperature: float = 0.7):
#         """Query the RAG system"""
#         self.llm.temperature = temperature
        
#         docs = self.vector_store.similarity_search(question, k=k)
        
#         context = "\n\n".join([doc.page_content for doc in docs])
        
#         prompt = self.PROMPT.format(context=context, question=question)
#         response = self.llm.predict(prompt)
        
#         return {
#             "answer": response,
#             "sources": [doc.page_content for doc in docs]
#         }
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

class RAGChain:
    def __init__(self, vector_store, model_name: str = "llama3"):
        # Use local LLaMA3 model from Ollama
        self.llm = Ollama(model=model_name, temperature=0.7)
        self.vector_store = vector_store

        self.prompt_template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""

        self.PROMPT = PromptTemplate(
            template=self.prompt_template,
            input_variables=["context", "question"]
        )

    def query(self, question: str, k: int = 4, temperature: float = 0.7):
        """Query the RAG system"""
        self.llm.temperature = temperature
        docs = self.vector_store.similarity_search(question, k=k)

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = self.PROMPT.format(context=context, question=question)
        response = self.llm.predict(prompt)

        return {
            "answer": response,
            "sources": [doc.page_content for doc in docs]
        }
