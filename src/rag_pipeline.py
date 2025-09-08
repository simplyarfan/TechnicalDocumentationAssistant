import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from .utils import format_error_message

class RAGPipeline:
    def __init__(self, persist_directory: str = "./faiss_db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
        except Exception as e:
            raise Exception(f"Failed to setup embeddings: {str(e)}")
    
    def create_vectorstore(self, documents: List[Document]) -> Dict[str, Any]:
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore.save_local(self.persist_directory)
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Increased for better context
            )
            
            return {
                "status": "success",
                "documents_indexed": len(documents),
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            raise Exception(f"Failed to create vector store: {str(e)}")
    
    def load_vectorstore(self) -> bool:
        try:
            if not os.path.exists(self.persist_directory):
                return False
            
            self.vectorstore = FAISS.load_local(
                self.persist_directory,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to load vector store: {str(e)}")
            return False
    
    def setup_qa_chain(self, openai_api_key: str, temperature: float = 0):
        """Setup QA chain with improved prompt and retrieval"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Create or load vector store first.")
        
        try:
            import openai
            openai.api_key = openai_api_key
            
            from langchain.llms import OpenAI
            
            llm = OpenAI(
                temperature=temperature,
                openai_api_key=openai_api_key,
                max_tokens=300  # Increased for better answers
            )
            
            # Improved prompt template
            prompt_template = """You are a helpful assistant that answers questions based on the provided context. Use the context below to answer the question accurately and completely.

Context:
{context}

Question: {question}

Instructions:
- Answer based only on the information provided in the context
- Be specific and detailed in your response
- If the answer requires information from multiple parts of the context, combine them
- If you cannot find the answer in the context, say "I cannot find this information in the provided document"

Answer:"""
            
            PROMPT = PromptTemplate(
                template=prompt_template, 
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": PROMPT}
            )
            
        except Exception as e:
            print(f"Error in setup_qa_chain: {str(e)}")
            self.qa_chain = None
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        if not self.qa_chain:
            raise ValueError("QA chain not set up properly.")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            result = self.qa_chain({"query": question})
            
            response = {
                "question": question,
                "answer": result["result"],
                "sources": result["source_documents"],
                "num_sources": len(result["source_documents"])
            }
            
            return response
            
        except Exception as e:
            raise Exception(f"Failed to get answer: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            raise Exception(f"Similarity search failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        if not self.vectorstore:
            return {"status": "not_initialized"}
        
        try:
            index_size = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 0
            
            return {
                "total_documents": index_size,
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "status": "ready"
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def clear_database(self):
        try:
            if os.path.exists(self.persist_directory):
                import shutil
                shutil.rmtree(self.persist_directory)
                self.vectorstore = None
                self.qa_chain = None
                self.retriever = None
                return True
        except Exception as e:
            raise Exception(f"Failed to clear database: {str(e)}")
        return False