import os
from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from .utils import format_error_message

class RAGPipeline:
    def __init__(self, persist_directory: str = "./faiss_db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize RAG pipeline with FAISS vector database and embeddings"""
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.retriever = None
        
        # Initialize embeddings
        self._setup_embeddings()
    
    def _setup_embeddings(self):
        """Setup HuggingFace embeddings"""
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': False}
            )
        except Exception as e:
            raise Exception(f"Failed to setup embeddings: {str(e)}")
    
    def create_vectorstore(self, documents: List[Document]) -> Dict[str, Any]:
        """Create FAISS vector store from documents"""
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        try:
            self.vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save to disk
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore.save_local(self.persist_directory)
            
            # Setup retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
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
        """Load existing FAISS vector store from disk"""
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
                search_kwargs={"k": 3}
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to load vector store: {str(e)}")
            return False
    
    def setup_qa_chain(self, openai_api_key: str, temperature: float = 0):
        """Setup the question-answering chain"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Create or load vector store first.")
    
    try:
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            temperature=temperature,
            api_key=openai_api_key,  # Updated parameter name
            model="gpt-3.5-turbo",
            max_tokens=500
        )
        
        prompt_template = """Use the following pieces of context from technical documentation to answer the question. 
        If you don't know the answer based on the context provided, just say "I don't have enough information in the provided documentation to answer this question."

        Context:
        {context}

        Question: {question}

        Answer: """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
    except Exception as e:
        raise Exception(f"Failed to setup QA chain: {format_error_message(e)}")
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """Ask a question and get answer with sources"""
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain first.")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            result = self.qa_chain.invoke({"query": question})  # Updated method name
            
            response = {
                "question": question,
                "answer": result["result"],
                "sources": result["source_documents"],
                "num_sources": len(result["source_documents"])
            }
            
            return response
            
        except Exception as e:
            raise Exception(f"Failed to get answer: {format_error_message(e)}")
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search without LLM"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            return results
        except Exception as e:
            raise Exception(f"Similarity search failed: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""
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
        """Clear the vector database"""
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
