import os
from typing import List, Dict, Any
import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from .utils import get_database_stats, format_error_message

class RAGPipeline:
    def __init__(self, persist_directory: str = "./chroma_db", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize RAG pipeline with vector database and embeddings
        
        Args:
            persist_directory: Directory to store vector database
            embedding_model: HuggingFace model for text embeddings
        """
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
                model_kwargs={'device': 'cpu'},  # Use CPU for compatibility
                encode_kwargs={'normalize_embeddings': False}
            )
        except Exception as e:
            raise Exception(f"Failed to setup embeddings: {str(e)}")
    
    def create_vectorstore(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Create vector store from documents
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Dictionary with creation statistics
        """
        if not documents:
            raise ValueError("No documents provided for indexing")
        
        try:
            # Create vector store
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name="technical_docs"
            )
            
            # Persist the vector store
            self.vectorstore.persist()
            
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
        """
        Load existing vector store from disk
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            if not os.path.exists(self.persist_directory):
                return False
            
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name="technical_docs"
            )
            
            # Setup retriever
            self.retriever = self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            return True
            
        except Exception as e:
            print(f"Failed to load vector store: {str(e)}")
            return False
    
    def setup_qa_chain(self, openai_api_key: str, temperature: float = 0):
        """
        Setup the question-answering chain
        
        Args:
            openai_api_key: OpenAI API key
            temperature: LLM temperature for response generation
        """
        if not self.vectorstore:
            raise ValueError("Vector store not initialized. Create or load vector store first.")
        
        try:
            # Setup LLM
            llm = OpenAI(
                temperature=temperature,
                openai_api_key=openai_api_key,
                max_tokens=500
            )
            
            # Custom prompt template for technical documentation
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
            
            # Create QA chain
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
        """
        Ask a question and get answer with sources
        
        Args:
            question: User question
            
        Returns:
            Dictionary containing answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("QA chain not set up. Call setup_qa_chain first.")
        
        if not question.strip():
            raise ValueError("Question cannot be empty")
        
        try:
            # Get answer from QA chain
            result = self.qa_chain({"query": question})
            
            # Format response
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
        """
        Perform similarity search without LLM
        
        Args:
            query: Search query
            k: Number of similar documents to return
            
        Returns:
            List of similar documents
        """
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
            stats = get_database_stats(self.vectorstore)
            stats.update({
                "embedding_model": self.embedding_model,
                "persist_directory": self.persist_directory,
                "status": "ready"
            })
            return stats
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