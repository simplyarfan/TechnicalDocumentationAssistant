import os
import time
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from .utils import validate_file_size, validate_file_type, create_file_hash

class DocumentProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document processor with chunking parameters
        
        Args:
            chunk_size: Maximum size of each text chunk
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        self.processed_files = set()  # Track processed file hashes
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and split PDF documents into chunks
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects containing text chunks
        """
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Add metadata about source file
            for doc in documents:
                doc.metadata['source_file'] = os.path.basename(file_path)
                doc.metadata['chunk_size'] = self.chunk_size
            
            # Split documents into chunks
            chunks = self.text_splitter.split_documents(documents)
            
            # Add chunk metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = i
                chunk.metadata['total_chunks'] = len(chunks)
            
            return chunks
            
        except Exception as e:
            raise Exception(f"Error processing PDF {file_path}: {str(e)}")
    
    def process_uploaded_file(self, uploaded_file) -> List[Document]:
        """
        Process uploaded file from Streamlit interface
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            List of Document objects containing text chunks
        """
        # Validate file
        if not validate_file_size(uploaded_file, max_size_mb=50):
            raise ValueError("File size exceeds 50MB limit")
        
        if not validate_file_type(uploaded_file.name):
            raise ValueError("Only PDF files are supported")
        
        # Check for duplicates
        file_content = uploaded_file.getvalue()
        file_hash = create_file_hash(file_content)
        
        if file_hash in self.processed_files:
            raise ValueError("This file has already been processed")
        
        # Create temporary file
        temp_dir = "./temp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_{int(time.time())}_{uploaded_file.name}")
        
        try:
            # Save uploaded file temporarily
            with open(temp_path, "wb") as f:
                f.write(file_content)
            
            # Process the file
            documents = self.load_pdf(temp_path)
            
            # Mark file as processed
            self.processed_files.add(file_hash)
            
            # Add upload metadata
            for doc in documents:
                doc.metadata['uploaded_filename'] = uploaded_file.name
                doc.metadata['file_hash'] = file_hash
                doc.metadata['processed_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return documents
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    def process_multiple_files(self, uploaded_files) -> List[Document]:
        """
        Process multiple uploaded files
        
        Args:
            uploaded_files: List of Streamlit uploaded file objects
            
        Returns:
            Combined list of Document objects from all files
        """
        all_documents = []
        processing_summary = {
            "successful": 0,
            "failed": 0,
            "total_chunks": 0,
            "errors": []
        }
        
        for file in uploaded_files:
            try:
                documents = self.process_uploaded_file(file)
                all_documents.extend(documents)
                processing_summary["successful"] += 1
                processing_summary["total_chunks"] += len(documents)
                
            except Exception as e:
                processing_summary["failed"] += 1
                processing_summary["errors"].append(f"{file.name}: {str(e)}")
        
        return all_documents, processing_summary
    
    def get_processing_stats(self) -> dict:
        """Get statistics about processed documents"""
        return {
            "files_processed": len(self.processed_files),
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap
        }