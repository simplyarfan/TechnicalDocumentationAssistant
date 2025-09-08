import os
import shutil
import hashlib
from typing import List, Dict, Any
import streamlit as st
from datetime import datetime
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('app.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_file_hash(file_content: bytes) -> str:
    """Create hash of file content to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()

def validate_file_size(file, max_size_mb: int = 50) -> bool:
    """Validate uploaded file size"""
    file_size = len(file.getvalue())
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes

def validate_file_type(filename: str, allowed_types: List[str] = ['.pdf']) -> bool:
    """Validate file extension"""
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in allowed_types

def clean_temp_files(temp_dir: str = "./temp"):
    """Clean up temporary files"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

def format_sources(sources: List[Any]) -> List[Dict[str, str]]:
    """Format source documents for display"""
    formatted_sources = []
    for i, source in enumerate(sources):
        formatted_sources.append({
            "id": i + 1,
            "content": source.page_content[:300] + "..." if len(source.page_content) > 300 else source.page_content,
            "metadata": source.metadata if hasattr(source, 'metadata') else {},
            "full_content": source.page_content
        })
    return formatted_sources

def get_database_stats(vectorstore) -> Dict[str, Any]:
    """Get statistics about the vector database"""
    try:
        collection = vectorstore._collection
        count = collection.count()
        return {
            "total_documents": count,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}

def display_metrics(total_docs: int, total_chunks: int, processing_time: float = None):
    """Display processing metrics in Streamlit"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Documents Processed", total_docs)
    
    with col2:
        st.metric("Text Chunks Created", total_chunks)
    
    with col3:
        if processing_time:
            st.metric("Processing Time", f"{processing_time:.2f}s")

def validate_api_key(api_key: str) -> bool:
    """Basic validation for API key format"""
    if not api_key:
        return False
    
    if api_key.startswith('sk-') and len(api_key) >= 40:
        return True
    
    return False

def get_sample_questions() -> List[str]:
    """Get sample questions for technical documentation"""
    return [
        "How do I get started with this API?",
        "What are the authentication requirements?",
        "What are the rate limits?",
        "How do I handle errors?",
        "What response formats are supported?",
        "How do I make a POST request?",
        "What are the required parameters?",
        "How do I test the API endpoints?"
    ]

def format_error_message(error: Exception) -> str:
    """Format error messages for user display"""
    error_mappings = {
        "RateLimitError": "API rate limit exceeded. Please wait a moment and try again.",
        "AuthenticationError": "Invalid API key. Please check your OpenAI API key.",
        "InvalidRequestError": "Invalid request. Please check your input.",
        "ConnectionError": "Connection error. Please check your internet connection."
    }
    
    error_type = type(error).__name__
    return error_mappings.get(error_type, f"An error occurred: {str(error)}")