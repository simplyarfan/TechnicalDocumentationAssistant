import streamlit as st
import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
from src.document_processor import DocumentProcessor
from src.rag_pipeline import RAGPipeline
from src.database import SessionDatabase
from src.utils import (
    validate_api_key, 
    display_metrics, 
    get_sample_questions, 
    format_sources,
    format_error_message
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SimpleDocs - Technical Documentation Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "processor" not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline()
    
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    
    if "processing_stats" not in st.session_state:
        st.session_state.processing_stats = {}
    
    # Add conversation history
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Add session metadata
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    
    if "session_start_time" not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    
    # Add database
    if "db" not in st.session_state:
        st.session_state.db = SessionDatabase()

def sidebar_configuration():
    """Setup sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        openai_key = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Enter your OpenAI API key to enable Q&A functionality"
        )
        
        # Validate API key
        if openai_key:
            if validate_api_key(openai_key):
                st.success("✅ API Key format looks valid")
            else:
                st.error("❌ Invalid API key format")
        
        st.divider()
        
        # Session info
        st.subheader("📊 Current Session")
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        st.write(f"**Started:** {st.session_state.session_start_time.strftime('%H:%M:%S')}")
        st.write(f"**Questions Asked:** {len(st.session_state.conversation_history)}")
        
        st.divider()
        
        # Database management
        st.subheader("🗃️ Database Management")
        
        # Show database stats
        stats = st.session_state.rag_pipeline.get_stats()
        if stats.get("status") == "ready":
            st.info(f"📊 Documents in database: {stats.get('total_documents', 'Unknown')}")
        
        # Clear database button
        if st.button("🗑️ Clear Database", type="secondary"):
            try:
                if st.session_state.rag_pipeline.clear_database():
                    st.success("Database cleared successfully!")
                    st.session_state.documents_processed = False
                    st.rerun()
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
        
        st.divider()
        
        # Analytics
        st.subheader("📈 Analytics")
        if st.button("View Usage Analytics"):
            analytics = st.session_state.db.get_session_analytics()
            st.json(analytics)
        
        st.divider()
        
        # Processing settings
        st.subheader("⚙️ Processing Settings")
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200, 50)
        
        if st.button("Apply Settings"):
            st.session_state.processor = DocumentProcessor(chunk_size, chunk_overlap)
            st.success("Settings applied!")
        
        return openai_key

def document_upload_section():
    """Handle document upload and processing"""
    st.header("📄 1. Upload Technical Documents")
    st.write("Upload PDF files containing technical documentation, API guides, manuals, etc.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Select one or more PDF files to process. Maximum file size: 50MB each."
    )
    
    if uploaded_files:
        st.write(f"**Selected files:** {len(uploaded_files)}")
        for file in uploaded_files:
            st.write(f"- {file.name} ({file.size / 1024 / 1024:.1f} MB)")
        
        # Process documents button
        if st.button("🔄 Process Documents", type="primary"):
            with st.spinner("Processing documents... This may take a few minutes."):
                try:
                    start_time = time.time()
                    
                    # Process multiple files
                    all_documents, summary = st.session_state.processor.process_multiple_files(uploaded_files)
                    
                    if all_documents:
                        # Create vector store
                        creation_stats = st.session_state.rag_pipeline.create_vectorstore(all_documents)
                        
                        processing_time = time.time() - start_time
                        
                        # Update session state
                        st.session_state.documents_processed = True
                        st.session_state.processing_stats = {
                            "total_files": len(uploaded_files),
                            "successful_files": summary["successful"],
                            "total_chunks": summary["total_chunks"],
                            "processing_time": processing_time
                        }
                        
                        # Save session to database
                        st.session_state.db.save_session({
                            "session_id": st.session_state.session_id,
                            "start_time": st.session_state.session_start_time.isoformat(),
                            "documents_processed": 1,
                            "total_questions": len(st.session_state.conversation_history),
                            "processing_stats": st.session_state.processing_stats
                        })
                        
                        # Display success message and metrics
                        st.success("✅ Documents processed successfully!")
                        display_metrics(
                            summary["successful"], 
                            summary["total_chunks"], 
                            processing_time
                        )
                        
                        # Show any errors
                        if summary["errors"]:
                            st.warning("⚠️ Some files had issues:")
                            for error in summary["errors"]:
                                st.write(f"- {error}")
                    else:
                        st.error("No documents were successfully processed.")
                
                except Exception as e:
                    st.error(f"Error processing documents: {format_error_message(e)}")

def qa_section(openai_key):
    """Handle question-answering with conversation history"""
    st.header("❓ 2. Ask Questions")
    
    if not openai_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to enable Q&A functionality.")
        return
    
    if not st.session_state.documents_processed:
        # Try to load existing vector store
        if st.session_state.rag_pipeline.load_vectorstore():
            st.info("📚 Loaded existing document database.")
        else:
            st.warning("⚠️ Please upload and process documents first.")
            return
    
    # Setup QA chain
    try:
        st.session_state.rag_pipeline.setup_qa_chain(openai_key)
        if not st.session_state.rag_pipeline.qa_chain:
            st.error("❌ Failed to setup Q&A system. Please check your OpenAI API key and billing status.")
            st.info("💡 You can still use 'Similarity Search Only' to test the document retrieval system.")
            return
    except Exception as e:
        st.error(f"Error setting up Q&A system: {format_error_message(e)}")
        st.info("💡 Try using 'Similarity Search Only' instead.")
        return
    
    # Display conversation history
    if st.session_state.conversation_history:
        with st.expander(f"💬 Conversation History ({len(st.session_state.conversation_history)} questions)", expanded=False):
            for i, item in enumerate(st.session_state.conversation_history):
                st.write(f"**Q{i+1}:** {item['question']}")
                st.write(f"**A{i+1}:** {item['answer'][:200]}{'...' if len(item['answer']) > 200 else ''}")
                st.caption(f"Asked at {item['timestamp'][:19]} • {item['search_type']}")
                if i < len(st.session_state.conversation_history) - 1:
                    st.write("---")
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        sample_questions = get_sample_questions()
        col1, col2 = st.columns(2)
        for i, question in enumerate(sample_questions):
            col = col1 if i % 2 == 0 else col2
            if col.button(question, key=f"sample_{i}"):
                st.session_state.current_question = question
    
    # Question input with context awareness
    question = st.text_input(
        "Ask a question about your documents:",
        value=st.session_state.get("current_question", ""),
        placeholder="e.g., Based on our previous discussion, what were the specific recommendations?"
    )
    
    # Search options
    search_type = st.radio(
        "Search type:",
        ["Q&A with AI", "Similarity Search Only"],
        horizontal=True
    )
    
    # Ask question button
    if question and st.button("🔍 Get Answer", type="primary"):
        with st.spinner("Searching for answer..."):
            try:
                # Build context from conversation history
                context_prompt = ""
                if st.session_state.conversation_history:
                    recent_qa = st.session_state.conversation_history[-2:]  # Last 2 Q&As for context
                    context_prompt = "\n\nPrevious conversation context:\n"
                    for qa in recent_qa:
                        context_prompt += f"Q: {qa['question']}\nA: {qa['answer'][:100]}...\n"
                
                if search_type == "Q&A with AI":
                    # Enhance question with context if needed
                    enhanced_question = question
                    if context_prompt and any(word in question.lower() for word in ['this', 'that', 'they', 'it', 'previous', 'mentioned']):
                        enhanced_question = f"{question}{context_prompt}"
                    
                    result = st.session_state.rag_pipeline.ask_question(enhanced_question)
                    
                    # Display answer
                    st.subheader("💬 Answer:")
                    st.write(result["answer"])
                    st.caption(f"Answer based on {result['num_sources']} relevant sections from your documents")
                    
                    # Store in conversation history
                    conversation_item = {
                        "question": question,
                        "answer": result["answer"],
                        "timestamp": datetime.now().isoformat(),
                        "sources_count": result['num_sources'],
                        "search_type": "Q&A with AI"
                    }
                    st.session_state.conversation_history.append(conversation_item)
                    
                    # Save to database
                    st.session_state.db.save_conversation(st.session_state.session_id, conversation_item)
                
                else:
                    # Similarity search
                    results = st.session_state.rag_pipeline.similarity_search(question, k=3)
                    
                    st.subheader("🔍 Most Relevant Sections:")
                    for i, doc in enumerate(results, 1):
                        with st.expander(f"📄 Section {i}"):
                            st.write(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    
                    # Store in conversation history
                    conversation_item = {
                        "question": question,
                        "answer": f"Found {len(results)} relevant sections",
                        "timestamp": datetime.now().isoformat(),
                        "sources_count": len(results),
                        "search_type": "Similarity Search"
                    }
                    st.session_state.conversation_history.append(conversation_item)
                    
                    # Save to database
                    st.session_state.db.save_conversation(st.session_state.session_id, conversation_item)
                
                # Update session in database
                st.session_state.db.save_session({
                    "session_id": st.session_state.session_id,
                    "start_time": st.session_state.session_start_time.isoformat(),
                    "documents_processed": 1 if st.session_state.documents_processed else 0,
                    "total_questions": len(st.session_state.conversation_history),
                    "processing_stats": st.session_state.processing_stats
                })
                
            except Exception as e:
                st.error(f"Error getting answer: {format_error_message(e)}")
    
    # Action buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔄 Clear Question"):
            if "current_question" in st.session_state:
                del st.session_state.current_question
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.rerun()

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("📚 SimpleDocs")
    st.markdown("""
    **Simple, powerful documentation search with AI.**
    Upload technical documents and get instant answers using Retrieval-Augmented Generation (RAG).
    """)
    
    # Sidebar configuration
    openai_key = sidebar_configuration()
    
    # Main content
    document_upload_section()
    
    st.divider()
    
    qa_section(openai_key)
    
    # Footer
    st.divider()
    with st.expander("ℹ️ About This Application"):
        st.markdown("""
        **SimpleDocs** is a RAG-based system that helps you:
        
        - 📄 **Process PDF documents** into searchable chunks
        - 🔍 **Semantic search** using HuggingFace embeddings  
        - 🤖 **AI-powered Q&A** with conversation history
        - 📚 **Session tracking** for usage analytics
        - 💬 **Context-aware responses** using previous conversations
        
        **Technologies used:**
        - **LangChain** for document processing and RAG pipeline
        - **FAISS** for vector storage
        - **HuggingFace** for text embeddings
        - **OpenAI** for language generation
        - **SQLite** for session and conversation storage
        - **Streamlit** for the web interface
        """)

if __name__ == "__main__":
    main()
