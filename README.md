# Technical Documentation Assistant

A powerful RAG (Retrieval-Augmented Generation) system for querying technical documentation using LangChain, OpenAI, and Streamlit.

## Features

- **PDF Document Processing**: Upload and process multiple PDF files
- **Semantic Search**: Find relevant information using HuggingFace embeddings
- **AI-Powered Q&A**: Get detailed answers using OpenAI's language models
- **Source Tracking**: See exactly where answers come from in your documents
- **Interactive Web Interface**: Easy-to-use Streamlit interface
- **Vector Database**: Persistent storage with ChromaDB

## Technologies Used

- **LangChain** - Document processing and RAG pipeline
- **OpenAI** - Language model for generating answers
- **HuggingFace** - Text embeddings (sentence-transformers)
- **ChromaDB** - Vector database for document storage
- **Streamlit** - Web interface
- **PyPDF2** - PDF processing

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Git (for cloning)

## 🛠Installation

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd technical-docs-assistant
