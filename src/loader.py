from langchain_community.document_loaders import PyMuPDFLoader
import os

def load_documents_lazy(data_path="data"):
    """
    Lazy load PDF documents from the specified directory.
    Uses generator to avoid loading all documents into memory at once.
    PyMuPDFLoader handles multi-column layouts better than basic loaders.
    
    Args:
        data_path: Path to directory containing PDF files
        
    Yields:
        Document objects with page_content and metadata
    """
    pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]
    
    for file in pdf_files:
        path = os.path.join(data_path, file)
        print(f"Loading: {file}")
        loader = PyMuPDFLoader(path)
        
        # Lazy load - yield documents one by one
        for doc in loader.lazy_load():
            yield doc

def load_documents(data_path="data"):
    """
    Load all PDF documents from the specified directory.
    For backward compatibility - loads all at once.
    
    Args:
        data_path: Path to directory containing PDF files
        
    Returns:
        List of Document objects with page_content and metadata
    """
    documents = list(load_documents_lazy(data_path))
    print(f"\nTotal pages loaded: {len(documents)}")
    return documents

