from langchain_community.vectorstores import FAISS

def create_vector_store(chunks, embeddings, save_path="data/vector_store"):
    """
    Create FAISS vector store from document chunks and embeddings.
    
    Args:
        chunks: List of chunked Document objects
        embeddings: Embedding model instance
        save_path: Path to save the vector store
        
    Returns:
        FAISS vector store instance
    """
    print(f"Creating FAISS index with {len(chunks)} chunks...")
    db = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving vector store to {save_path}...")
    db.save_local(save_path)
    
    print("✅ Vector store created and saved successfully!")
    return db

def load_vector_store(embeddings, load_path="data/vector_store"):
    """
    Load existing FAISS vector store.
    
    Args:
        embeddings: Embedding model instance
        load_path: Path to load the vector store from
        
    Returns:
        FAISS vector store instance
    """
    print(f"Loading vector store from {load_path}...")
    db = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector store loaded successfully!")
    return db
