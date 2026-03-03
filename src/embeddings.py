from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Initialize HuggingFace embedding model.
    Using sentence-transformers/all-MiniLM-L6-v2:
    - Lightweight and fast
    - 384-dimension vectors
    - Strong for semantic search
    - Perfect for assignment scale
    
    Returns:
        HuggingFaceEmbeddings model instance
    """
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return model
