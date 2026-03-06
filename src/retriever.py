from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model

def retrieve_with_scores(query, k=3, vector_store_path="data/vector_store"):
    """
    Retrieve documents with similarity scores for trust calculation.
    
    Args:
        query: User query string
        k: Number of documents to retrieve
        vector_store_path: Path to the saved vector store
        
    Returns:
        List of tuples (Document, similarity_score)
        Lower score = higher similarity (FAISS uses distance)
    """
    embeddings = get_embedding_model()
    
    db = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Get documents with similarity scores
    results = db.similarity_search_with_score(query, k=k)
    return results