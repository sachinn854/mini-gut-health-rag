from langchain_community.vectorstores import FAISS
from src.embeddings import get_embedding_model

def get_retriever(k=3, vector_store_path="data/vector_store"):
    """
    Create a retriever from the FAISS vector store.
    
    Args:
        k: Number of top documents to retrieve (default: 3)
        vector_store_path: Path to the saved vector store
        
    Returns:
        Retriever instance configured for top-k search
    """
    embeddings = get_embedding_model()
    
    db = FAISS.load_local(
        vector_store_path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = db.as_retriever(search_kwargs={"k": k})
    return retriever

def retrieve_documents(query, k=3):
    """
    Retrieve top-k relevant documents for a query.
    
    Args:
        query: User query string
        k: Number of documents to retrieve
        
    Returns:
        List of relevant Document objects
    """
    retriever = get_retriever(k=k)
    results = retriever.invoke(query)
    return results

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
