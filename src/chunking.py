from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents(docs):
    """
    Split documents into smaller chunks for better embedding and retrieval.
    
    Args:
        docs: List of Document objects
        
    Returns:
        List of chunked Document objects with preserved metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(docs)
    return chunks
