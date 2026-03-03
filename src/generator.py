from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from src.prompt import get_rag_prompt
import os

load_dotenv()

def get_llm():
    """
    Initialize Groq LLM using ChatOpenAI interface.
    Using llama-3.1-8b-instant for fast, quality responses.
    Temperature set to 0 for deterministic, grounded responses.
    
    Returns:
        ChatOpenAI instance configured for Groq
    """
    model = ChatOpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.1-8b-instant",
        temperature=0.15,  # Deterministic responses, no creativity
    )
    return model

def generate_answer(prompt_text):
    """
    Generate answer using Groq LLM.
    Simple function for non-chain usage.
    
    Args:
        prompt_text: Formatted prompt string
        
    Returns:
        Generated answer string
    """
    llm = get_llm()
    response = llm.invoke(prompt_text)
    return response.content

def create_rag_chain(retriever):
    """
    Create RAG chain with retriever, prompt, and LLM.
    Uses LangChain runnables for chaining.
    
    Args:
        retriever: Document retriever instance
        
    Returns:
        Runnable RAG chain
    """
    llm = get_llm()
    prompt = get_rag_prompt()
    
    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])
    
    # Create RAG chain using runnables
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: []  # Empty for now, will add memory later
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain
