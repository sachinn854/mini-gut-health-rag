from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
        temperature=0.1,  # Deterministic responses, no creativity
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