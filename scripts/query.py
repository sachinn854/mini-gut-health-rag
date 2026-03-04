import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve_with_scores
from src.prompt import build_prompt
from src.generator import generate_answer
from src.trust_score import calculate_trust_score, get_trust_level
from langchain_core.messages import HumanMessage, AIMessage

# Initialize conversation memory
chat_history = []

# Get absolute path to vector store
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")

def query_rag_system(query, show_chunks=False, chat_history=[]):
    """
    Query the RAG system and return answer with sources and trust score.
    
    Args:
        query: User question
        show_chunks: Whether to display retrieved chunks
        chat_history: Previous conversation messages (optional)
        
    Returns:
        tuple: (answer, docs, scores, trust_score)
    """
    # Retrieve relevant documents with scores
    print("\n🔍 Retrieving relevant documents...")
    results = retrieve_with_scores(query, k=3, vector_store_path=VECTOR_STORE_PATH)
    
    # Separate documents and scores
    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]
    
    # Calculate trust score
    trust_score = calculate_trust_score(scores)
    
    # Show chunks if requested
    if show_chunks:
        print("\n" + "="*80)
        print("📄 RETRIEVED CHUNKS (Context for LLM)")
        print("="*80)
        for i, (doc, score) in enumerate(results, 1):
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            page = doc.metadata.get('page', 'N/A')
            print(f"\n--- CHUNK {i} (Similarity: {1/(1+score):.3f}) ---")
            print(f"Source: {source} (Page {page})")
            print(f"Length: {len(doc.page_content)} chars")
            print(f"\nContent:")
            print(doc.page_content)
            print("-" * 80)
    
    # Build prompt with context
    print("\n🧠 Generating answer...")
    prompt = build_prompt(query, docs, chat_history)
    
    # Generate answer
    answer = generate_answer(prompt)
    
    return answer, docs, scores, trust_score

def main():
    """
    Interactive query loop with conversation memory.
    """
    print("="*80)
    print("GUT HEALTH RAG SYSTEM")
    print("="*80)
    print("\nInitializing system...")
    
    print("✅ System ready!")
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - Type 'chunks' to toggle showing retrieved chunks")
    print("  - Type 'history' to see conversation history")
    print("  - Type 'exit' or 'quit' to end")
    print("="*80)
    
    show_chunks = False
    
    while True:
        # Get user input
        query = input("\n💬 You: ").strip()
        
        if not query:
            continue
            
        if query.lower() in ['exit', 'quit']:
            print("\n👋 Goodbye!")
            break
        
        if query.lower() == 'chunks':
            show_chunks = not show_chunks
            status = "ON" if show_chunks else "OFF"
            print(f"\n✅ Chunk display is now {status}")
            continue
            
        if query.lower() == 'history':
            print("\n📜 Conversation History:")
            print("="*80)
            for i, msg in enumerate(chat_history):
                if isinstance(msg, HumanMessage):
                    print(f"\n💬 You: {msg.content}")
                elif isinstance(msg, AIMessage):
                    print(f"\n🤖 Assistant: {msg.content}")
            print("="*80)
            continue
        
        try:
            # Query the system
            answer, docs, scores, trust_score = query_rag_system(query, show_chunks, chat_history)
            
            # Store in memory
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=answer))
            
            # Display answer
            print("\n" + "="*80)
            print("🤖 ANSWER")
            print("="*80)
            print(answer)
            
            # Display citations
            print("\n" + "="*80)
            print("📚 CITATIONS")
            print("="*80)
            for i, doc in enumerate(docs, 1):
                source = os.path.basename(doc.metadata.get('source', 'unknown'))
                page = doc.metadata.get('page', 'N/A')
                print(f"[{i}] {source} (Page {page})")
            
            # Display trust score
            print("\n" + "="*80)
            print("🎯 TRUST SCORE")
            print("="*80)
            trust_level = get_trust_level(trust_score)
            print(f"Score: {trust_score:.3f} ({trust_level} confidence)")
            print(f"\nInterpretation:")
            if trust_score >= 0.8:
                print("✅ High confidence - Strong semantic match with retrieved context")
            elif trust_score >= 0.6:
                print("⚠️  Medium confidence - Moderate match with retrieved context")
            else:
                print("❌ Low confidence - Weak match with retrieved context")
            
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            print("Please try again.")

if __name__ == "__main__":
    main()
