import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import retrieve_with_scores
from src.prompt import build_prompt
from src.generator import generate_answer
from src.trust_score import calculate_trust_score, get_trust_level, explain_trust_score

# Get absolute path to vector store
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "data", "vector_store")

print("="*80)
print("TESTING RAG SYSTEM")
print("="*80)

print("\n📝 Note: This requires GROQ_API_KEY in .env file")
print("Get your token from: https://console.groq.com/keys")

# Check if token exists
from dotenv import load_dotenv
load_dotenv()

token = os.getenv("GROQ_API_KEY")
if not token or token == "your_groq_api_key_here":
    print("\n⚠️  WARNING: Groq API key not set!")
    print("\nTo test generation:")
    print("1. Get free API key from https://console.groq.com/keys")
    print("2. Add to .env file: GROQ_API_KEY=your_key")
    print("3. Run this script again")
    
else:
    print("\n✅ Groq API key found! Testing RAG system...")
    
    query = "What is the relationship between gut microbiota and IBD?"
    
    print(f"\n💬 Query: {query}")
    print("\n🔍 Retrieving documents with scores...")
    results = retrieve_with_scores(query, k=3, vector_store_path=VECTOR_STORE_PATH)
    
    # Separate documents and scores
    docs = [doc for doc, score in results]
    scores = [score for doc, score in results]
    
    print(f"✅ Retrieved {len(docs)} documents")
    
    # Calculate trust score
    trust_score = calculate_trust_score(scores)
    trust_level = get_trust_level(trust_score)
    
    # Show retrieved chunks
    print("\n" + "="*80)
    print("📄 RETRIEVED CHUNKS (Context for LLM)")
    print("="*80)
    for i, (doc, score) in enumerate(results, 1):
        source = os.path.basename(doc.metadata.get('source', 'unknown'))
        page = doc.metadata.get('page', 'N/A')
        similarity = 1 / (1 + score)
        print(f"\n--- CHUNK {i} (Similarity: {similarity:.3f}) ---")
        print(f"Source: {source} (Page {page})")
        print(f"Distance Score: {score:.4f}")
        print(f"Length: {len(doc.page_content)} chars")
        print(f"\nContent:")
        print(doc.page_content)
        print("-" * 80)
    
    print("\n🧠 Generating answer with Groq (temperature=0)...")
    prompt = build_prompt(query, docs)
    
    try:
        answer = generate_answer(prompt)
        
        print("\n" + "="*80)
        print("🤖 GENERATED ANSWER")
        print("="*80)
        print(answer)
        
        print("\n" + "="*80)
        print("📚 CITATIONS")
        print("="*80)
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get('source', 'unknown'))
            page = doc.metadata.get('page', 'N/A')
            print(f"[{i}] {source} (Page {page})")
        
        print("\n" + "="*80)
        print("🎯 TRUST SCORE")
        print("="*80)
        print(f"Score: {trust_score:.3f} ({trust_level} confidence)")
        print(f"\nInterpretation:")
        if trust_score >= 0.8:
            print("✅ High confidence - Strong semantic match")
        elif trust_score >= 0.6:
            print("⚠️  Medium confidence - Moderate match")
        else:
            print("❌ Low confidence - Weak match")
        
        print("\n" + "="*80)
        print("📊 TRUST SCORE METHODOLOGY")
        print("="*80)
        print(explain_trust_score())
        
        print("\n" + "="*80)
        print("✅ SYSTEM VERIFICATION")
        print("="*80)
        print("✓ Retrieval working - Top-3 chunks retrieved")
        print("✓ Generation working - Answer synthesized from context")
        print("✓ Citations provided - Source tracking enabled")
        print("✓ Trust score calculated - Confidence measure available")
        print("\nCompare the answer with retrieved chunks to verify grounding.")
        
    except Exception as e:
        print(f"\n❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\nMake sure your Groq API key is valid.")
