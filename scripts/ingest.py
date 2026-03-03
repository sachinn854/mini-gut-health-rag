import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.loader import load_documents
from src.chunking import chunk_documents
from src.embeddings import get_embedding_model
from src.vectorstore import create_vector_store

print("="*80)
print("DOCUMENT INGESTION PIPELINE")
print("="*80)

print("\n📄 Step 1: Loading documents...")
docs = load_documents()

print("\n✂️  Step 2: Chunking documents...")
chunks = chunk_documents(docs)
print(f"✅ Created {len(chunks)} chunks")

print("\n🧠 Step 3: Loading embedding model...")
embeddings = get_embedding_model()

print("\n💾 Step 4: Creating vector store...")
create_vector_store(chunks, embeddings)

print("\n" + "="*80)
print("✅ INGESTION COMPLETE!")
print("="*80)
print("Vector store saved to: data/vector_store/")
print("Ready for querying!")
