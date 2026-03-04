# RAG System - Design Decisions & Architecture

This document explains the key design decisions and production-readiness approach for the Gut Health RAG system.

---

## 1. Why This Chunk Size?

### Chosen: 800 characters with 150 overlap

**The Problem:**
- Too small (< 500): Sentences get cut, context is lost, retrieval becomes noisy
- Too large (> 1500): Multiple topics in one chunk, embeddings become diluted, precision drops

**Why 800 Works:**
- Captures 2-3 complete paragraphs
- Maintains semantic coherence for the embedding model (384-dim vectors)
- Balances precision (finding exact info) with recall (getting enough context)

**Why 150 Overlap:**
- Prevents information loss at chunk boundaries
- If a key sentence is split between chunks, the overlap ensures it appears complete in at least one chunk
- ~18% overlap is enough redundancy without bloating the database

**Validation:**
I tested 500, 800, 1000, and 1500 character chunks:
- 500: Too fragmented, poor context
- 800: Best balance - coherent chunks, good retrieval
- 1000+: Multiple topics per chunk, reduced precision

**RecursiveCharacterTextSplitter** splits on natural boundaries (`\n\n` → `\n` → `. ` → ` `), preserving paragraph structure instead of cutting mid-sentence.

---

## 2. How Hallucinations Are Reduced

### Multi-Layer Defense Strategy

**Layer 1: Retrieval-First Architecture**
```
Query → Retrieve Context → Generate Answer (using ONLY retrieved context)
```
The LLM never sees the full query in isolation - it only gets the retrieved chunks as context. This forces grounding.

**Layer 2: Strict Prompt Engineering**
The prompt explicitly instructs:
- "Answer ONLY using information EXPLICITLY stated in the provided context"
- "If context does NOT mention the topic, respond: 'I don't know based on the provided context'"
- "Do NOT use prior knowledge or infer beyond what is written"

This creates clear behavioral boundaries for the LLM.

**Layer 3: Temperature  is close to zero**
Deterministic generation with no creativity. The model extracts from context rather than generating novel content, so it looks like, LLM is using the provided context only.

**Layer 4: Citation Tracking**
Every answer includes source documents with page numbers. This makes answers verifiable and creates accountability.
-> I also verified manually to check the citiation was actually correct or not...it was correct

**Layer 5: Trust Score**
Quantifies retrieval quality using `trust = average(1 / (1 + distance))` for top-3 chunks.
- Low trust score → Weak retrieval → Potential hallucination risk
- Alerts users when the system is less confident

**Validation:**
Tested with out-of-scope queries like "Does gut microbiome affect Alzheimer's?" (not in documents).
Result: System correctly responds "I don't know based on the provided context" instead of hallucinating.

---

## 3. Scaling to 500,000+ Documents

### Current Limitations (Prototype)
- FAISS local file storage
- Single machine, in-memory index
- Limited to ~10-50K documents
- Sequential ingestion (slow)

### Production Architecture

**3.1 Distributed Vector Database**

Replace FAISS with:
- **Pinecone**: Fully managed, auto-scaling, 99.9% SLA
- **Weaviate**: Open-source, hybrid search, self-hosted for cost control
- **Qdrant**: High-performance, Rust-based

These support:
- Horizontal scaling (add nodes as data grows)
- Distributed search (parallel queries across shards)
- High availability (no single point of failure)

**3.2 Metadata Filtering**

Add rich metadata to chunks:
```python
metadata = {
    "publication_year": 2024,
    "category": "clinical_study",
    "author": "Smith et al.",
    "status": "active"
}
```

Filter at query time:
```python
results = db.search(
    query="probiotics",
    filter={"publication_year": {"$gte": 2022}, "status": "active"},
    k=3
)
```

Benefits:
- Reduces search space from 500K to relevant subset
- Faster retrieval
- More relevant results

**3.3 Hybrid Search**

Combine dense (semantic) + sparse (keyword) search:
```python
final_results = 0.7 * vector_similarity + 0.3 * BM25_keyword_match
```

Why:
- Vector search handles semantic meaning
- Keyword search handles exact terms (e.g., "IBD", "microbiome")
- Together they improve recall and precision

**3.4 Batch Ingestion Pipeline**

Current: Sequential processing (slow)

Production: Async batch processing
```python
async def ingest_batch(documents, batch_size=100):
    for batch in chunks(documents, batch_size):
        embeddings = await embed_batch(batch)  # Parallel embedding
        await vector_db.upsert(embeddings)     # Batch upsert
```

Result: 10-100x faster ingestion

**3.5 Caching Layer**

Add Redis for frequent queries:
```python
cache_key = hash(query)
if cache.exists(cache_key):
    return cache.get(cache_key)  # Instant response
```

Benefits:
- Sub-100ms latency for cached queries
- Reduced compute costs
- Better user experience

**3.6 Monitoring**

Track:
- Query latency (p50, p95, p99)
- Retrieval quality (trust scores over time)
- Error rates
- Cost per query

Tools: Prometheus, Grafana, LangSmith

### Performance Comparison

| Metric | Current (FAISS) | Production (Pinecone) |
|--------|-----------------|----------------------|
| Max Documents | 10-50K | 500K+ |
| Query Latency | 100-200ms | 50-100ms |
| Ingestion Speed | 10 docs/sec | 1000+ docs/sec |
| Availability | Single machine | 99.9% SLA |
| Cost | $0 | ~$70-200/month |

---

## 4. Handling Temporal Updates

### The Problem

Research evolves. New studies replace old findings:
- 2020: "Probiotics show limited effect"
- 2024: "Probiotics significantly improve gut health"

If both are in the database, the system might return conflicting information.

### Solution: Version-Aware Architecture

**4.1 Document Versioning**

Add version metadata:
```python
metadata = {
    "publication_year": 2024,
    "version": 2,
    "supersedes": "probiotics_study_v1.pdf",
    "status": "active"  # or "deprecated"
}
```

**4.2 Temporal Filtering**

Prefer recent documents:
```python
results = db.search(
    query=query,
    filter={"publication_year": {"$gte": 2022}},  # Last 2 years
    boost={"publication_year": "desc"}            # Boost newer docs
)
```

**4.3 Soft Delete Strategy**

Don't delete old documents - mark them as deprecated:
```python
old_doc.metadata["status"] = "deprecated"
old_doc.metadata["superseded_by"] = "new_doc_id"

# Filter in retrieval
results = db.search(query, filter={"status": "active"})
```

Benefits:
- Audit trail (compliance, research history)
- Rollback capability if new study is retracted
- Historical analysis

**4.4 Incremental Re-indexing**

Only update changed documents:
```python
new_docs = get_documents_since(last_update_time)
for doc in new_docs:
    embedding = embed(doc)
    vector_db.upsert(doc_id, embedding, metadata)  # Update in place
```

Benefits:
- Faster updates (minutes vs hours)
- Lower cost (only re-embed changed docs)
- Minimal downtime

**4.5 Time-Decay Scoring**

Reduce weight of old documents:
```python
def time_decay_score(base_score, publication_year):
    age = current_year - publication_year
    decay_factor = 0.95 ** age  # 5% decay per year
    return base_score * decay_factor
```

This naturally prioritizes recent research without hard cutoffs.

**4.6 Conflict Resolution**

When multiple versions exist:
1. Prefer latest version (via temporal filtering)
2. Show publication dates in citations (transparency)
3. Optionally show "Updated: 2024" tag
4. Allow users to request historical view if needed

### Implementation

```python
def retrieve_with_temporal(query, prefer_recent=True):
    results = db.search(query, k=10)
    
    if prefer_recent:
        # Filter deprecated docs
        results = [r for r in results 
                   if r.metadata.get("status") != "deprecated"]
        
        # Sort by publication year
        results.sort(
            key=lambda x: x.metadata.get("publication_year", 0),
            reverse=True
        )
    
    return results[:3]
```

---

## Production-Readiness Summary

This system demonstrates production thinking:

1. **Evidence-Based Decisions**: Chunk size validated through testing, not guesswork
2. **Defense in Depth**: Multiple layers of hallucination prevention
3. **Scalability Path**: Clear migration from FAISS → Pinecone/Weaviate with specific optimizations
4. **Temporal Awareness**: Version control and soft deletes for evolving research
5. **Observability**: Trust scores, citations, and monitoring hooks

The prototype is simple, but the architecture is designed with scale and reliability in mind.
