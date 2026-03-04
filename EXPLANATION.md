# RAG System - Design Decisions

This document explains the key design choices I made for the Gut Health RAG system and the reasoning behind them.

---

## 1. Chunk Size Selection

I used a chunk size of 800 characters with 150-character overlap.

This size keeps enough context together so that concepts like diseases, mechanisms, or bacterial names are not split awkwardly across chunks. Very small chunks (< 500 chars) hurt retrieval because semantic meaning gets fragmented - you might get half a sentence about probiotics in one chunk and the rest in another. Very large chunks (> 1500 chars) reduce precision because unrelated topics get embedded together, making it harder to pinpoint the exact relevant information.

I tested 500, 800, 1000, and 1500 character chunks on the actual documents. The 800-character size gave the most coherent chunks while maintaining good retrieval precision.

The 150-character overlap (~18%) ensures that if a key sentence falls near a chunk boundary, it appears complete in at least one chunk. This prevents information loss at split points.

I used RecursiveCharacterTextSplitter because it splits on natural boundaries (paragraphs, then sentences, then words) rather than cutting text at arbitrary positions.

---

## 2. Reducing Hallucinations

The system uses a retrieval-first architecture (RAG). The LLM only receives the top-3 most relevant chunks before generating an answer - it never sees the query in isolation.

I also used several other techniques:

**Strict prompt engineering**: The prompt explicitly instructs the model to answer ONLY from the provided context and to respond "I don't know based on the provided context" if the information is not present.

**Temperature = 0**: This makes responses deterministic and focused on extracting information rather than generating creative content.

**Citation tracking**: Every answer includes source documents with page numbers, making it easy to verify grounding.

**Trust score**: I calculate a confidence measure based on retrieval quality (how well the chunks match the query). Low trust scores indicate weak retrieval, which can signal potential hallucination risk.


When tested with out-of-scope queries (e.g., "What is gradient descent?"), the system correctly responds "I don't know based on the provided context" even though the LLM knows about gradient descent from its training data. The trust score also correctly flags low confidence (0.377), indicating weak retrieval match.

These multiple layers work together to keep the system grounded in the actual document content.

---

## 3. Vector Database Choice

For this assignment I used FAISS because it is lightweight, fast, and runs locally without any external infrastructure.

Since the dataset is small (3 PDFs, ~280 chunks), FAISS provides excellent similarity search performance. It's also easy to set up and doesn't require API keys or cloud services, which keeps the prototype simple.

For production systems handling 500,000+ documents, I would switch to a managed vector database like Pinecone, Weaviate, or Qdrant. These support:
- Distributed indexing across multiple nodes
- Horizontal scaling as data grows
- High availability (99.9% SLA)
- Built-in metadata filtering
- Real-time updates without downtime

But for this project, FAISS is the right choice - it's fast enough and keeps the architecture simple.

---

## 4. Retrieval Strategy

The current system uses standard semantic similarity search (top-k retrieval with k=3).

I did not use MMR (Maximum Marginal Relevance) because this project focuses on focused medical Q&A, where retrieving multiple chunks that confirm the same fact actually improves reliability. If three chunks all mention that "probiotics help gut health," that's a good signal - not redundancy.

MMR is useful when you want diverse results across different aspects of a topic (like "tell me everything about gut health"). But for specific questions like "What is IBD?" or "How do probiotics work?", semantic similarity gives better, more focused results.

For production, I would add:
- **Hybrid search** (semantic + keyword/BM25): This handles both semantic meaning and exact term matches (like "IBD" or "Lactobacillus")
- **Re-ranking**: A second-stage model to improve precision on the top results
- **MMR as an option**: For exploratory queries where users want broad coverage

But for the current use case, simple semantic search is the right starting point.

---

## 5. Scaling to 500,000+ Documents

The current pipeline separates document ingestion and query processing, which makes scaling straightforward.

To handle large datasets, I would make these changes:

**Replace FAISS with a distributed vector database** (Pinecone, Weaviate, or Qdrant). These support horizontal scaling - you can add more nodes as data grows, and they handle sharding automatically.

**Add metadata filtering**: Attach metadata like publication year, category, author, and status to each chunk. At query time, filter the search space (e.g., only search documents from 2022 onwards). This reduces the search space from 500K to a relevant subset, making retrieval faster and more accurate.

**Batch ingestion pipeline**: Instead of processing documents sequentially, use async batch processing to generate embeddings in parallel. This can speed up ingestion by 10-100x.

**Hybrid search**: Combine semantic search (vector similarity) with keyword search (BM25). This improves both recall (finding relevant docs) and precision (ranking them correctly).

**Caching layer**: Add Redis to cache frequent queries. If someone asks "What is gut microbiome?" multiple times, serve the cached result instantly instead of re-running retrieval and generation.

**Monitoring**: Track query latency, retrieval quality (trust scores over time), error rates, and cost per query. Tools like Prometheus, Grafana, or LangSmith help identify bottlenecks.

These changes would allow the system to scale to hundreds of thousands of documents while keeping retrieval latency under 100ms.

---

## 6. Handling Temporal Updates

Scientific knowledge evolves. New research can replace or contradict older findings.

To handle this, I would add version-aware metadata to documents:

```python
metadata = {
    "publication_year": 2024,
    "version": 2,
    "status": "active",  # or "deprecated"
    "supersedes": "old_study_v1.pdf"
}
```


**Temporal filtering**: During retrieval, prioritize recent documents by filtering on publication year (e.g., only search papers from 2022 onwards) or boosting newer documents in the ranking.

**Soft delete strategy**: Instead of deleting old documents, mark them as "deprecated" in metadata. This keeps an audit trail for compliance and allows rollback if a new study is later retracted.

**Incremental re-indexing**: When new research arrives, only re-embed and update the changed documents instead of rebuilding the entire index. This makes updates fast (minutes instead of hours) and reduces cost.

**Time-decay scoring**: Apply a decay factor to older documents (e.g., 5% decay per year). This naturally prioritizes recent research without hard cutoffs.

**Conflict resolution**: When multiple versions exist, the system prefers the latest version via temporal filtering. Citations include publication dates for transparency, so users can see if information is recent or older.

This approach keeps the system aligned with the latest research while maintaining historical data for audit and analysis purposes.

---

## Summary

This system demonstrates production-ready thinking:

- **Evidence-based decisions**: Chunk size validated through testing, not guesswork
- **Defense in depth**: Multiple layers of hallucination prevention
- **Scalability path**: Clear migration from FAISS to distributed databases with specific optimizations
- **Temporal awareness**: Version control and soft deletes for evolving research
- **Practical trade-offs**: Simple semantic search for prototype, hybrid search for production

The prototype is intentionally simple, but the architecture is designed with scale and reliability in mind.
