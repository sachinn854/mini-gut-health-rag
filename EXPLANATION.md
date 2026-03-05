# RAG System - Design Decisions

This document explains the key design choices I made for the Gut Health RAG system and the reasoning behind them.

---

## 1. Chunk Size Selection

I used a chunk size of 800 characters with 150-character overlap.

This size keeps enough context together so that concepts like diseases, mechanisms, or bacterial names are not split awkwardly across chunks. 

**Why not smaller (300-400 chars)?**

While smaller chunks can make retrieval more precise, they create a bigger problem: context fragmentation. In medical/scientific content, explanations often span multiple sentences. For example:

- Paragraph 1: "Gut microbiota imbalance can affect immune function..."
- Paragraph 2: "This imbalance has been linked with IBD and IBS..."

If I use 300-char chunks, these get split. When a user asks "What diseases are linked to gut microbiota?", the retriever might fetch the first paragraph but miss the second one with the actual disease names. The answer becomes incomplete.

**Why not larger (1500+ chars)?**

Very large chunks reduce precision because unrelated topics get embedded together, making it harder to pinpoint the exact relevant information. The embedding becomes diluted with noise.

**Why 800 works:**

I tested 500, 800, 1000, and 1500 character chunks on the actual documents. The 800-character size gave the most coherent chunks while maintaining good retrieval precision. It captures 2-3 complete paragraphs - enough context to preserve meaning without adding noise.

The 150-character overlap (~18%) ensures that if a key sentence falls near a chunk boundary, it appears complete in at least one chunk. This prevents information loss at split points.

I used RecursiveCharacterTextSplitter because it splits on natural boundaries (paragraphs, then sentences, then words) rather than cutting text at arbitrary positions.

---

## 2. Reducing Hallucinations

I designed the system using a **retrieval-first architecture (RAG)**. The LLM never answers the query in isolation; it only receives the **top-3 most relevant chunks retrieved from the vector database** before generating a response. This keeps the model grounded in the actual document content.

I used several additional techniques to reduce hallucinations:

**Strict prompt instructions**

I explicitly instruct the model to answer **only from the provided context**. If the information is not present in the retrieved chunks, the model is instructed to respond with:

*"I don't know based on the provided context."*

**Low temperature (0)**

I set the generation temperature close to **0** so the model focuses on extracting information from the retrieved context instead of producing creative or speculative responses.

**Source citations**

Each generated answer includes **citations with document names and page numbers**. This makes the responses transparent and allows the user to verify the grounding.

**Trust score for retrieval quality**

I implemented a **trust score** based on the similarity distance between the query and the retrieved chunks. Lower similarity leads to a lower trust score, indicating weaker retrieval support for the answer.

To verify that the system stays grounded, I tested it with **out-of-scope queries** (for example: *"What is gradient descent?"*). Since this concept does not appear in the documents, the system correctly responded with:

*"I don't know based on the provided context."*

In those cases, the trust score also dropped significantly (around **0.37**), which correctly indicated weak retrieval alignment.

Overall, these mechanisms help ensure that the model relies on **retrieved document evidence rather than its pre-trained knowledge**, which significantly reduces hallucination risk.

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

For this prototype-scale dataset, FAISS is the most practical choice - it's fast enough and keeps the architecture simple.

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

The current implementation focuses on a simple and reliable prototype, while the architecture keeps a clear path toward production scalability.
