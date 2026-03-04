# Mini Gut Health RAG System

A Retrieval-Augmented Generation (RAG) system for gut health and microbiome information, built with LangChain, FAISS, and Groq LLM.

## 🎯 Features

- **Intelligent Chunking**: 800-character chunks with 150-character overlap for optimal context preservation
- **Semantic Search**: FAISS vector database with sentence-transformers embeddings
- **Grounded Generation**: Strict context-only responses using Groq's llama-3.1-8b-instant
- **Source Citations**: Full traceability with document and page number references
- **Trust Scoring**: Quantitative confidence measure based on retrieval quality
- **Conversation Memory**: Maintains chat history for contextual follow-ups

## 📁 Project Structure

```
mini-gut-health-rag/
├── data/
│   ├── raw/                    # Original PDF documents
│   └── vector_store/           # FAISS index files
├── src/
│   ├── loader.py              # PDF loading with PyMuPDF
│   ├── chunking.py            # Text splitting strategy
│   ├── embeddings.py          # HuggingFace embeddings
│   ├── vectorstore.py         # FAISS vector database
│   ├── retriever.py           # Top-k retrieval with scores
│   ├── prompt.py              # Prompt templates
│   ├── generator.py           # Groq LLM integration
│   └── trust_score.py         # Confidence calculation
├── scripts/
│   ├── ingest.py              # Document ingestion pipeline
│   ├── query.py               # Interactive query interface
│   └── test_generation.py    # System testing
├── outputs/
│   └── sample_results.txt     # Example outputs
├── README.md
├── EXPLANATION.md             # Design decisions & rationale
├── requirements.txt
└── .env                       # API keys (not in repo)
```

## 🚀 Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd mini-gut-health-rag
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free Groq API key from: https://console.groq.com/keys

### 5. Add Documents

Place your PDF documents in the `data/` folder:
- Research papers
- Blog articles
- Transcripts (converted to PDF)

## 📊 Usage

**Important**: Make sure your virtual environment is activated before running any commands:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 1: Ingest Documents

Run the ingestion pipeline to process PDFs and create the vector database:

```bash
python scripts/ingest.py
```

**What it does:**
1. Loads PDFs from `data/` folder
2. Splits into 800-character chunks with 150-char overlap
3. Generates embeddings using sentence-transformers
4. Creates FAISS vector index
5. Saves to `data/vector_store/`

**Expected output:**
```
📄 Step 1: Loading documents...
Loading: article_blog.pdf
Loading: research.pdf
Loading: youtube_transcript.pdf
Total pages loaded: 79

✂️  Step 2: Chunking documents...
✅ Created 281 chunks

🧠 Step 3: Loading embedding model...
✅ Model loaded

💾 Step 4: Creating vector store...
✅ Vector store created and saved successfully!
```

### Step 2: Query the System

Start the interactive query interface:

```bash
python scripts/query.py
```

**Available commands:**
- Type your question to get an answer
- `chunks` - Toggle display of retrieved context
- `history` - View conversation history
- `exit` or `quit` - End session

**Example interaction:**
```
💬 You: What is the gut microbiome?

🔍 Retrieving relevant documents...
🧠 Generating answer...

================================================================================
🤖 ANSWER
================================================================================
The gut microbiome refers to the entire population of microorganisms that 
reside within the human gastrointestinal tract, consisting of bacteria that 
outnumber the host's cells by a factor of 10 and encode genes that outnumber 
their host's genes by more than 100 times.

================================================================================
📚 CITATIONS
================================================================================
[1] article_blog.pdf (Page 0)
[2] research.pdf (Page 0)
[3] article_blog.pdf (Page 0)

================================================================================
🎯 TRUST SCORE
================================================================================
Score: 0.677 (Medium confidence)

Interpretation:
⚠️  Medium confidence - Moderate match with retrieved context
```

### Step 3: Test the System

Run comprehensive tests:

```bash
python scripts/test_generation.py
```

This displays:
- Retrieved chunks (full context)
- Generated answer
- Citations
- Trust score with methodology
- System verification

## 📖 Sample Queries

Try these questions:

1. **Basic Definition**
   - "What is the gut microbiome?"
   - "What is gut health?"

2. **Health Conditions**
   - "What diseases are linked to gut microbiota imbalance?"
   - "How does gut microbiome affect IBD?"

3. **Mechanisms**
   - "How do probiotics work?"
   - "What factors affect gut microbiome composition?"

4. **Out-of-Scope** (Tests grounding)
   - "What is gradient descent?"
   - Expected: "I don't know based on the provided context."
   - Result: System correctly refuses to answer + Low trust score (0.377)

## 🎓 Key Design Decisions

### Chunking Strategy
- **Size**: 800 characters (2-3 paragraphs)
- **Overlap**: 150 characters (~18%)
- **Rationale**: Balances semantic coherence with retrieval precision

### Hallucination Prevention
1. Retrieval-first architecture
2. Strict prompt engineering
3. Temperature = 0 (deterministic)
4. Citation tracking
5. Trust score monitoring

### Trust Score Formula
```
trust_score = average(1 / (1 + distance)) for top-3 chunks
```
- High (≥0.8): Strong match
- Medium (0.6-0.8): Moderate match
- Low (<0.6): Weak match

For detailed explanations, see [EXPLANATION.md](EXPLANATION.md)

## 🔧 System Requirements

- Python 3.8+
- 4GB RAM minimum
- Internet connection (for Groq API)
- ~500MB disk space

## 📦 Dependencies

Core libraries:
- `langchain` - RAG framework
- `langchain-community` - Community integrations
- `langchain-openai` - Groq LLM interface
- `faiss-cpu` - Vector database
- `sentence-transformers` - Embeddings
- `pymupdf` - PDF processing
- `python-dotenv` - Environment variables

See `requirements.txt` for complete list.

## 🚀 Scaling to Production

For 500K+ documents, see [EXPLANATION.md](EXPLANATION.md#3-scaling-to-500k-documents) for:
- Distributed vector databases (Pinecone/Weaviate)
- Metadata filtering
- Hybrid search
- Batch ingestion pipelines
- Caching strategies

## 📝 Assignment Deliverables

✅ **Code**: Clean, modular RAG pipeline
✅ **README**: Setup and usage instructions
✅ **Sample Outputs**: See `outputs/sample_results.txt`
✅ **Explanation**: Design rationale in `EXPLANATION.md`

## 🤝 Contributing

This is an assignment project. For questions or issues, please contact the author.

## 📄 License

Educational project - see assignment requirements.

## 👤 Author

Created for RAG Engineer hiring assignment.

---

**Note**: This system demonstrates production-ready thinking in a prototype implementation, balancing simplicity with scalability awareness.

