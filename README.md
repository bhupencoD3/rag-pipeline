# Project RAG – Retrieval-Augmented Generation Learning Pipeline

This repository is a step-by-step learning and experimentation project on **Retrieval-Augmented Generation (RAG)**. The goal is to build an end-to-end RAG pipeline from scratch, including data ingestion, parsing, chunking, embedding, and vector storage, to create a robust knowledge base for retrieval and generation.

The project is modularly structured into three core stages:

1. **Data Ingestion & Parsing** – converting raw files into structured LangChain Document objects.
2. **Vector Embeddings** – transforming documents into numerical embeddings.
3. **Vector Stores** – persisting and managing embeddings for fast similarity-based retrieval.

---

## Project Structure

```
Project RAG/
├── 0-DataIngestParsing/
│   ├── dataingestion.ipynb
│   ├── dataingestion-pdf.ipynb
│   ├── dataingestionDoc.ipynb
│   ├── dataingestionCsvExcel.ipynb
│   ├── dataingestionJson.ipynb
│   ├── dataingestionSQL.ipynb
│   └── data/
│       ├── text_files/
│       ├── pdf/
│       ├── word_files/
│       ├── structured_files/
│       ├── json_files/
│       └── databases/
├── 1-VectorEmbeddings/
│   ├── embedding.ipynb
│   └── openai-embeddings.ipynb
└── 2-Vector_store/
    ├── chromaDB.ipynb
    ├── faiss.ipynb
    ├── otherVectorStore.ipynb
    ├── pinecone.ipynb
└── 3-AdvancedChunking/
    └── semantic_chunking.ipynb
```

---

## Section 1: Data Ingestion and Parsing

This stage processes diverse document formats (Text, PDF, Word, CSV/Excel, JSON, SQL) into structured, chunked, metadata-rich LangChain Document objects.

**Key points:**

* Multiple loaders: TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredCSVLoader, JSONLoader, SQLDatabase.
* Intelligent chunking strategies using RecursiveCharacterTextSplitter and TokenTextSplitter.
* Metadata attached to each chunk (source, author, file type, page number, etc.).
* Data cleaning and validation (PDF OCR, whitespace normalization, filtering small chunks).

**Outcome:** Consistent document structures ready for embeddings.

---

## Section 2: Vector Embeddings

Transforms text chunks into high-dimensional semantic vectors suitable for retrieval.

### Notebooks

1. **embedding.ipynb**

   * Explained embeddings with simple 2D visualizations.
   * Manual cosine similarity computation.
   * Used HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2).

2. **openai-embeddings.ipynb**

   * Used OpenAI Embeddings (text-embedding-3-small).
   * Computed similarity scores for semantic matching.
   * Implemented a mini semantic search demo.

**Outcome:** Generated embeddings ready for vector storage.

---

## Section 3: Vector Stores (Updated)

Manages document embeddings for **fast similarity-based retrieval** using multiple vector database backends.

### chromaDB.ipynb

* Local ChromaDB vector store for embedding persistence.
* Collections created, embeddings inserted, and similarity search implemented.
* Efficient **Top-k retrieval** demonstrated.

### faiss.ipynb

* FAISS vector store integration.
* Sample documents embedded using OpenAI.
* FAISS index created, saved locally, and queried efficiently.
* Used in **Simple**, **Streaming**, and **Conversational RAG** chains.

### otherVectorStore.ipynb

* Demonstrates **In-memory vector store** for quick experimentation and demos.
* Built using `InMemoryVectorStore` from LangChain Core.
* Embeddings generated with OpenAI (`text-embedding-3-small`).
* Ideal for **lightweight prototypes and local testing**.

### pinecone.ipynb

* Integrates **Pinecone cloud vector database** for scalable storage and retrieval.
* Uses `PineconeVectorStore` with `text-embedding-3-small` embeddings.
* Automatically creates index (`1536d`, cosine metric) if not found.
* Enables **serverless, persistent RAG pipelines** on production-scale workloads.

**Outcome:** Local, in-memory, and cloud-based retrieval pipelines implemented — enabling both rapid prototyping and production-level deployment.

---

## Section 4: Advanced Chunking

Explores **semantic-aware text chunking** techniques to preserve meaning across document splits, improving retrieval accuracy and generation relevance.

### semantic_chunking.ipynb

* Implements custom **Threshold-based Semantic Chunker** using `SentenceTransformer (all-MiniLM-L6-v2)` and cosine similarity.
* Groups semantically related sentences into coherent chunks based on a **similarity threshold** (default 0.7).
* Integrates with FAISS vector store and Groq LLM (`gemma2-9b-it`) for retrieval-augmented QA.
* Includes LangChain’s **SemanticChunker** variant powered by `OpenAIEmbeddings`.
* Demonstrates **context-preserving retrieval** and improved grounding quality.

**Outcome:** Introduces semantic understanding to the preprocessing stage — a crucial step for **context-rich, high-fidelity RAG systems**.

---

## Notes and Design Decisions

* Chunking: Recursive, token-based, and semantic approaches for preserving meaning.
* Metadata schema: Unified across all data sources.
* Error handling: Try-except guards for malformed inputs.
* Extensibility: Ready for RAG querying and generation.

---

## Dependencies

* langchain
* langchain-community
* langchain-openai
* langgraph
* openai
* faiss-cpu / chroma
* pinecone-client
* pandas
* python-dotenv
* beautifulsoup4
* streamlit
* ipykernel
* matplotlib
* numpy
* langchain-huggingface
* sentence-transformers
* scikit-learn

---

## Author

**Bhopindrasingh Parmar**
[LinkedIn](https://www.linkedin.com/in/bhupenparmar/) | [GitHub](https://github.com/bhupencoD3)
