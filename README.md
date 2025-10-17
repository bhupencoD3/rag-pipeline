# Project RAG – Retrieval-Augmented Generation Learning Pipeline

This repository is a step-by-step learning and experimentation project on Retrieval-Augmented Generation (RAG). The goal is to build an end-to-end RAG pipeline from scratch, including data ingestion, parsing, chunking, embedding, and vector storage, to create a robust knowledge base for retrieval and generation.

The project is modularly structured into three core stages:

1. Data Ingestion & Parsing – converting raw files into structured LangChain Document objects.
2. Vector Embeddings – transforming documents into numerical embeddings.
3. Vector Stores – persisting and managing embeddings for fast similarity-based retrieval.

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
    └── otherVectorStore.ipynb
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

1. embedding.ipynb

   * Explained embeddings with simple 2D visualizations.
   * Manual cosine similarity computation.
   * Used HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2).

2. openai-embeddings.ipynb

   * Used OpenAI Embeddings (text-embedding-3-small).
   * Computed similarity scores for semantic matching.
   * Implemented a mini semantic search demo.

**Outcome:** Generated embeddings ready for vector storage.

---

## Section 3: Vector Stores

Manages document embeddings for fast similarity-based retrieval. Two vector database integrations are implemented.

### chromaDB.ipynb

* Local ChromaDB vector store for embedding persistence.
* Collection created and embeddings inserted.
* Similarity search queries implemented.
* Top-k retrieval works efficiently.

### faiss.ipynb

* FAISS vector store integration.
* Sample documents created and chunked.
* Embeddings generated with OpenAI.
* FAISS index created and saved locally.
* Similarity search implemented with optional metadata filtering.
* Retrieval used for RAG pipelines: simple RAG, streaming RAG, and conversational RAG chains.

**Outcome:** Local and FAISS-based retrieval pipelines fully functional.

---

## Notes and Design Decisions

* Chunking: Recursive and token-based to preserve semantic meaning.
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
* pandas
* python-dotenv
* beautifulsoup4
* streamlit
* ipykernel
* matplotlib
* numpy
* langchain-huggingface
* sentence-transformers

---

## Author

Bhopindrasingh Parmar
[LinkedIn](https://www.linkedin.com/in/bhupenparmar/) | [GitHub](https://github.com/bhupencoD3)
