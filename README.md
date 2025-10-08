# 🚀 Project RAG – Retrieval-Augmented Generation Learning Pipeline

This repository contains a step-by-step learning and experimentation project on **Retrieval-Augmented Generation (RAG)**.
The goal is to build an end-to-end RAG pipeline from scratch, including **data ingestion, parsing, chunking, embedding, and vector storage**, preparing a robust knowledge base for retrieval and generation.

The project is modularly structured into **three core stages**:

1. **Data Ingestion & Parsing** – converting raw files into structured LangChain Document objects
2. **Vector Embeddings** – transforming documents into numerical embeddings
3. **Vector Stores** – persisting and managing embeddings for fast similarity-based retrieval

---

## 🗂️ Project Structure

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
    └── Datastaxdb.ipynb
```

---

## 🧩 Section 1: Data Ingestion and Parsing

Processes diverse document formats — **Text, PDF, Word, CSV/Excel, JSON, SQL** — into structured, chunked, metadata-rich **LangChain Document objects**.

**Highlights:**

* Implemented multiple loaders: *TextLoader, PyPDFLoader, Docx2txtLoader, UnstructuredCSVLoader, JSONLoader, SQLDatabase*
* Used intelligent chunking strategies (`RecursiveCharacterTextSplitter`, `TokenTextSplitter`)
* Attached detailed metadata (source, author, file type, page number, etc.)
* Cleaned and validated data across formats (PDF OCR, whitespace normalization, small chunk filtering)

**Outcome:** Consistent document structures ready for embedding.

---

## 🔍 Section 2: Vector Embeddings

Transforms text chunks into **high-dimensional semantic vectors** suitable for retrieval.

### Notebook 2.1: `embedding.ipynb`

* Explained embeddings with simple 2D visualization examples
* Implemented cosine similarity manually
* Used **HuggingFaceEmbeddings** (`sentence-transformers/all-MiniLM-L6-v2`)

### Notebook 2.2: `openai-embeddings.ipynb`

* Used **OpenAI Embeddings** (`text-embedding-3-small`)
* Computed similarity scores for semantic matching
* Implemented mini **semantic search** demo

**Outcome:** Generated embeddings ready for storage in vector databases.

---

## 🛠️ Section 3: Vector Stores

This section stores and manages document embeddings for fast retrieval. Two vector database integrations are implemented.

### 🔹 Notebook 3.1: `chromaDB.ipynb`

* Integrated **ChromaDB**, a local vector store for embedding persistence
* Created collection and inserted embeddings generated in previous steps
* Implemented **similarity search** queries
* Retrieved top-k most relevant chunks efficiently

**Outcome:** Local retrieval pipeline fully functional using Chroma.

---

### 🔹 Notebook 3.2: `Datastaxdb.ipynb`

* Integrated **DataStax AstraDB**, a managed distributed vector database
* Used **LangChain DatastaxVectorStore** to connect via Astra tokens
* Inserted embeddings and metadata from multiple document types
* Performed **semantic retrieval** queries directly from cloud DB

**Outcome:** Scalable vector retrieval pipeline leveraging AstraDB.

---

## 🗒️ Notes & Design Decisions

* **Chunking:** Recursive and token-based to preserve meaning
* **Metadata Schema:** Unified across all data sources
* **Error Handling:** Try-except guards for incomplete or malformed inputs
* **Extensibility:** Ready for next phase – RAG Querying and Generation

---

## ⚙️ Dependencies

* `langchain`
* `langchain-community`
* `langchain-openai`
* `langgraph`
* `openai`
* `faiss-cpu` / `chroma`
* `pandas`
* `python-dotenv`
* `beautifulsoup4`
* `streamlit`
* `ipykernel`
* `matplotlib`
* `numpy`
* `langchain-huggingface`
* `sentence-transformers`

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👨‍💻 Author

**Bhopindrasingh Parmar**
🔗 [LinkedIn](https://www.linkedin.com/in/bhupenparmar/) | 🖥️ [GitHub](https://github.com/bhupencoD3)
