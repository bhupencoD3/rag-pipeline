# Project RAG – Retrieval-Augmented Generation Learning Pipeline

This repository contains a step-by-step learning and experimentation project on **Retrieval-Augmented Generation (RAG)**. The goal is to build an end-to-end RAG pipeline from scratch, including **data ingestion, parsing, chunking, embedding, and querying with LLMs**. The project emphasizes understanding each stage of RAG, experimenting with different document types, and preparing a robust knowledge base for retrieval.

The work is structured across multiple Jupyter notebooks, organized into two sections: **Data Ingestion and Parsing** and **Vector Embeddings**.

---

## 📂 Project Structure

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
└── 1-VectorEmbeddings/
    └── embedding.ipynb
```

---

## 📘 Section 1: Data Ingestion and Parsing

This section covers the ingestion and preprocessing of various document types into structured **LangChain Document objects**, including parsing, chunking, and metadata attachment.

### 📄 Notebook 1.1: `dataingestion.ipynb` – Plain Text Files

* Created LangChain Document objects from raw text files, attaching metadata including author, page number, source, and custom fields.
* Generated sample text files (`python_intro.txt`, `intro_to_rag.txt`).
* Loaded documents using **TextLoader** and **DirectoryLoader**.
* Explored chunking strategies:

  * `CharacterTextSplitter` (character-based with overlap)
  * `RecursiveCharacterTextSplitter` (multi-separator recursive splitting)
  * `TokenTextSplitter` (token-based fine-grained splitting)

**Outcome:** Structured chunks ready for embedding with consistent metadata.

### 📑 Notebook 1.2: `dataingestion-pdf.ipynb` – PDF Documents

* Loaded PDFs using **PyPDFLoader**, **PyMuPDFLoader**, and **UnstructuredPDFLoader**.
* Implemented `SmartPDFProcessor` class:

  * Cleans extracted text (OCR/encoding artifacts).
  * Splits pages into chunks using `RecursiveCharacterTextSplitter`.
  * Attaches metadata (page number, total pages, chunk method, char count).
* Error handling and filtering of small/empty chunks.

**Outcome:** Clean, structured, and chunked PDF documents ready for embedding.

### 📃 Notebook 1.3: `dataingestionDoc.ipynb` – Word Documents

* Loaded `.docx` files using **Docx2txtLoader** and **UnstructuredWordDocumentLoader**.
* Extracted elements with metadata (element type, position).
* Converted into LangChain Document objects.

**Outcome:** Structured Word document elements ready for retrieval pipelines.

### 📊 Notebook 1.4: `dataingestionCsvExcel.ipynb` – CSV & Excel Files

**CSV Ingestion:**

* Loaded row-wise data with **CSVLoader** and **UnstructuredCSVLoader**.
* Created Document objects with metadata (product name, category, price, row index).
* Implemented `process_csv_intelligently` for readable structured content.

**Excel Ingestion:**

* Read sheets using **pandas** and **UnstructuredExcelLoader**.
* Converted sheets into Document objects with metadata (sheet name, rows, columns, data type).
* Supported multi-sheet Excel files.

**Outcome:** Tabular data converted into structured, metadata-rich Document objects.

### 📦 Notebook 1.5: `dataingestionJson.ipynb` – JSON Files

* Loaded JSON using **JSONLoader** and custom functions.
* Extracted nested arrays and flattened structures.
* Implemented `process_json_intelligently`:

  * Constructs readable content from nested objects.
  * Attaches detailed metadata (employee ID, name, role, project info).

**Outcome:** Structured JSON documents ready for embedding and retrieval.

### 🗄️ Notebook 1.6: `dataingestionSQL.ipynb` – SQL Databases

* Created a sample SQLite database (`company.db`) with two tables:

  * **employees** (name, role, department, salary)
  * **projects** (name, status, budget, lead)
* Loaded schema using **SQLDatabase** from LangChain.
* Implemented `sql_to_documents`:

  * Table overview documents (schema, columns, record counts, samples).
  * Relationship documents via joins (e.g., employees leading projects).
  * Rich metadata (source, table\_name, num\_records, data\_type).

**Example Relationship Output:**

```
Employee-Project Relationship:
John Doe, Senior Developer leads RAG Implementation - Status: Active
Jane Smith, Data Scientist leads Data Pipeline - Status: Completed
```

**Outcome:** SQL databases transformed into Document objects representing schema, rows, and relationships.

---

## 📘 Section 2: Vector Embeddings

This section focuses on generating vector representations of processed documents for **semantic search and retrieval**.

### 🔎 Notebook 2.1: `embedding.ipynb` – Vector Embeddings Fundamentals

* Introduced embeddings with a 2D toy example (e.g., *cat, kitten, dog, puppy, car, truck*).
* Visualized embeddings with **matplotlib**.
* Implemented `cosine_similarity` for vector comparison.
* Used **HuggingFaceEmbeddings** (`sentence-transformers/all-MiniLM-L6-v2`) to generate embeddings.
* Demonstrated single-query and multi-sentence embeddings.

**Outcome:** Generated embeddings for text data, ready for storage in a vector DB.

---

## 📝 Notes on Methods and Parameters

* **Chunking:** Recursive & token-based strategies preserve semantic coherence.
* **Metadata:** Uniform schema across text, PDF, Word, CSV, Excel, JSON, SQL.
* **Error Handling:** Robust try-except logic for incomplete/corrupted files.
* **Preprocessing:** Cleaning applied uniformly (whitespace, OCR corrections).
* **Scalability:** Supports multi-file, multi-sheet, large PDFs, and varied text inputs.

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
🔗 [LinkedIn](https://www.linkedin.com/in/bhupenparmar/)
💻 [GitHub](https://github.com/bhupencoD3)
