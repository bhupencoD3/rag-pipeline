
# Project RAG – Retrieval-Augmented Generation Learning Pipeline

This repository contains a step-by-step learning and experimentation project on **Retrieval-Augmented Generation (RAG)**. The goal is to build an end-to-end RAG pipeline from scratch, including data ingestion, parsing, chunking, embedding, and querying with LLMs. The project emphasizes understanding each stage of RAG, experimenting with different document types, and preparing a robust knowledge base for retrieval.

The work is structured across multiple Jupyter notebooks, each focusing on a specific data type or processing technique.

---

```
0-DataIngestParsing/
├── dataingestion.ipynb
├── dataingestion-pdf.ipynb
├── dataingestionDoc.ipynb
├── dataingestionCsvExcel.ipynb
├── dataingestionJson.ipynb
├── dataingestionSQL.ipynb
└── data/
    ├── text_files/
    ├── pdf/
    ├── word_files/
    ├── structured_files/
    ├── json_files/
    └── databases/
```


### Notebook 1: `dataingestion.ipynb` – Plain Text Files

* Created LangChain `Document` objects from raw text files, attaching metadata including author, page number, source, and custom fields.
* Generated sample text files (`python_intro.txt` and `intro_to_rag.txt`) in a structured directory.
* Loaded documents using `TextLoader` for single files and `DirectoryLoader` for entire folders.
* Explored chunking strategies:

  * `CharacterTextSplitter` for character-based chunks with overlap.
  * `RecursiveCharacterTextSplitter` for recursive splitting using multiple separators.
  * `TokenTextSplitter` for token-based fine-grained splitting.
* Outcome: Structured chunks ready for embedding with consistent metadata.

### Notebook 2: `dataingestion-pdf.ipynb` – PDF Documents

* Loaded PDF files using `PyPDFLoader`, `PyMuPDFLoader`, and `UnstructuredPDFLoader`.
* Implemented `SmartPDFProcessor` class that:

  * Cleans extracted text for OCR/encoding artifacts.
  * Splits pages into chunks using `RecursiveCharacterTextSplitter`.
  * Attaches metadata including page number, total pages, chunk method, and character count.
* Demonstrated error handling and filtering of small or empty chunks.
* Outcome: Clean, structured, and chunked PDF documents ready for embedding.

### Notebook 3: `dataingestionDoc.ipynb` – Word Documents

* Loaded `.docx` files using `Docx2txtLoader` and `UnstructuredWordDocumentLoader`.
* Extracted elements from documents with metadata (e.g., element type, position).
* Converted documents/elements into LangChain `Document` objects.
* Outcome: Structured Word document elements ready for retrieval pipelines.

### Notebook 4: `dataingestionCsvExcel.ipynb` – CSV and Excel Files

* **CSV ingestion:**

  * Used `CSVLoader` and `UnstructuredCSVLoader` to load row-wise data.
  * Created `Document` objects for each row with metadata like product name, category, price, and row index.
  * Function `process_csv_intelligently` created readable structured content for each row.

* **Excel ingestion:**

  * Read Excel sheets using `pandas` and `UnstructuredExcelLoader`.
  * Converted each sheet to `Document` objects with metadata including sheet name, rows, columns, and data type.
  * Supported multi-sheet Excel files.

* Outcome: Tabular data converted to structured, metadata-rich `Document` objects.

### Notebook 5: `dataingestionJson.ipynb` – JSON Files

* Loaded JSON files using `JSONLoader` and custom processing functions.
* Extracted nested arrays and flattened hierarchical structures.
* Function `process_json_intelligently`:

  * Constructs readable content from nested objects.
  * Attaches detailed metadata including employee ID, name, role, and project information.
* Outcome: Structured JSON documents ready for embedding and retrieval.

### Notebook 6: `dataingestionSQL.ipynb` – SQL Databases

* Created a sample SQLite database (`company.db`) inside `data/databases/` with two tables:

  * **employees** – employee details like name, role, department, salary.
  * **projects** – project details like name, status, budget, and lead.

* Loaded and inspected database schema using `SQLDatabase` from LangChain.

* Implemented `sql_to_documents` function:

  * Generates **table overview documents** (schema, columns, record counts, and sample records).
  * Generates **relationship documents** by joining tables (e.g., employees leading projects).
  * Attaches rich metadata (`source`, `table_name`, `num_records`, `data_type`).

* Example relationship output:

  ```
  Employee-Project Relationship:

  John Doe, Senior Developer leads RAG Implementation - Status: Active
  Jane Smith, Data Scientist leads Data Pipeline - Status: Completed
  ...
  ```

* **Outcome:** SQL databases are transformed into LangChain `Document` objects representing table structures, sample rows, and inter-table relationships—ready for embeddings and retrieval in RAG pipelines.

---

## Notes on Methods and Parameters

* **Chunking:** Recursive and token-based chunking ensures that documents are manageable for embedding models while preserving semantic coherence.
* **Metadata:** Uniform metadata schema across text, PDF, Word, CSV, Excel, and JSON allows precise filtering and retrieval.
* **Error Handling:** All loaders include try-except blocks, ensuring robust ingestion even if files are missing or partially corrupted.
* **Preprocessing:** Cleaning text (removing whitespace, correcting OCR errors) is applied uniformly to all document types.
* **Scalability:** Loaders and chunking logic support directories of files, multi-sheet Excel files, and large PDFs.

---

## Dependencies

* `langchain`, `langchain-community`, `langchain-openai`
* `langgraph`
* `openai`
* `faiss-cpu` / `chroma`
* `pandas`, `python-dotenv`, `beautifulsoup4`
* `streamlit`, `ipykernel`

---

## License

This project is licensed under the MIT License.

---

## Author

**Bhopindrasingh Parmar**
LinkedIn: [https://www.linkedin.com/in/bhupenparmar/](https://www.linkedin.com/in/bhupenparmar/)
GitHub: [https://github.com/bhupencoD3](https://github.com/bhupencoD3)