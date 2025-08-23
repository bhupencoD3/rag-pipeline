# 🧩 RAG (Retrieval-Augmented Generation)

This is my **step-by-step Retrieval-Augmented Generation (RAG) learning project**.  
It covers the full pipeline: data parsing, ingestion, embeddings, vector databases, and querying using **LangChain**, **FAISS/ChromaDB**, and **LLMs (Groq/OpenAI)**.  

The goal is to **learn RAG from scratch**, breaking it into small stages and building a working pipeline piece by piece.

---

## 📂 Project Structure

```bash
.
├── DataIngestParsing/
│   ├── dataingestion.ipynb     
│   └── data/
│       └── text_files/
│           ├── intro_to_rag.txt   
│           └── python_intro.txt   
├── main.py                        
├── pyproject.toml                 
├── README.md                      
├── requirements.txt              
└── uv.lock                        
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/your-username/rag.git
cd rag
```

### 2. Install dependencies  
With [uv](https://github.com/astral-sh/uv):  
```bash
uv sync
```

Or with pip:  
```bash
pip install -r requirements.txt
```

### 3. Run notebooks  
```bash
jupyter notebook
```

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/) – RAG framework  
- [ChromaDB](https://www.trychroma.com/) / [FAISS](https://faiss.ai/) – vector databases  
- [Sentence Transformers](https://www.sbert.net/) – embeddings  
- [PyPDF](https://pypi.org/project/pypdf/) – PDF parsing  
- [dotenv](https://pypi.org/project/python-dotenv/) – environment config  
- [Groq](https://groq.com/) / [OpenAI](https://platform.openai.com/) – LLMs  


---

## 📌 Roadmap

- [x] Data ingestion & parsing  
- [ ] Vector DB integration (FAISS / Chroma)  
- [ ] Embedding & indexing  
- [ ] Retriever setup  
- [ ] Connecting with LLMs  
- [ ] End-to-end RAG pipeline  

---

## 🤝 Contributing

This is a personal learning project, but contributions/suggestions are welcome!  

---

MIT License
## 📜 License
This project is licensed under the [MIT License](LICENSE).

---

## 📌 Author
**bhopindrasingh parmar**  
👤 [LinkedIn](https://www.linkedin.com/in/bhupenparmar/) | [GitHub](https://github.com/bhupencoD3)