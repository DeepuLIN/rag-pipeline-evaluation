# RAG Pipeline Evaluation

An end-to-end Retrieval-Augmented Generation (RAG) project built with a custom ingestion pipeline, local embeddings, ChromaDB retrieval, local Ollama-based generation, and retrieval evaluation using MRR and nDCG.

This project focuses on building a practical RAG system from scratch rather than relying on heavy orchestration frameworks.

---

## 🚀 Features

* Custom RAG pipeline built from scratch
* Local embeddings with `sentence-transformers`
* Vector storage with `ChromaDB`
* Local LLM inference via `Ollama`
* Retrieval evaluation using:

  * Mean Reciprocal Rank (MRR)
  * Normalized Discounted Cumulative Gain (nDCG)
* Modular and extensible project structure

---

## 📁 Project Structure

```
rag-pipeline-evaluation/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── knowledge-base/
│   │   ├── company/
│   │   ├── contracts/
│   │   ├── employees/
│   │   └── products/
│   └── vectorstores/              # ignored in git
├── src/
│   └── rag/
│       ├── answer.py
│       ├── app.py
│       ├── config.py
│       ├── eval.py
│       ├── ingest.py
│       └── test_data.py
├── tests/
│   └── tests.jsonl
├── .gitignore
├── README.md
└── requirements.txt
```

---

## ⚙️ Pipeline Overview

1. **Ingestion** – Load, chunk, and store documents in ChromaDB
2. **Embedding** – Convert chunks into vectors
3. **Retrieval** – Fetch top-K relevant chunks
4. **Query Rewriting** – Improve query (optional)
5. **Reranking** – Heuristic / LLM-based refinement
6. **Answer Generation** – Generate response via Ollama
7. **Evaluation** – Measure performance with MRR & nDCG

---

## 🧰 Tech Stack

* Python
* ChromaDB
* Sentence Transformers
* Ollama
* Gradio
* Pydantic
* python-dotenv

---

## 🛠 Installation

### Clone repo

```bash
git clone https://github.com/DeepuLIN/rag-pipeline-evaluation.git
cd rag-pipeline-evaluation
```

### Create environment

```bash
conda create -n rag python=3.11 -y
conda activate rag
```

or

```bash
python -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🤖 Ollama Setup

```bash
ollama pull llama3.2:latest
ollama serve
```

---

## ⚙️ Configuration

Edit:

```
src/rag/config.py
```

Example:

```python
OLLAMA_MODEL = "llama3.2:latest"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

RETRIEVAL_K = 20
FINAL_K = 10
```

---

## ▶️ How to Run

### Build vector DB

```bash
python -m src.rag.ingest
```

### Run app

```bash
python -m src.rag.app
```

### Run evaluation

```bash
python -m src.rag.eval
```

---

## 📊 Evaluation Metrics

**MRR** → measures how early the first relevant result appears
**nDCG** → measures ranking quality across results

---

## 📈 Example Output

```python
{'question': 'When was Insurellm founded?', 'mrr': 1.0, 'ndcg': 0.8772}
{'question': 'Who founded Insurellm?', 'mrr': 1.0, 'ndcg': 0.9197}
```

---

## 🧠 Strengths

* Direct factual queries
* Employee lookup
* Product / contract queries

---

## ⚠️ Limitations

* Multi-hop reasoning
* Cross-document relationships
* Aggregated questions

---

## 🎯 Design Choices

**No LangChain**

* Full control
* Easier debugging

**Local embeddings**

* No API dependency

**Ollama**

* Simple local inference

---

## 🔮 Future Improvements

* Hybrid retrieval (BM25 + dense)
* Better embeddings (BGE)
* Cross-encoder reranking
* Answer-level evaluation

---

## 👤 Author

Deepak L
https://github.com/DeepuLIN

---

## 📄 License

For learning and portfolio use.

