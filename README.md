# Document RAG Chatbot

## Project Description

Document RAG Chatbot is a retrieval-augmented question-answering system that lets users query their own documents through a conversational interface.

The application ingests PDF, TXT, and Markdown files, builds a semantic vector index with FAISS, and answers questions using an **extractive RAG pipeline** grounded in retrieved evidence. Instead of relying on a generative LLM to invent long answers, it prioritizes sentences found in the source documents and returns the file names used as references.

The project includes both a **CLI chat mode** and a **Flask web interface**, plus an automated batch-evaluation workflow for testing prompts and measuring response quality.

---

## Technologies Used

- **Python** – Core application logic and NLP heuristics
- **LangChain** – Document loading, chunking, and vector store integration
- **Sentence Transformers** – Semantic embeddings (`all-MiniLM-L6-v2`)
- **FAISS** – Fast similarity search over document vectors
- **Flask** – Web API and browser-based chat interface
- **PyPDF** – PDF document parsing

---

## Main Features

- Document ingestion from PDF, TXT, and MD files
- Semantic search with FAISS over custom embeddings
- Extractive answers based on retrieved evidence
- Source attribution for every response
- Question intent detection (factual, comparative, summary, integration, consequence)
- Domain-specific rules for legal/constitutional queries (articles, rights, capital, etc.)
- Sentence-level reranking and noise filtering
- Interactive CLI chat
- Web chat interface with REST API
- Batch evaluation mode with Markdown report output
- Persistent FAISS index (build once, reuse on next runs)

---

## RAG Pipeline

1. **Document ingestion** – Load files from a local folder and attach source metadata
2. **Text chunking** – Split documents into overlapping segments (700 chars, 120 overlap)
3. **Embedding generation** – Encode chunks with `sentence-transformers/all-MiniLM-L6-v2`
4. **Vector indexing** – Store embeddings in a FAISS index on disk
5. **Query processing** – Normalize text, detect intent, and apply domain rules
6. **Semantic retrieval** – Fetch top-k relevant chunks by vector similarity
7. **Sentence reranking** – Score candidate sentences against the question
8. **Answer composition** – Build a grounded response and return source documents

---

## Documents & Evaluation

* Supports **PDF**, **TXT**, and **MD** formats
* Recommended: at least **5 documents** (10–50 pages total)
* Default corpus: Spanish constitutional and legal reference material
* Includes **10 evaluation prompts** covering:
  * Factual questions
  * Comparative questions
  * Simple reasoning / consequence queries
  * Cross-document integration
* Batch mode exports results to a Markdown table with prompts, answers, and sources

Example evaluation run:

```bash
python chatbot_rag.py --prompts-file prompts_10_evaluacion.txt --prompts-output resultados_prompts.md
```

---

## Project Structure

```text
Chatbot/
├── chatbot_rag.py              # RAG engine, CLI chat, batch evaluation
├── web_chat.py                 # Flask web server
├── templates/
│   └── index.html              # Web chat UI
├── data/
│   ├── docs/                   # User documents (not included in repo)
│   └── faiss_index/            # Generated vector index
├── prompts_10_evaluacion.txt   # Sample evaluation prompts
├── prompts_evaluacion.md       # Evaluation guidelines
├── requirements.txt
└── README.md
```

---

## Screenshots

> Add your own screenshots to an `images/` folder before publishing.

### Web Chat Interface

![Web Chat Interface](images/web_chat.jpg)

### CLI Interactive Mode

![CLI Chat](images/cli_chat.jpg)

### Batch Evaluation Output

![Evaluation Results](images/evaluation_results.jpg)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/germanmm04/document-rag-chatbot.git
cd document-rag-chatbot
```

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

Add your documents to `data/docs/` (at least 5 files in PDF, TXT, or MD format).

---

## Usage

### CLI chat (interactive)

```bash
python chatbot_rag.py
```

Rebuild the index after adding or updating documents:

```bash
python chatbot_rag.py --docs data/docs --index data/faiss_index --rebuild-index
```

### Web interface

```bash
python web_chat.py
```

Open in your browser:

```text
http://127.0.0.1:5000
```

Custom host and port:

```bash
python web_chat.py --host 127.0.0.1 --port 5000
```

---

## What I Learned

* Retrieval-Augmented Generation (RAG) fundamentals
* Semantic search with embeddings and vector databases
* Document chunking strategies and overlap tuning
* Sentence-level reranking and extractive answer composition
* Reducing hallucinations by grounding responses in source evidence
* Intent classification and heuristic NLP for query understanding
* Building dual interfaces (CLI + web) around the same AI backend
* Designing evaluation workflows for QA systems
* Integrating Hugging Face models with LangChain and FAISS

---

## Future Improvements

* Migrate to `langchain-huggingface` and newer LangChain APIs
* Add support for more document formats (DOCX, HTML)
* Integrate a lightweight generative LLM for answer synthesis with strict citation
* Improve cross-document integration for multi-source queries
* Add a simple admin panel to upload documents from the web UI
* Deploy as a containerized service (Docker)
* Include automated metrics (precision, recall, faithfulness)
* Support multilingual document collections

---

## Author

Personal project developed as part of my portfolio in **Artificial Intelligence**, **Big Data**, and **Software Development**.

**GitHub:** [germanmm04](https://github.com/germanmm04)
