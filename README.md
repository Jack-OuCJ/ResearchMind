# ResearchMind

ResearchMind is an **Agentic RAG research assistant** for academic and analytical workflows.

It helps with:
- literature review,
- data analysis planning,
- paper writing support.

The current version uses Alibaba DashScope-compatible chat models (Qwen family).

## Project Overview

ResearchMind combines multi-agent orchestration and retrieval-augmented generation to answer research questions with grounded context.

- Splits a complex question into smaller sub-questions.
- Runs sub-tasks in parallel.
- Retrieves evidence from local documents and web pages.
- Aggregates sub-results into a structured final response.

## Tech Stack (Brief)

- **Python 3.12**: core language.
- **Streamlit**: web UI for uploading sources and chatting.
- **LangChain**: LLM integration and prompt workflows.
- **LangGraph**: multi-agent state graph orchestration.
- **ChromaDB**: local vector database persistence.
- **HuggingFace Embeddings** (`all-MiniLM-L6-v2`): dense retrieval embeddings.
- **PyPDF / python-docx / BeautifulSoup**: multi-source ingestion from PDF, DOCX, and web pages.

## Architecture Highlights

- **Main graph (LangGraph)**:
	- routing decision,
	- question decomposition,
	- parallel subgraph dispatch,
	- final aggregation.
- **Subgraph**:
	- retrieval,
	- context compression,
	- literature review,
	- data analysis,
	- writing synthesis.
- **RAG pipeline**:
	- parent/child chunking,
	- hybrid dense + sparse recall,
	- lightweight rerank,
	- source reference tracking.

## Project Structure

```text
ResearchMind/
├── app.py
├── main.py
├── pyproject.toml
├── README.md
├── README.zh-CN.md
└── researchmind/
		├── config.py
		├── llm.py
		├── rag.py
		├── agents.py
		└── ui.py
```

## Quick Start with `uv`

### 1) Create environment and install dependencies

```bash
uv venv
source .venv/bin/activate
uv sync
```

### 2) Configure API key

```bash
export DASHSCOPE_API_KEY="your_key"
```

Also compatible with:

```bash
export OPENAI_API_KEY="your_key"
```

### 3) Run the app

```bash
uv run streamlit run app.py
```

Open the local Streamlit URL shown in terminal.

## Typical Usage

1. Upload one or more `.pdf` / `.docx` files, or add a webpage URL.
2. Import sources into the RAG knowledge base.
3. Ask a research question in chat.
4. Review the generated answer and the referenced source snippets.

## Current Limitations

- `.doc` is not supported yet (only `.docx`).
- DashScope-compatible endpoint only.
- Vector data persists locally under `data/chroma/`.

## Project Characteristics

- End-to-end LLM application design with ingestion, retrieval, orchestration, and UI layers.
- Multi-agent execution workflow using graph-based task decomposition and aggregation.
- Practical RAG implementation with hybrid retrieval and source traceability.
