# 🔷 Azure AI Search RAG Agent

A production-grade Retrieval-Augmented Generation (RAG) application built entirely on the Azure stack. Upload PDF documents, ask questions, get cited answers — all running inside your own Azure subscription.

---

## Table of Contents

- [Overview](#overview)
- [Capabilities](#capabilities)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Azure Setup](#azure-setup)
- [Local Setup](#local-setup)
- [Running the App](#running-the-app)
- [How Each Capability Works](#how-each-capability-works)
- [Configuration Reference](#configuration-reference)
- [Cost Model](#cost-model)
- [Security](#security)
- [Deploying to Azure](#deploying-to-azure)

---

## Overview

This application lets you index your own PDF documents into Azure AI Search and interact with them through six capabilities — all powered by Azure OpenAI GPT-4o mini and exposed through a Streamlit web interface.

**The full Azure stack:**

| Service | Role |
|---|---|
| Azure AI Search (S1) | Hybrid BM25 + HNSW vector search + Semantic reranker |
| Azure OpenAI `text-embedding-3-small` | Converts text to 1536-dim vectors |
| Azure OpenAI `gpt-4o-mini` | Answer generation, summarisation, comparison, evaluation |
| Azure Key Vault | Secret management + Customer-Managed Key (CMK) encryption |
| Azure Private Endpoints | Network isolation — no public internet exposure |

Everything runs inside your Azure subscription. No data leaves your tenant.

---

## Capabilities

### 1. 💬 RAG Chat
Ask questions about your documents. The pipeline runs hybrid search (BM25 + vector + semantic reranker) to find the most relevant passages, then generates a cited answer using GPT-4o mini. Every claim in the answer is linked to the exact source chunk via hoverable citation badges. Supports multi-turn follow-up questions.

### 2. 📋 Summarise
Select any indexed document and generate a structured five-section summary: Executive Summary, Key Topics, Main Arguments, Notable Details, and Suggested Follow-up Questions.

### 3. ⚖️ Compare
Select two documents for a six-section comparative analysis: Overview, Common Themes, Key Differences, Unique to A, Unique to B, and Synthesis & Recommendation.

### 4. 🧪 Evaluation Pipeline
Run Azure AI Evaluation SDK evaluators against any query to measure RAG quality across four dimensions: Relevance, Groundedness, Coherence, and Fluency — all scored 1–5 by GPT-4o mini acting as judge.

### 5. 🔒 Security
Live reference panel showing the RBAC access matrix, active Azure security controls (TLS 1.3, CMK, Private Endpoints, Managed Identity).

### 6. 💰 Cost Model
Monthly cost breakdown at 1,000 queries/day.

---

## Architecture

### RAG Pipeline (per query)

```
User query
    │
    ▼  embed_single()
    │  text-embedding-3-small → 1536-dim vector
    │
    ▼  hybrid_search()
    │  BM25 full-text  ─┐
    │  HNSW vector     ─┼─► Reciprocal Rank Fusion ──► Semantic reranker
    │  (Azure AI Search)│                               (cross-encoder)
    │                  ─┘
    ▼  PromptBuilder.rag_user_message()
    │  SOURCES block [1]..[N] + QUESTION → user message
    │
    ▼  GenerationService.generate()
    │  GPT-4o mini → cited answer
    │  Stores: clean Q&A in sliding-window history (no dangling [N])
    │
    ▼  render_answer()
    │  [N] → @@CITE_N@@ → markdown → HTML tooltip badges
    │
    Answer with interactive citation badges
```

### Document Ingestion (on upload)

```
PDF bytes
    │
    ▼  PDFExtractor.from_bytes()
    │  PyMuPDF → per-page text blocks → Document + metadata["page_texts"]
    │
    ▼  DocumentChunker.chunk()
    │  Page-aware sentence-boundary chunking
    │  Deterministic IDs: {doc_id}-chunk-{i:04d}
    │
    ▼  DocumentChunker.embed_texts()  +  EmbeddingService.embed_batch()
    │  Overlap-enhanced embeddings (wider context for vectors)
    │  Stored chunk.text stays clean — overlap never persisted
    │
    ▼  AzureSearchRetriever.upload_chunks()
    │  merge_or_upload_documents → idempotent upsert
    │
    Chunks indexed in Azure AI Search
```

### Index Schema

```
chunk_id      String   key, filterable
doc_id        String   filterable, facetable
doc_name      String   searchable, filterable, facetable
chunk_index   Int32    filterable, sortable
page_num      Int32    filterable, sortable
text          String   searchable  (en.microsoft analyzer — BM25)
text_vector   Collection(Single)   1536-dim HNSW cosine (m=4, ef=400/500)

Semantic config: content=text, keywords=doc_name
```

---

## Project Structure

```
rag_pkg/
├── app.py                  Streamlit entry point  (streamlit run app.py)
├── agent.py                RAGAgent — central orchestrator
├── config.py               AzureConfig — loads all credentials from .env
├── models.py               Shared dataclasses: Document, Chunk, ChatMessage, CostEstimate
├── requirements.txt
├── .env.example            Environment variable template
│
├── core/                   Azure service wrappers
│   ├── index_manager.py    SearchIndexClient — HNSW + BM25 + Semantic schema
│   ├── embeddings.py       EmbeddingService — text-embedding-3-small + LRU cache
│   ├── chunker.py          DocumentChunker — page-aware sentence-boundary chunker
│   ├── retriever.py        AzureSearchRetriever — hybrid_search, upload, delete
│   └── pdf_extractor.py    PDFExtractor — PyMuPDF → Document
│
├── generation/             Prompt construction + GPT-4o mini
│   ├── prompts.py          PromptBuilder — rag, summary, compare prompts
│   └── service.py          GenerationService — stateful + stateless generation
│
├── summary/                Capability 2 — document summarisation
│   └── summariser.py       Summariser.run(doc) → five-section summary
│
├── compare/                Capability 3 — cross-document comparison
│   └── comparator.py       Comparator.run(doc_a, doc_b) → six-section analysis
│
├── evaluation/             Capability 4 — Azure AI Evaluation SDK
│   ├── scores.py           EvalScores dataclass — four scores + verdict
│   └── evaluator.py        AzureEvaluationService — four LLM-judge evaluators
│
├── cost_model/             Capability 6 — cost estimation
│   └── model.py            CostModel.estimate() 
│
├── security/               Capability 5 — security reference data
│   └── rbac.py             RBAC_MATRIX, SECURITY_CONTROLS
│
└── ui/                     Streamlit frontend
    ├── styles.py            inject_css() — Azure dark theme + citation CSS
    ├── helpers.py           render_answer(), cite_badge(), score_bar_html()
    ├── session.py           init_session(), bootstrap_agent()
    ├── sidebar.py           PDF upload panel + document list
    └── tabs/
        ├── rag_chat.py      Tab 1 — RAG Chat
        ├── summarise.py     Tab 2 — Summarise
        ├── compare.py       Tab 3 — Compare
        ├── evaluation.py    Tab 4 — Evaluation Pipeline
        ├── security.py      Tab 5 — Security Matrix
        └── cost.py          Tab 6 — Cost Model
```

---

## Prerequisites

- Python 3.10 or later
- An Azure subscription
- Azure AI Search resource — **Standard S1 tier minimum** (Free tier does not support vector search)
- Azure OpenAI resource with two deployments:
  - `text-embedding-3-small` — for document and query embeddings
  - `gpt-4o-mini` — for answer generation and evaluation

---

## Azure Setup

### 1. Create Azure AI Search (S1)

In the Azure Portal, create an Azure AI Search resource. Select the **Standard S1** pricing tier. The Free and Basic tiers do not support vector fields or the semantic reranker.

Once created, navigate to **Keys** and copy the Admin key and the service endpoint URL.

```
Endpoint format: https://<your-service-name>.search.windows.net
```

The application creates the index schema automatically on first startup — you do not need to create the index manually.

### 2. Create Azure OpenAI Resource

Create an Azure OpenAI resource in a region that supports both `text-embedding-3-small` and `gpt-4o-mini` (East US, West Europe, and Sweden Central are reliable choices as of 2025).

Navigate to **Azure OpenAI Studio → Deployments** and create two deployments:

| Deployment Name | Model | Notes |
|---|---|---|
| `text-embedding-3-small` | text-embedding-3-small | Set capacity to at least 120K TPM |
| `gpt-4o-mini` | gpt-4o-mini | Set capacity based on expected load |

The deployment names in your `.env` file must match exactly what you set here.

Copy the endpoint URL and API key from **Keys and Endpoint**.

### 3. Enable Semantic Ranker

The semantic ranker is available on the S1 tier at no extra charge for the first 1,000 queries/month (then $1/1,000 queries). It is enabled automatically when you set `USE_SEMANTIC_RANK=true` in your `.env` file — no manual Portal configuration required.

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/subodh556/Azure_RAG_Agent.git
cd Azure_RAG_Agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in your values:

```env
AZURE_SEARCH_ENDPOINT=https://<your-service>.search.windows.net
AZURE_SEARCH_API_KEY=<your-admin-key>
RAG_INDEX_NAME=rag-documents

AZURE_OPENAI_ENDPOINT=https://<your-service>.openai.azure.com
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_EMBED_DEP=text-embedding-3-small
AZURE_OPENAI_CHAT_DEP=gpt-4o-mini
EMBED_DIMENSIONS=1536

TOP_K=5
USE_SEMANTIC_RANK=true
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

On first startup you will see a **Startup log** expander confirming connection to both Azure services and index creation. If any environment variable is missing, the error message lists exactly which variables need to be set.

### Upload your first document

1. In the sidebar, click **Choose PDF files** and select one or more PDFs
2. Click **Index Uploaded PDFs**
3. Wait for the progress bar — extraction, chunking, embedding, and indexing all happen in sequence
4. Once complete, the document appears in the **Indexed Documents** list

### Ask your first question

Switch to the **💬 RAG Chat** tab, type a question in the input box, and press **Send ➤**. The answer appears with inline citation badges — hover over any badge to see the source document, page number, and the exact chunk text the claim is based on.

---

## How Each Capability Works

### RAG Chat — multi-turn with citations

Each query runs a full hybrid search (BM25 + HNSW vectors, fused via Reciprocal Rank Fusion, optionally re-ranked by the semantic reranker). The top-K chunks are assembled into a SOURCES block which is sent to GPT-4o mini alongside your question. The model is instructed to cite inline using `[N]` markers.

**Multi-turn history** is maintained correctly across follow-up questions. The key design: the full SOURCES block is sent to the API each turn but never stored in history. Only the clean question (no sources) and a citation-stripped answer are persisted. This prevents dangling `[N]` markers from a previous turn's source set from appearing in subsequent context, which would cause the model to freeze or hallucinate.

### Summarise — structured five-section output

All chunks for the selected document are fetched from Azure AI Search in original reading order (`chunk_index asc`) and concatenated. GPT-4o mini is given the full text and a structured prompt enforcing five sections. Uses `generate_once()` — stateless, no history read or written, no contamination of the RAG Chat conversation.

### Compare — six-section comparative analysis

Both documents are fetched in full from the index. The entire text of both documents is embedded into the system prompt alongside a six-section structure (Overview, Common Themes, Key Differences, Unique to A, Unique to B, Synthesis). The model reads both simultaneously. Like Summarise, uses `generate_once()` — the large document context never enters conversation history.

### Evaluation Pipeline — Azure AI Evaluation SDK

Runs the same retrieval and generation pipeline as RAG Chat, then passes the answer through four Azure AI Evaluation SDK evaluators:

| Evaluator | What it measures | Inputs |
|---|---|---|
| `RelevanceEvaluator` | Does the answer address the query? | query + response + context |
| `GroundednessEvaluator` | Is every claim supported by sources? | query + response + context |
| `CoherenceEvaluator` | Is the answer logically structured? | query + response |
| `FluencyEvaluator` | Is the answer natural to read? | query + response |

All four use GPT-4o mini as the judge model. Each returns a 1–5 score and a natural language explanation. An overall score is computed as the mean of all four, with verdict thresholds: PASS (≥ 4.0), PARTIAL (≥ 2.5), FAIL (< 2.5).

### Security

Static reference tab — no API calls. Displays the RBAC access matrix (six roles × six permission categories), active Azure security controls. Embedded in the app so developers, auditors, and stakeholders can review security posture without accessing the Azure Portal.

---

## Configuration Reference

All configuration is loaded from environment variables. The `.env` file is the recommended way to set them locally.

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_SEARCH_ENDPOINT` | ✅ | — | Full URL of your Azure AI Search service |
| `AZURE_SEARCH_API_KEY` | ✅ | — | Admin key from Azure Portal → Keys |
| `RAG_INDEX_NAME` | ❌ | `rag-documents` | Name of the search index to create/use |
| `AZURE_OPENAI_ENDPOINT` | ✅ | — | Full URL of your Azure OpenAI resource |
| `AZURE_OPENAI_API_KEY` | ✅ | — | Key from Azure Portal → Keys and Endpoint |
| `AZURE_OPENAI_EMBED_DEP` | ❌ | `text-embedding-3-small` | Embedding deployment name |
| `AZURE_OPENAI_CHAT_DEP` | ❌ | `gpt-4o-mini` | Chat completion deployment name |
| `EMBED_DIMENSIONS` | ❌ | `1536` | Vector dimensions — must match deployment |
| `TOP_K` | ❌ | `5` | Number of chunks to retrieve per query |
| `USE_SEMANTIC_RANK` | ❌ | `true` | Enable semantic reranker (`true`/`false`) |

### Tuning TOP_K

`TOP_K` controls how many chunks are retrieved and sent to the LLM per query. Higher values give the model more context but increase input token costs and latency.

| TOP_K | Use case |
|---|---|
| 3 | Simple factual questions, cost-sensitive |
| 5 | Default — good balance for most queries |
| 8–10 | Complex analytical questions spanning many sections |

### Disabling Semantic Ranking

Setting `USE_SEMANTIC_RANK=false` skips the semantic reranker and uses only BM25 + vector (RRF) scoring. This reduces per-query cost by ~$0.001 and latency by ~200–400ms. Recommended when your queries are short and keyword-heavy rather than long and conversational.

---

## Cost Model

### Pricing (USD, 2025)

| Meter | Rate |
|---|---|
| text-embedding-3-small | $0.02 / 1M tokens |
| GPT-4o mini — input | $0.15 / 1M tokens |
| GPT-4o mini — output | $0.60 / 1M tokens |
| Azure AI Search S1 | $250 / month / search unit |
| Semantic reranker | $1.00 / 1,000 queries (above 1K free/month) |

### Monthly estimate at 1,000 queries/day

| Component | Monthly |
|---|---|
| Azure AI Search S1 (2 replicas) | $500.00 |
| Log Analytics (2 GB/day) | $165.60 |
| Azure OpenAI GPT-4o mini | $25.20 |
| Azure Search semantic ranker | $30.00 |
| Azure OpenAI text-embedding-3-small | $6.00 |
| Private Endpoints × 2 | $14.40 |
| Other (Blob, Key Vault, Monitor) | $14.30 |
| **Total** | **~$755–$1,136** |

---

## Security

### RBAC summary

| Role | Read | Write | Delete | Manage Index | View Logs |
|---|---|---|---|---|---|
| Search Admin | ✅ | ✅ | ✅ | ✅ | ✅ |
| Contributor | ✅ | ✅ | ❌ | ❌ | ❌ |
| Search Reader | ✅ | ❌ | ❌ | ❌ | ❌ |
| Indexer (MI) | ❌ | ✅ | ❌ | ✅ | ❌ |
| Auditor | ✅ | ❌ | ❌ | ❌ | ✅ |
| Anonymous | ❌ | ❌ | ❌ | ❌ | ❌ |

---

## Deploying to Azure

### Option 1 — Azure App Service

```bash
# Build and deploy via Azure CLI
az webapp up \
  --name <your-app-name> \
  --resource-group <your-rg> \
  --runtime PYTHON:3.11 \
  --sku B2

# Set environment variables in App Service
az webapp config appsettings set \
  --name <your-app-name> \
  --resource-group <your-rg> \
  --settings \
    AZURE_SEARCH_ENDPOINT="https://..." \
    AZURE_SEARCH_API_KEY="..." \
    AZURE_OPENAI_ENDPOINT="https://..." \
    AZURE_OPENAI_API_KEY="..."
```

Set the startup command in App Service configuration:

```
python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0
```

### Option 2 — Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

```bash
docker build -t rag-agent .
docker run -p 8501:8501 --env-file .env rag-agent
```
---

## Common Issues

**Startup failed: Missing environment variables**
Your `.env` file is either missing, in the wrong directory, or has incorrect variable names. The `.env` file must be in the same directory as `app.py`. Check that variable names match exactly — they are case-sensitive.

**Index creation failed: 403 Forbidden**
The API key you provided is a Query key, not an Admin key. Index creation requires an Admin key. In the Azure Portal, go to your Search service → Keys → copy the Primary admin key.

**Upload failed: scanned / image-only PDF**
PyMuPDF cannot extract text from PDFs that contain only scanned images. The PDF must have a text layer. Use Adobe Acrobat or an OCR tool to add a text layer before uploading.

**Evaluation tab shows install prompt**
Install the optional evaluation dependency: `pip install azure-ai-evaluation==0.3.3` and restart the app.

**Chunks retrieved: 0 on every query**
The index exists but is empty — documents were not indexed successfully. Check the sidebar for error messages from the upload step. If the upload appeared to succeed, verify in the Azure Portal under your Search service → Indexes → your index name → Document count.