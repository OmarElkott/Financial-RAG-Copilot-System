# Finance RAG Capstone — Day 1 Design

**Owner:** Omar ElKott  
**Timeline:** Two focused weeks  
**Status:** Initial design; validate assumptions during implementation

## 1. Objective and success criteria

Build a local, production-minded finance question-answering system over 50–100 earnings transcripts, 10-K excerpts, or similar public finance documents. The system must retrieve relevant evidence, generate an answer grounded only in that evidence, attach precise citations, abstain when evidence is weak, expose a FastAPI `/ask` endpoint, log structured request telemetry, support reproducible evaluation, and run with Docker.

The first success milestone is a working vertical slice: ingest a small corpus, index chunks in Qdrant, retrieve evidence, and return a cited answer. Final success requires reproducible evaluation results and a documented demo.

## 2. Initial architecture

```text
Documents
   ↓
Ingestion + validation + deduplication
   ↓
Parser / text normalization
   ↓
Structure-aware chunking + metadata
   ↓
Embedding model
   ↓
Qdrant collection: vectors + payload
   ↓
Question validation + query embedding
   ↓
Qdrant top-k retrieval + optional metadata filters
   ↓
Evidence quality check / abstention decision
   ↓
LLM answer generation with citation requirements
   ↓
Citation validation + structured response
   ↓
FastAPI /ask + structured logs + evaluation traces
```

The local deployment will use an API container and a Qdrant container with persistent storage. Ingestion and evaluation will be repeatable commands rather than separate production services.

## 3. Data and ingestion

Start with 10–20 documents for the vertical slice, then expand to 50–100 after the pipeline is stable. Prefer consistent document types initially, such as earnings-call transcripts or annual-report excerpts, to reduce parser variability.

The ingestion pipeline will:

- Assign a stable `document_id` derived from source identity and document version.
- Store raw files separately from normalized text and indexed chunks.
- Validate required metadata and reject or quarantine malformed documents.
- Normalize whitespace while preserving headings, paragraphs, tables where possible, and page or section boundaries.
- Deduplicate documents before indexing.
- Be idempotent: rerunning ingestion must not create duplicate Qdrant points.

Each chunk will contain:

```text
document_id, source_uri, company, ticker, document_type,
fiscal_period, published_at, page_or_section, chunk_id, text
```

The metadata is intentionally designed for both citations and future filtering by company, date, document type, or fiscal period.

## 4. Chunking and embeddings

Use structure-aware chunking where possible: keep headings with their following content and avoid splitting a table or a short financial statement row from its label. Begin with an approximately 500-token target and 50-token overlap, then test alternatives during retrieval evaluation. Chunk size is an empirical choice; the decision will be based on recall, context precision, answer quality, and latency rather than preference.

Use one documented embedding model for the baseline. Record its model name, vector dimension, normalization behavior, and version in configuration and evaluation results. Do not mix embedding models within one Qdrant collection.

## 5. Qdrant indexing and retrieval

Create one versioned collection using cosine similarity unless experiments show a different metric is more appropriate. Each point will have:

- A deterministic point ID based on `document_id` and `chunk_id`.
- The embedding vector.
- The complete citation and filtering payload.

The initial retrieval strategy is dense top-`k` search with `k = 5`. The API may accept a bounded `k` override for experiments, but production defaults remain configuration-controlled. Add payload indexes for fields used frequently in filters, such as ticker, document type, and fiscal period.

Retrieval output must remain visible to the application before generation. For every result, retain chunk ID, document ID, source metadata, similarity score, and text. This allows retrieval quality to be evaluated independently from answer quality.

## 6. Answer generation and citations

The generation prompt will instruct the model to:

- Use only the supplied retrieved context.
- Distinguish reported facts from interpretation.
- Avoid unsupported calculations or claims.
- State when the evidence is insufficient.
- Attach one or more citation identifiers to each factual claim.

The API response will include:

```text
answer: string
citations: [{document_id, source_uri, page_or_section, chunk_id}]
abstained: boolean
retrieved_chunks: integer
latency_ms: number
```

After generation, validate that cited chunk IDs exist in the retrieved set. A response with unsupported citation identifiers is invalid and should be converted to an error or abstention rather than returned as trustworthy output.

## 7. Weak-evidence fallback

The system will abstain when no results are returned, the best retrieval score is below a tuned threshold, the evidence does not match the requested company or period, or generated claims cannot be mapped to retrieved chunks. The initial threshold will be treated as a tunable baseline and selected using answerable and unanswerable evaluation questions.

Example fallback:

> I don’t have enough evidence in the indexed documents to answer that reliably. Try specifying a company, reporting period, or document type.

The system must not silently use the model’s general knowledge when the corpus lacks evidence.

## 8. Evaluation plan

Create 20–30 questions with expected answers, relevant document or chunk IDs, and an `answerable` label. Include direct lookups, multi-chunk questions, period comparisons, numerical questions, ambiguous questions, and out-of-corpus questions.

Evaluate retrieval separately using:

- Recall@k: whether relevant evidence appears in the top k.
- Precision@k or context precision: how much retrieved context is relevant.
- MRR: rank of the first relevant result.

Evaluate answers using:

- Faithfulness: claims are supported by retrieved context.
- Citation accuracy: citations support the claims they reference.
- Answer correctness against the expected answer.
- Abstention correctness for unanswerable questions.

Keep a manually reviewed sample for numerical answers and citations. Report the dataset size, scoring method, model versions, and known limitations.

## 9. Observability and reliability

Use structured JSON logs with `request_id`, timestamp, model and collection versions, result count, top score, stage latencies, total latency, abstention status, and error type. Do not log secrets or full document text by default.

Add timeouts and explicit error handling around embedding, Qdrant, and generation calls. Validate API input length and return stable error responses. Include health checks for the API and Qdrant.

## 10. Deployment and documentation

Provide a Dockerfile and Compose configuration for the API and Qdrant, including persistent Qdrant storage and environment-variable configuration. The README must document installation, ingestion, indexing, evaluation, API startup, example requests, expected responses, and troubleshooting.

The final repository will include an architecture diagram, evaluation results, representative successful and abstention examples, design trade-offs, and limitations.

## 11. Initial decisions to validate

- Initial corpus: earnings transcripts or 10-K excerpts with stable source metadata.
- Initial chunking: approximately 500 tokens with 50-token overlap.
- Initial retrieval: dense Qdrant search, cosine similarity, top 5 chunks.
- Initial fallback: no results or empirically tuned low-evidence threshold.
- Initial deployment: local Docker Compose with API and Qdrant.
- Explicitly deferred: hybrid retrieval, reranking, frontend, multi-provider abstraction, and online monitoring.

These are baselines, not irreversible commitments. Every change will be justified with evaluation results, latency impact, or operational evidence.
