import logging, time, uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langfuse import get_client

from logging_config import setup_logging
setup_logging()  # must run before the app handles requests

from generation import answer_with_citations

logger = logging.getLogger("rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield                     # the app runs here
    get_client().shutdown()   # on shutdown, flush remaining Langfuse traces


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)  # cap length to limit cost
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    chunk_id: str
    document_id: str
    source_uri: str | None = None
    ticker: str | None = None
    document_type: str | None = None
    fiscal_period: str | None = None


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    abstained: bool
    retrieved_chunks: int
    request_id: str
    trace_id: str | None = None
    latency_ms: float


@app.get("/")
def root():
    return {"message": "Finance RAG API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    request_id = str(uuid.uuid4())
    query = request.query.strip()

    # A whitespace-only query passes min_length, so reject it here
    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query must not be empty",
            headers={"X-Request-ID": request_id},
        )

    start = time.perf_counter()
    try:
        result = answer_with_citations(
            question=query,
            top_k=request.top_k,
            request_id=request_id,
        )
    except Exception as e:
        logger.error(
            "ask_failed",
            exc_info=True,
            extra={"fields": {
                "request_id": request_id,
                "error_type": type(e).__name__,
                "latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }},
        )
        # Generic message to the user; details stay in logs and Langfuse
        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer at this time.",
            headers={"X-Request-ID": request_id},
        ) from None

    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    logger.info("ask_completed", extra={"fields": {
        "request_id": request_id,
        "trace_id": result.get("trace_id"),
        "query_len": len(query),
        "top_k": request.top_k,
        "retrieved_chunks": result["retrieved_chunks"],
        "top_score": result.get("top_score"),
        "abstained": result["abstained"],
        "abstain_reason": result.get("abstain_reason"),
        "n_citations": len(result["citations"]),
        "latency_ms": latency_ms,
        **result.get("timings", {}),
    }})

    citations = [Citation(**c) for c in result["citations"]]
    return AskResponse(
        answer=result["answer"],
        citations=citations,
        abstained=result["abstained"],
        retrieved_chunks=result["retrieved_chunks"],
        request_id=request_id,
        trace_id=result.get("trace_id"),
        latency_ms=latency_ms,
    )