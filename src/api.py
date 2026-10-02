from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation import answer_with_citations

app = FastAPI()


class AskRequest(BaseModel):
    query: str
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


@app.get("/")
def root():
    return {"message": "Finance RAG API is running"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    # Basic input validation
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")

    # Call your existing generation logic
    result = answer_with_citations(
        question=request.query.strip(),
        top_k=request.top_k,
    )

    # Convert citation dicts to Pydantic models
    citations = [
        Citation(
            chunk_id=c["chunk_id"],
            document_id=c["document_id"],
            source_uri=c.get("source_uri"),
            ticker=c.get("ticker"),
            document_type=c.get("document_type"),
            fiscal_period=c.get("fiscal_period"),
        )
        for c in result["citations"]
    ]

    return AskResponse(
        answer=result["answer"],
        citations=citations,
        abstained=result["abstained"],
        retrieved_chunks=result["retrieved_chunks"],
    )