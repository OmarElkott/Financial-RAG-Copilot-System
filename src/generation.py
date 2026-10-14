import logging, os, re, time
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client, observe, propagate_attributes
from langfuse.openai import OpenAI

from retrieval import retrieve

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

logger = logging.getLogger("rag.generation")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set. Add it to your .env file.")

llm = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MIN_SCORE = 0.35
APP_VERSION = "0.1.0"
REQUIRE_CITATIONS = True   # set to False to return uncited answers
REFUSAL_PREFIX = "i do not have enough evidence"

ABSTAIN_NO_RESULTS = "I do not have enough evidence to answer that."
ABSTAIN_LOW_SCORE = ("I do not have enough evidence in the indexed documents "
                     "to answer that reliably.")

CITATION_PATTERN = re.compile(r"[\[【]\s*(\d+(?:\s*,\s*\d+)*)[^\]】]*[\]】]")


def _abstain(message, n_chunks, reason, top_score=None):
    return {"answer": message, "citations": [], "abstained": True,
            "retrieved_chunks": n_chunks, "abstain_reason": reason,
            "top_score": top_score}


@observe(name="retrieve_chunks")
def retrieve_chunks(query, top_k=5):
    chunks = retrieve(query, top_k=top_k)
    if not chunks:
        return _abstain(ABSTAIN_NO_RESULTS, 0, "no_results")
    if chunks[0]["score"] < MIN_SCORE:
        return _abstain(ABSTAIN_LOW_SCORE, len(chunks), "low_score", chunks[0]["score"])
    return chunks


def build_context(chunks):
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, start=1))


def build_prompt(context, question):
    system_prompt = """
    You are a financial research assistant.

    Answer the user's question using only the supplied context.
    Do not use outside knowledge.
    Cite every factual claim with the context number, such as [1] or [2].
    Use plain ASCII square brackets for citations, like [1]. Never use any other bracket style.
    Quote numbers exactly as they appear in the context, with their units.
    Do not add, subtract, or compute totals yourself. If the exact figure is not stated
    in the context, say you cannot find it.
    If the context does not contain enough evidence, say:
    "I do not have enough evidence in the indexed documents to answer that."
    Do not invent numbers, dates, or conclusions.
    """
    user_prompt = f"""
    Context:
    {context}

    Question:
    {question}

    Provide a concise answer with citations.
    """
    return system_prompt, user_prompt


@observe(name="generate_answer")
def generate_answer(system_prompt, user_prompt):
    try:
        response = llm.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            timeout=30,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    except Exception as e:
        logger.error("llm_call_failed", extra={"fields": {"error_type": type(e).__name__}})
        raise
    return response.choices[0].message.content or ""


def normalize_citations(answer_text):
    """Convert 【1】 or [1, 2] style markers into plain [1][2] markers."""
    def repl(match):
        nums = re.findall(r"\d+", match.group(1))
        return "".join(f"[{n}]" for n in nums)
    return CITATION_PATTERN.sub(repl, answer_text)


@observe(name="extract_citations")
def extract_citations(answer_text, chunks):
    indices = set()
    for group in CITATION_PATTERN.findall(answer_text):
        indices.update(int(n) for n in re.findall(r"\d+", group))

    citations = []
    for idx in sorted(indices):
        if 1 <= idx <= len(chunks):
            c = chunks[idx - 1]
            citations.append({
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "source_uri": c.get("source_uri"),
                "ticker": c.get("ticker"),
                "document_type": c.get("document_type"),
                "fiscal_period": c.get("fiscal_period"),
            })
    return citations


@observe(name="answer_with_citations")
def answer_with_citations(question: str, top_k: int = 5, request_id: str | None = None) -> dict:
    langfuse = get_client()
    t0 = time.perf_counter()
    timings = {}

    with propagate_attributes(
        tags=["finance-rag"],
        version=APP_VERSION,
        metadata={"requestid": request_id or "none", "model": MODEL_NAME},
    ):
        t = time.perf_counter()
        retrieved = retrieve_chunks(question, top_k=top_k)
        timings["retrieval_ms"] = round((time.perf_counter() - t) * 1000, 1)

        if isinstance(retrieved, dict):
            result = retrieved
            top_score = result.get("top_score")
        else:
            chunks = retrieved
            top_score = chunks[0]["score"]

            context = build_context(chunks)
            system_prompt, user_prompt = build_prompt(context, question)

            t = time.perf_counter()
            answer_text = generate_answer(system_prompt, user_prompt)
            timings["generation_ms"] = round((time.perf_counter() - t) * 1000, 1)

            logger.info("llm_answer", extra={"fields": {
                "request_id": request_id,
                "answer_len": len(answer_text),
                "answer_preview": answer_text[:300],
            }})

            citations = extract_citations(answer_text, chunks)
            refused = answer_text.strip().lower().startswith(REFUSAL_PREFIX)

            if refused:
                result = _abstain(ABSTAIN_LOW_SCORE, len(chunks), "model_refused", top_score)
            elif REQUIRE_CITATIONS and not citations:
                result = _abstain(ABSTAIN_LOW_SCORE, len(chunks), "no_citations", top_score)
            else:
                result = {
                    "answer": normalize_citations(answer_text),
                    "citations": citations,
                    "abstained": False,
                    "retrieved_chunks": len(chunks),
                    "abstain_reason": None,
                }

        timings["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)

        langfuse.update_current_span(metadata={
            "top_score": top_score,
            "min_score": MIN_SCORE,
            "retrieved_chunks": result["retrieved_chunks"],
            "n_citations": len(result["citations"]),
            "abstained": result["abstained"],
            "abstain_reason": result.get("abstain_reason"),
            "embedding_model": EMBEDDING_MODEL,
            "llm_model": MODEL_NAME,
            **timings,
        })

        result["timings"] = timings
        result["top_score"] = top_score
        result["trace_id"] = langfuse.get_current_trace_id()
        return result