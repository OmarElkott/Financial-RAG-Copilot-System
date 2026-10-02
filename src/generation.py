from pathlib import Path
from openai import OpenAI
from retrieval import retrieve
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# llm = OpenAI()
llm = Groq()

# MODEL_NAME = "gpt-4o-mini"
MODEL_NAME = "openai/gpt-oss-20b"
MIN_SCORE = 0.35

def retrieve_chunks(query, top_k=5):
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer": "I do not have enough evidence to answer that.",
            "citations": [],
            "abstained": True,
            "retrieved_chunks": 0,
        }

    if chunks[0]["score"] < MIN_SCORE:
        return {
            "answer": "I do not have enough evidence in the indexed documents to answer that reliably.",
            "citations": [],
            "abstained": True,
            "retrieved_chunks": len(chunks),
        }
    
    return chunks
    
def build_context(chunks):
    chunk_text = []
    for i, chunk in enumerate(chunks, start=1):
        chunk_text.append(f"[{i}] {chunk['text']}")
    context = "\n\n".join(chunk_text)
    return context
    
def build_prompt(context, question):

    system_prompt = """
    You are a financial research assistant.

    Answer the user's question using only the supplied context.
    Do not use outside knowledge.
    Cite every factual claim with the context number, such as [1] or [2].
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

def generate_answer(system_prompt, user_prompt):
    response = llm.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
    
    answer = response.choices[0].message.content
    
    return answer

def extract_citations(answer_text: str, chunks: list[dict]) -> list[dict]:
    cited_indices = set(map(int, re.findall(r"\[(\d+)\]", answer_text)))

    citations = []
    for idx in sorted(cited_indices):
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

def answer_with_citations(question: str, top_k: int = 5) -> dict:
    retrieved = retrieve_chunks(question, top_k=top_k)

    # If abstention already decided
    if isinstance(retrieved, dict) and retrieved.get("abstained"):
        return retrieved

    chunks = retrieved  # list of dicts

    context = build_context(chunks)
    system_prompt, user_prompt = build_prompt(context, question)
    answer_text = generate_answer(system_prompt, user_prompt)
    citations = extract_citations(answer_text, chunks)

    return {
        "answer": answer_text,
        "citations": citations,
        "abstained": False,
        "retrieved_chunks": len(chunks),
    }
