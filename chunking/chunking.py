from pathlib import Path
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"

INPUT_PATH = DATA_DIR / "documents.jsonl"

# Simple, solid defaults for RAG
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # characters
    chunk_overlap=50,    # characters
    separators=["\n\n", "\n", ". ", " ", ""],
)

def chunk_doc(doc: dict):
    text = doc.get("full_text", "")
    if not text:
        return []

    chunks = text_splitter.split_text(text)

    result = []
    for i, chunk_text in enumerate(chunks):
        chunk = {
            "chunk_id": f"{doc['document_id']}_chunk_{i:04d}",
            "document_id": doc["document_id"],
            "text": chunk_text,
            "ticker": doc.get("ticker"),
            "company": doc.get("company"),
            "document_type": doc.get("document_type"),
            "fiscal_period": doc.get("fiscal_period"),
            "source_uri": doc.get("source_uri"),
        }
        result.append(chunk)
    return result

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "chunks.jsonl"

    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            doc = json.loads(line)
            chunks = chunk_doc(doc)
            for chunk in chunks:
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()