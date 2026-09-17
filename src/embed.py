from sentence_transformers import SentenceTransformer
from pathlib import Path
import json

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "chunks"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "embeddings"

INPUT_PATH = DATA_DIR / "chunks.jsonl"
OUTPUT_PATH = OUTPUT_DIR / "chunks_with_embeddings.jsonl"

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def vector_embed(text: str):
    embedding = model.encode(text)
    return embedding.tolist()  # convert ndarray → list for JSON

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            chunk = json.loads(line)
            text = chunk.get("text", "")
            embedding = vector_embed(text)
            chunk["vector_embedding"] = embedding
            fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()