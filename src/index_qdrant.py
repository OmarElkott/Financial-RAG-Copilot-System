import json
from pathlib import Path
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http import models


client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "finance_chunks"

points = []

ROOT = Path(__file__).resolve().parent.parent  # project root
EMBEDDINGS_PATH = ROOT / "data" / "embeddings" / "chunks_with_embeddings.jsonl"

if client.collection_exists(COLLECTION_NAME):
    client.delete_collection(COLLECTION_NAME)

with EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
    line = f.readline()
    chunk = json.loads(line)
    vec = chunk["vector_embedding"]
        
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=len(vec),
        distance=Distance.COSINE,
    ),
)

with EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        chunk = json.loads(line)

        point = PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk["chunk_id"])), 
            vector=chunk["vector_embedding"],
            payload={
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "text": chunk["text"],
                "ticker": chunk.get("ticker"),
                "company": chunk.get("company"),
                "document_type": chunk.get("document_type"),
                "fiscal_period": chunk.get("fiscal_period"),
                "source_uri": chunk.get("source_uri"),
            },
        )
        points.append(point)

client.upsert(collection_name=COLLECTION_NAME, points=points)
