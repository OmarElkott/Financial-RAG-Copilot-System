from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# Initialize once at module load
client = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

COLLECTION_NAME = "finance_chunks"

def retrieve(query: str, top_k: int = 5):
    # 1. Embed the query
    query_vector = model.encode(query).tolist()

    # 2. Query Qdrant
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    )

    # 3. Convert to a simple list of dicts
    chunks = []
    for hit in results.points:
        chunks.append({
            "chunk_id": hit.payload["chunk_id"],
            "document_id": hit.payload["document_id"],
            "text": hit.payload["text"],
            "ticker": hit.payload.get("ticker"),
            "company": hit.payload.get("company"),
            "document_type": hit.payload.get("document_type"),
            "fiscal_period": hit.payload.get("fiscal_period"),
            "source_uri": hit.payload.get("source_uri"),
            "score": hit.score,
        })
    return chunks