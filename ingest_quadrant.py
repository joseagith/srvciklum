import os
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# -------- CONFIG --------
BASE_DIR = r"C:\Capstone insurance"
CHUNKS_METADATA_FILE = os.path.join(BASE_DIR, "chunks", "chunks_metadata.json")

QDRANT_URL = "http://localhost:6333"   # adjust if needed
QDRANT_COLLECTION = "insurance_guidelines"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 outputs 384-dim vectors
# ------------------------


def load_chunks():
    path = Path(CHUNKS_METADATA_FILE)
    if not path.exists():
        raise FileNotFoundError(f"chunks_metadata.json not found at {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def init_qdrant_client() -> QdrantClient:
    client = QdrantClient(
        url=QDRANT_URL,  # or host="localhost", port=6333
        prefer_grpc=False,
    )
    return client


def recreate_collection(client: QdrantClient):
    """
    Create or recreate the collection with the right vector size and metric.
    This will **drop** existing data in that collection.
    """
    print(f"Recreating collection: {QDRANT_COLLECTION}")
    client.recreate_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM,
            distance=models.Distance.COSINE,
        ),
    )


def ingest_chunks():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from metadata.")

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    client = init_qdrant_client()
    recreate_collection(client)

    points = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        vec = model.encode(text, convert_to_numpy=True)
        vec = vec.astype(np.float32)

        # Use index or chunk_id as point ID
        point_id = i  # or hash(chunk["chunk_id"])

        payload = {
            "chunk_id": chunk["chunk_id"],
            "doc_id": chunk["doc_id"],
            "chunk_file": chunk["chunk_file"],
            "title": chunk["title"],
            "section_title": chunk["section_title"],
            "section_index": chunk["section_index"],
            "department": chunk.get("department", ""),
            "doc_type": chunk.get("doc_type", ""),
            "version": chunk.get("version", ""),
            "effective_date": chunk.get("effective_date", ""),
            "text": text,
        }

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vec.tolist(),
                payload=payload,
            )
        )

    print(f"Uploading {len(points)} points to Qdrant...")
    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
    )
    print("Ingestion complete.")


def main():
    ingest_chunks()


if __name__ == "__main__":
    main()
