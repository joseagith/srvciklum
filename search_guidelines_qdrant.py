import os
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

# -------- CONFIG --------
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "insurance_guidelines"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# ------------------------


class QdrantGuidelineSearcher:
    def __init__(self):
        self.client = QdrantClient(
            url=QDRANT_URL,
            prefer_grpc=False,
        )
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _encode_query(self, query: str) -> List[float]:
        vec = self.model.encode([query], convert_to_numpy=True)[0]
        vec = vec.astype(np.float32)
        return vec.tolist()

    def _build_filter(self, filters: Dict[str, Any]) -> Optional[models.Filter]:
        """
        Convert simple equality filters (department, policy_type, doc_type, etc.)
        into a Qdrant Filter with must conditions.
        """
        conditions = []
        for key, value in filters.items():
            if not value:
                continue
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )

        if not conditions:
            return None

        return models.Filter(must=conditions)

    def search_guidelines(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
        score_threshold: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Search Qdrant for relevant INSURANCE guideline chunks.
        Returns list of dicts with chunk_id, text, metadata, and score.
        """
        if not query.strip():
            return []

        filters = filters or {}
        query_vector = self._encode_query(query)
        q_filter = self._build_filter(filters)

        # Using .search() is valid for qdrant-client 1.5.1
        results = self.client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            query_filter=q_filter,
            limit=top_k,
        )

        # results: list[ScoredPoint], each has .score and .payload
        output = []
        for point in results:
            score = float(point.score)
            if score < score_threshold:
                continue

            payload = point.payload or {}
            output.append(
                {
                    "score": score,
                    "chunk_id": payload.get("chunk_id"),
                    "doc_id": payload.get("doc_id"),
                    "title": payload.get("title"),
                    "section_title": payload.get("section_title"),
                    "section_index": payload.get("section_index"),
                    "department": payload.get("department"),
                    "policy_type": payload.get("policy_type"),
                    "doc_type": payload.get("doc_type"),
                    "version": payload.get("version"),
                    "effective_date": payload.get("effective_date"),
                    "text": payload.get("text"),
                }
            )

        # Sort defensively (Qdrant already returns sorted)
        output.sort(key=lambda x: x["score"], reverse=True)
        return output


def demo_cli():
    searcher = QdrantGuidelineSearcher()
    print("\nInsurance guideline search demo. Type your question, or 'exit' to quit.\n")

    while True:
        query = input("Question> ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break

        # Simple heuristic-based filter inference for demo (INSURANCE)
        filters: Dict[str, Any] = {}
        q_lower = query.lower()

        # Underwriting-related
        if any(w in q_lower for w in ["underwriting", "health policy", "pre-existing", "medical test"]):
            filters["department"] = "Underwriting"

        # Motor / auto claims
        elif any(w in q_lower for w in ["motor", "vehicle", "garage", "survey", "accident claim"]):
            filters["department"] = "Claims"

        # Fraud
        elif any(w in q_lower for w in ["fraud", "red flag", "investigation", "suspicious"]):
            filters["department"] = "Fraud Control"

        # Renewal / lapse
        elif any(w in q_lower for w in ["renewal", "lapse", "reinstatement", "grace period"]):
            filters["department"] = "Customer Service"

        # Life claims
        elif any(w in q_lower for w in ["life claim", "death certificate", "nominee", "accidental death"]):
            filters["department"] = "Life Claims"

        print(f"\nUsing filters: {filters or 'none'}")

        results = searcher.search_guidelines(
            query=query,
            filters=filters,
            top_k=3,
            score_threshold=0.3,
        )

        if not results:
            print("No relevant insurance guideline sections found above threshold.")
            continue

        print("\nTop results:")
        for i, res in enumerate(results, start=1):
            print(f"\n--- Result {i} ---")
            print(f"Score       : {res['score']:.3f}")
            print(f"Doc ID      : {res['doc_id']}")
            print(f"Title       : {res['title']}")
            print(f"Section     : {res['section_title']}")
            print(f"Department  : {res.get('department')}")
            print(f"Policy Type : {res.get('policy_type')}")
            print(f"Doc Type    : {res.get('doc_type')}")
            snippet = (res["text"] or "").replace("\n", " ")
            print(f"Snippet     : {snippet[:400]}...")
        print("\n")


if __name__ == "__main__":
    demo_cli()
