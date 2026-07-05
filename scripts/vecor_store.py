import os
import requests
import logging
from typing import List, Dict, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import uuid as uuid_lib

logger = logging.getLogger(__name__)

GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

COLLECTION_NAME = "proinsight_chunks"
EMBEDDING_DIM = 3072  # text-embedding-004 output size

qdrant_client = None
if QDRANT_URL and QDRANT_API_KEY:
    try:
        qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        existing = [c.name for c in qdrant_client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
        logger.info("Connected to Qdrant")
    except Exception as e:
        logger.error(f"Qdrant connection failed: {e}")
        qdrant_client = None
else:
    logger.warning("Qdrant env vars not set — vector search disabled")


def get_embedding(text: str) -> Optional[List[float]]:
    """Generate an embedding vector for a piece of text using Gemini."""
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set — cannot generate embeddings")
        return None

    try:
        response = requests.post(
            f"{GEMINI_EMBED_URL}?key={GEMINI_API_KEY}",
            json={
                "model": "models/text-embedding-001",
                "content": {"parts": [{"text": text[:8000]}]}  # safety truncation
            },
            timeout=15
        )
        response.raise_for_status()
        return response.json()["embedding"]["values"]
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Split text into chunks, preferring your existing '=== filename ===' section markers."""
    if "===" in text:
        sections = text.split("===")
        chunks = [s.strip() for s in sections if s.strip() and len(s.strip()) > 20]
    else:
        chunks = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
    return chunks


def store_chunks(project_id: str, project_name: str, text: str) -> int:
    """Embed and store text chunks for a project. Returns number of chunks stored."""
    if not qdrant_client:
        logger.warning("Qdrant unavailable, skipping vector storage")
        return 0

    chunks = chunk_text(text)
    points = []

    for idx, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        if vector is None:
            continue
        points.append(PointStruct(
            id=f"{project_id}_{idx}",
            vector=vector,
            payload={
                "project_id": project_id,
                "project_name": project_name,
                "chunk_index": idx,
                "text": chunk[:2000]  # cap payload size
            }
        ))

    if points:
        try:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Stored {len(points)} chunks for project {project_id}")
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            return 0

    return len(points)


def search_similar_chunks(query_text: str, top_k: int = 5, exclude_project_id: str = None) -> List[Dict]:
    """Find the most semantically similar chunks across all stored projects."""
    if not qdrant_client:
        return []

    vector = get_embedding(query_text)
    if vector is None:
        return []

    try:
        results = qdrant_client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            limit=top_k,
        ).points

        matches = []
        for r in results:
            if exclude_project_id and r.payload.get("project_id") == exclude_project_id:
                continue
            matches.append({
                "project_id": r.payload.get("project_id"),
                "project_name": r.payload.get("project_name"),
                "text": r.payload.get("text"),
                "score": r.score
            })
        return matches
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []

def store_chunks(project_id: str, project_name: str, text: str, max_chunks: int = 10) -> int:
    """Embed and store text chunks for a project. Capped to avoid blocking the request too long."""
    if not qdrant_client:
        logger.warning("Qdrant unavailable, skipping vector storage")
        return 0

    chunks = chunk_text(text)[:max_chunks]
    points = []

    for idx, chunk in enumerate(chunks):
        vector = get_embedding(chunk)
        if vector is None:
            continue
        points.append(PointStruct(
            id=str(uuid_lib.uuid4()),  # valid UUID, not "project_id_idx"
            vector=vector,
            payload={
                "project_id": project_id,
                "project_name": project_name,
                "chunk_index": idx,
                "text": chunk[:2000]
            }
        ))

    if points:
        try:
            qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
            logger.info(f"Stored {len(points)} chunks for project {project_id}")
        except Exception as e:
            logger.error(f"Qdrant upsert failed: {e}")
            return 0

    return len(points)
