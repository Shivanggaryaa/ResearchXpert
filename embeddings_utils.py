"""
Helpers for embedding text chunks and querying them with FAISS.
"""

from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Load the lightweight MiniLM model once at import time.
# ------------------------------------------------------------------
_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_model = SentenceTransformer(_MODEL_NAME)

# ------------------------------------------------------------------
# Embedding + FAISS helpers
# ------------------------------------------------------------------
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Turn a list of strings into a float32 NumPy array of shape
    (len(texts), embedding_dim).
    """
    emb = _model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return emb.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build an in‑memory FAISS index using inner‑product search
    (works like cosine similarity once vectors are L2‑normalised).
    """
    # Normalise in‑place for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)  # type: ignore[arg-type]
    return index


def get_top_k(
    query: str,
    chunks: List[str],
    index: faiss.IndexFlatIP,
    k: int = 3,
) -> List[Tuple[int, float]]:
    """
    Embed `query`, search the index, and return (chunk_id, score) pairs.
    """
    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)

    scores, ids = index.search(q_emb, k)
    return [
        (int(idx), float(score))
        for idx, score in zip(ids[0], scores[0])
        if idx != -1
    ]
