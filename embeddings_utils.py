from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_model = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(chunks: List[str]) -> np.ndarray:
    emb = _model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    return emb.astype("float32")

def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index

def get_top_k(query: str, chunks: List[str], index: faiss.IndexFlatIP, k: int = 3) -> List[Tuple[int, float]]:
    q_emb = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    scores, ids = index.search(q_emb, k)
    return [(int(idx), float(score)) for idx, score in zip(ids[0], scores[0]) if idx != -1]
