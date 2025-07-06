# qa_utils.py

import os
import requests
from typing import List, Tuple
from dotenv import load_dotenv

from embeddings_utils import get_top_k

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing — add it to .env")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

def answer_with_groq(
    question: str,
    chunks: List[str],
    index,
    k: int = 4,
    max_tokens: int = 400,
) -> Tuple[str, List[Tuple[int, float]]]:
    hits = get_top_k(question, chunks, index, k=k)

    # Limit excerpt size
    max_total_tokens = 4000
    total_words = 0
    selected_chunks = []
    for i, (idx, _) in enumerate(hits):
        chunk_words = chunks[idx].split()
        if total_words + len(chunk_words) > max_total_tokens:
            break
        selected_chunks.append(f"Excerpt {i+1}:\n" + " ".join(chunk_words))
        total_words += len(chunk_words)

    system_msg = (
        "You are an assistant answering questions about ONE research paper. "
        "Use only the provided excerpts and cite them like (Excerpt 1)."
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": "\n\n".join(selected_chunks) + f"\n\nQuestion: {question}"},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }

    res = requests.post(
        GROQ_ENDPOINT,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    ).json()

    if "choices" not in res:
        raise RuntimeError(res.get("error", {}).get("message", "Groq API error"))

    return res["choices"][0]["message"]["content"].strip(), hits
