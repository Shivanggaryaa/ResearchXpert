import os
import requests
from typing import List, Tuple
from dotenv import load_dotenv
from embeddings_utils import get_top_k
from collections import OrderedDict

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY missing — add it to .env")

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.1-8b-instant"

def format_response(raw_response: str, keywords: List[str] = []) -> str:
    """
    Post-process LLM response:
    - Merge duplicate lines
    - Number key contributions/method steps
    - Bold key terms
    """
    lines = [line.strip() for line in raw_response.split("\n") if line.strip()]
    
    # Remove duplicate lines while preserving order
    seen = set()
    cleaned_lines = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            cleaned_lines.append(line)

    formatted = []
    for idx, line in enumerate(cleaned_lines, 1):
        # Detect if it looks like a key point and number it
        if any(line.lower().startswith(prefix) for prefix in ["dataset", "method", "evaluation", "objective", "challenge", "future work", "key"]):
            line = f"{idx}. {line}"
        else:
            # Make it a regular bullet
            if not line.startswith("•") and not line.startswith("-"):
                line = f"• {line}"
        # Bold keywords
        for kw in keywords:
            line = line.replace(kw, f"**{kw}**")
        formatted.append(line)
    return "\n".join(formatted)

def answer_with_groq(
    question: str,
    chunks: List[str],
    index,
    k: int = 4,
    max_tokens: int = 400,
    keywords: List[str] = [],
) -> Tuple[str, List[Tuple[int, float]]]:
    """
    Ask Groq LLM a question about a research paper using selected chunks.
    Returns (formatted_answer, hits)
    """
    hits = get_top_k(question, chunks, index, k=k)

    max_total_tokens = 4000
    total_words = 0
    selected_chunks = []
    for i, (idx, _) in enumerate(hits):
        chunk_words = chunks[idx].split()
        if total_words + len(chunk_words) > max_total_tokens:
            break
        selected_chunks.append(f"Excerpt {i+1}:\n" + " ".join(chunk_words))
        total_words += len(chunk_words)

    system_msg = """
You are an AI research assistant. Answer questions about ONE research paper using ONLY the provided excerpts.
Instructions:
- Use concise bullet points.
- Merge duplicate points.
- Bold important keywords using markdown (**term**).
- Number methodology/contributions where relevant.
- Cite excerpts like (Excerpt 1), (Excerpt 2), etc.
- Only include information present in the excerpts.
- If the answer is unknown, reply: "The document does not provide this information."
"""

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

    raw_answer = res["choices"][0]["message"]["content"].strip()
    formatted_answer = format_response(raw_answer, keywords=keywords)
    return formatted_answer, hits