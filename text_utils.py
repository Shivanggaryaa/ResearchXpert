"""
Utility helpers for cleaning raw PDF text and splitting it into
overlapping chunks that are ready for vector embeddings.
"""

import re
from typing import List


def clean_text(text: str) -> str:
    """
    Collapse excessive whitespace and new‑lines so the downstream
    chunker sees a tidy, continuous block of text.
    """
    # replace multiple new lines with a single newline
    text = re.sub(r"\n{2,}", "\n", text)
    # collapse all other runs of whitespace into one space
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> List[str]:
    """
    Split `text` into word‑level chunks.

    Parameters
    ----------
    text : str
        Cleaned text from the research paper.
    chunk_size : int
        Target number of words per chunk (default 400).
    overlap : int
        How many words to repeat from the previous chunk (default 80).

    Returns
    -------
    List[str]
        List of chunk strings, in order.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    words = text.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap

    return chunks
