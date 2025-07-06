# scholar_utils.py
import re
import requests

def _clean_query(text: str, max_len: int = 200) -> str:
    """Keep only letters, numbers, and spaces; truncate to max_len."""
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def find_related_papers(raw_query: str, limit: int = 5):
    """
    Search Semantic Scholar for papers related to `raw_query`.
    Returns list[dict] with title, authors, year, url.
    """
    query = _clean_query(raw_query)

    if not query or len(query) < 4:  # too short → no search
        return []

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,url",
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    results = []
    for p in data.get("data", []):
        # Sometimes authors list is empty
        author_names = ", ".join(a["name"] for a in p.get("authors", [])) or "Unknown"
        results.append(
            {
                "title": p.get("title", "Untitled"),
                "authors": author_names,
                "year": p.get("year", "n.d."),
                "url": p.get("url", "#"),
            }
        )
    return results
