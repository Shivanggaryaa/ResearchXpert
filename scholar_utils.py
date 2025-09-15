import re
import requests
import time

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
MAX_QUERY_LEN = 150  # characters to safely avoid too long queries
MAX_RETRIES = 3      # number of retries if API rate-limits

def _clean_query(text: str, max_len: int = MAX_QUERY_LEN) -> str:
    """
    Clean and shorten the query for Semantic Scholar API.
    - Remove special characters
    - Replace multiple spaces with single space
    - Truncate to max_len characters
    """
    text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]

def find_related_papers(raw_query: str, limit: int = 5):
    """
    Given a text snippet (like paper title or summary), fetch related papers.
    Handles API 429 rate-limit with retries.
    Returns a list of dicts: [{title, authors, year, url}, ...]
    """
    query = _clean_query(raw_query)
    if not query or len(query) < 4:
        return []

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,url"
    }

    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(SEMANTIC_SCHOLAR_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            results = []

            for p in data.get("data", []):
                author_names = ", ".join(a.get("name", "Unknown") for a in p.get("authors", []))
                results.append({
                    "title": p.get("title", "Untitled"),
                    "authors": author_names or "Unknown",
                    "year": p.get("year", "n.d."),
                    "url": p.get("url", "#"),
                })
            return results

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429:
                wait_time = 2 ** attempt  # exponential backoff: 1,2,4 seconds
                print(f"Semantic Scholar API rate-limit hit. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Semantic Scholar API error: {e}")
                break
        except Exception as e:
            print(f"Semantic Scholar API error: {e}")
            break

    # If we reach here, either retries exhausted or permanent failure
    return []
