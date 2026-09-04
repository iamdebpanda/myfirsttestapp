import re
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup


def extract_text_from_url(url: str, timeout: int = 15) -> Tuple[str, str]:
    """
    Fetch webpage content and extract cleaned text and title.
    Returns (title, cleaned_text).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    
    response = requests.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Extract title
    title = soup.title.string.strip() if soup.title and soup.title.string else url

    # Remove non-content tags
    for element in soup(["script", "style", "noscript", "svg", "header", "footer", "nav", "aside"]):
        element.extract()

    # Get body or full html text
    target = soup.body if soup.body else soup
    text = target.get_text(separator="\n")

    # Clean up excessive blank lines and whitespace
    cleaned_lines = []
    for line in text.splitlines():
        clean_line = re.sub(r"\s+", " ", line).strip()
        if clean_line:
            cleaned_lines.append(clean_line)

    cleaned_text = "\n".join(cleaned_lines)
    return title, cleaned_text


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks suitable for embeddings.
    """
    if not text.strip():
        return []

    # If text is already smaller than chunk_size, return it directly
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        
        # If not at the end of the text, try to break at a natural punctuation or newline
        if end < text_length:
            # Look backwards from end for natural split points
            split_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind(" ", start, end),
            ]
            
            valid_splits = [idx for idx in split_candidates if idx != -1 and idx > start + (chunk_size // 2)]
            if valid_splits:
                end = max(valid_splits) + 1  # include punctuation/space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by (chunk_size - chunk_overlap)
        start = max(start + 1, end - chunk_overlap)

    return chunks
