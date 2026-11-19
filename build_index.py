"""
Step two: turn data/pages_final.jsonl into a searchable index.

What it does
  1. Splits each page into chunks of roughly 900 characters, keeping headings with their text.
  2. Prefixes every chunk with the page title so retrieval has context.
  3. Embeds all chunks with fastembed (small ONNX model, no PyTorch needed).
  4. Saves:
       index/chunks.jsonl      chunk text plus title, source URL and site
       index/embeddings.npy    one normalised vector per chunk

Run:
  pip install fastembed numpy
  python build_index.py
"""

import json
import re
from pathlib import Path

import numpy as np

DATA_FILE = Path("data/pages_final.jsonl")
INDEX_DIR = Path("index")
MODEL_NAME = "BAAI/bge-small-en-v1.5"

MAX_CHARS = 900
MIN_CHARS = 120
OVERLAP_PARAGRAPHS = 1


def split_long(paragraph, max_chars):
    """Break an oversized paragraph on sentence boundaries."""
    if len(paragraph) <= max_chars:
        return [paragraph]
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    pieces, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) + 1 > max_chars:
            pieces.append(current.strip())
            current = s
        else:
            current = f"{current} {s}".strip()
    if current:
        pieces.append(current.strip())
    return pieces


def split_sections(text):
    """Split page text at markdown headings. Returns a list of (heading, paragraphs)."""
    sections, heading, paras = [], "", []
    for block in re.split(r"\n+", text):  # one line per unit, so lists and tables split cleanly
        block = block.strip()
        if not block:
            continue
        if re.match(r"^#{1,4}\s", block):
            if paras:
                sections.append((heading, paras))
            heading, paras = block.lstrip("#").strip(), []
        else:
            paras.extend(split_long(block, MAX_CHARS))
    if paras:
        sections.append((heading, paras))
    return sections


def chunk_page(text):
    chunks = []
    for heading, paras in split_sections(text):
        current = []
        size = 0
        for para in paras:
            if current and size + len(para) > MAX_CHARS:
                chunks.append((heading, "\n".join(current)))
                tail = current[-OVERLAP_PARAGRAPHS:] if len(current) > 1 else []
                current = [t for t in tail if len(t) <= 250]  # short overlap only, never repeat a whole chunk
                size = sum(len(t) for t in current)
            current.append(para)
            size += len(para)
        if current:
            chunks.append((heading, "\n".join(current)))
    return [(h, t) for h, t in chunks if len(t) >= MIN_CHARS]


def main():
    from fastembed import TextEmbedding

    INDEX_DIR.mkdir(exist_ok=True)
    records = [json.loads(line) for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

    chunks = []
    for rec in records:
        for heading, body in chunk_page(rec["text"]):
            label = rec["title"] if not heading or heading == rec["title"] else f"{rec['title']} | {heading}"
            chunks.append(
                {
                    "id": len(chunks),
                    "site": rec["site"],
                    "title": rec["title"],
                    "url": rec["url"],
                    "text": f"{label}\n{body}",
                }
            )
    print(f"{len(records)} pages -> {len(chunks)} chunks")

    model = TextEmbedding(model_name=MODEL_NAME)
    vectors = np.array(list(model.embed([c["text"] for c in chunks])), dtype="float32")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    np.save(INDEX_DIR / "embeddings.npy", vectors)
    with open(INDEX_DIR / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"Saved {vectors.shape[0]} vectors of dimension {vectors.shape[1]} to {INDEX_DIR}/")


if __name__ == "__main__":
    main()
