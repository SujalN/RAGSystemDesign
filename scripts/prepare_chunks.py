# scripts/prepare_chunks.py
import re
from pathlib import Path
from typing import List
from dotenv import load_dotenv
load_dotenv()

RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
CHUNK_DIR = Path(__file__).parent.parent / "data" / "chunks"

# adjust these to taste
MAX_TOKENS    = 1000
OVERLAP_TOKENS = 200

def clean_text(text: str) -> str:
    # remove extra whitespace, headers, footers, etc.
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def chunk_text(tokens: List[str]) -> List[List[str]]:
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + MAX_TOKENS
        chunks.append(tokens[start:end])
        start += MAX_TOKENS - OVERLAP_TOKENS
    return chunks

def main():
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    for txt in RAW_DIR.glob("*.txt"):
        raw = txt.read_text(encoding="utf-8")
        cleaned = clean_text(raw)
        tokens = cleaned.split()  # simple whitespace tokenizer; replace with HuggingFace if you like
        for i, chunk in enumerate(chunk_text(tokens)):
            out_path = CHUNK_DIR / f"{txt.stem}_chunk{i:03d}.txt"
            out_path.write_text(" ".join(chunk), encoding="utf-8")
            print(f"  → wrote {out_path.name}")

if __name__ == "__main__":
    main()