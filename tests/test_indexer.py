import os
import pytest
from embeddings.indexer import process_chunk, CHUNK_DIR

@pytest.mark.parametrize("chunk_file", list(CHUNK_DIR.glob("*.txt"))[:2])
def test_process_chunk_does_not_throw(chunk_file, monkeypatch):
    # Monkey‐patch the actual upsert so it doesn’t hit Pinecone
    monkeypatch.setattr("embeddings.indexer.index.upsert", lambda *args, **kwargs: None)
    # Also stub out embeddings API
    monkeypatch.setattr("embeddings.indexer.embed_text", lambda text: [0.0]*1536)
    # Should complete without exception
    process_chunk(chunk_file)
