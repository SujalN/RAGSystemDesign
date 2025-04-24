import os
import pytest
from retriever.retriever import Retriever

@pytest.fixture
def retriever():
    return Retriever(top_k=1)

def test_retrieve_returns_list(retriever):
    docs = retriever.retrieve("anything")
    assert isinstance(docs, list)
    # Each entry should be a tuple (id, score, snippet)
    if docs:
        _id, score, snippet = docs[0]
        assert isinstance(_id, str)
        assert isinstance(score, float)
        assert isinstance(snippet, str)
