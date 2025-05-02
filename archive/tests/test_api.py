# tests/test_api.py
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.server import app

client = TestClient(app)

@pytest.mark.parametrize("endpoint,payload", [
    ("/qa", {"q":"test"}),
    ("/summarize", {"q":"test"})
])
def test_endpoints_return_200(endpoint, payload):
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200
    data = response.json()
    if endpoint == "/qa":
        assert "answer" in data and "citations" in data
    else:
        assert "summary" in data
