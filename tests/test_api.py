from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_api_status():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"

def test_api_ingest():
    payload = {
        "documents": [
            {
                "content": "LangGraph is a library for building stateful, multi-actor applications with LLMs.",
                "source": "langgraph_doc.txt"
            }
        ]
    }
    response = client.post("/api/documents/ingest", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "stats" in data
    assert data["stats"]["num_documents"] == 1

def test_api_query():
    query_payload = {"query": "What is LangGraph used for?"}
    response = client.post("/api/query", json=query_payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "grounded_score" in data
