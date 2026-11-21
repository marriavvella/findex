from fastapi.testclient import TestClient
from findex.index import InvertedIndex
from findex.web.app import app, get_index


def test_web_endpoints():
    idx = InvertedIndex()
    app.dependency_overrides[get_index] = lambda: idx
    try:
        with TestClient(app) as client:
            assert client.get("/search", params={"q": "hello"}).status_code == 200
            assert client.get("/search", params={"q": ""}).status_code == 422
            assert client.get("/search", params={"q": "hello", "k": 0}).status_code == 422
            assert client.get("/search", params={"q": "hello", "scorer": "bad"}).status_code == 422
            assert client.get("/docs/99999").status_code == 404
            assert client.get("/stats").status_code == 200
            assert client.get("/").status_code == 200
    finally:
        app.dependency_overrides.clear()


def test_health_unavailable():
    with TestClient(app) as client:
        client.app.state.index = None
        assert client.get("/health").status_code == 503
