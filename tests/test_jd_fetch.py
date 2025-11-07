import types
from src import jd_fetch


class DummyResponse:
    def __init__(self, text="ok", status_code=200):
        self.text = text
        self.status_code = status_code
        self.content = text.encode("utf-8")


def test_fetch_and_extract_basic(monkeypatch):
    def fake_client_get(self, url, headers=None):
        return DummyResponse("<p>Hello world!</p>")

    monkeypatch.setattr(jd_fetch.httpx.Client, "get", fake_client_get)
    text = jd_fetch.fetch_and_extract("https://example.com")
    assert "Hello" in text
