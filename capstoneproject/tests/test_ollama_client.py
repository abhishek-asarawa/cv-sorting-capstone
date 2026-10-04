import pytest
import requests
import ollama_client as oc


class FakeResp:
    def __init__(self, data, status=200):
        self._d, self.status_code = data, status
        self.text = str(data)

    def json(self):
        return self._d


def test_generate_returns_response_text(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen.update(url=url, body=json)
        return FakeResp({"response": "hello"})

    monkeypatch.setattr(requests, "post", fake_post)
    out = oc.generate("http://h:1", "m", "p", json_mode=True)
    assert out == "hello"
    assert seen["url"] == "http://h:1/api/generate"
    assert seen["body"]["model"] == "m" and seen["body"]["stream"] is False
    assert seen["body"]["format"] == "json"


def test_embed_returns_vector(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp({"embeddings": [[0.1, 0.2]]}))
    assert oc.embed("http://h:1", "x") == [0.1, 0.2]


def test_connection_error_becomes_ollama_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(oc.OllamaError, match="Cannot reach Ollama"):
        oc.generate("http://h:1", "m", "p")


def test_http_error_status_becomes_ollama_error(monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: FakeResp({"error": "model not found"}, 404))
    with pytest.raises(oc.OllamaError, match="model not found"):
        oc.generate("http://h:1", "m", "p")


def test_generate_sets_context_window(monkeypatch):
    seen = {}
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: seen.update(b=json) or FakeResp({"response": "x"}))
    oc.generate("http://h:1", "m", "p")
    assert seen["b"]["options"]["num_ctx"] >= 8192


def test_generate_caps_output_tokens(monkeypatch):
    seen = {}
    monkeypatch.setattr(requests, "post", lambda url, json=None, timeout=None: seen.update(b=json) or FakeResp({"response": "x"}))
    oc.generate("http://h:1", "m", "p")
    assert 0 < seen["b"]["options"]["num_predict"] <= 4096


def test_timeout_is_reported_as_timeout_not_unreachable(monkeypatch):
    def slow(*a, **k):
        raise requests.ReadTimeout("slow")

    monkeypatch.setattr(requests, "post", slow)
    with pytest.raises(oc.OllamaError, match="timed out") as ei:
        oc.generate("http://h:1", "m", "p")
    assert "Cannot reach" not in str(ei.value)
