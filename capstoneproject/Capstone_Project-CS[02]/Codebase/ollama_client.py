"""Thin wrapper around the Ollama REST API (plain `requests`)."""
from typing import List

import requests

EXTRACTION_MODEL = "llama3.2:3b"
SCORING_MODEL = "qwen2.5:7b-instruct"
EMBED_MODEL = "nomic-embed-text"
TIMEOUT = 300
NUM_CTX = 8192  # Ollama truncates silently past its small default window


class OllamaError(Exception):
    """Raised when Ollama is unreachable or returns an error."""


def _post(host: str, path: str, body: dict) -> dict:
    """POST JSON to Ollama and return the decoded reply, raising OllamaError on any failure."""
    try:
        resp = requests.post(host.rstrip("/") + path, json=body, timeout=TIMEOUT)
    except requests.RequestException as exc:
        raise OllamaError("Cannot reach Ollama at %s: %s" % (host, exc))
    try:
        data = resp.json()
    except ValueError:
        raise OllamaError("Ollama returned non-JSON reply (HTTP %s)" % resp.status_code)
    if resp.status_code != 200:
        raise OllamaError("Ollama error (HTTP %s): %s" % (resp.status_code, data.get("error", data)))
    return data


def generate(host: str, model: str, prompt: str, json_mode: bool = False) -> str:
    """Run a non-streaming completion with `model`; json_mode asks Ollama to emit valid JSON."""
    body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0, "num_ctx": NUM_CTX}}
    if json_mode:
        body["format"] = "json"
    return _post(host, "/api/generate", body)["response"]


def embed(host: str, text: str, model: str = EMBED_MODEL) -> List[float]:
    """Return the embedding vector for `text`."""
    return _post(host, "/api/embed", {"model": model, "input": text})["embeddings"][0]
