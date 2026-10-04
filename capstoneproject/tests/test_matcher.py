import math
import pytest
import matcher


def test_cosine_basic():
    assert matcher.cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert matcher.cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert matcher.cosine([0, 0], [1, 1]) == 0.0


def test_to_percent_clamps():
    assert matcher.to_percent(-0.5) == 0.0
    assert matcher.to_percent(0.5) == 50.0
    assert matcher.to_percent(1.2) == 100.0


def test_combine_formula():
    assert matcher.combine(100, 0) == pytest.approx(60.0)
    assert matcher.combine(50, 100) == pytest.approx(70.0)


def test_cv_chunks_and_empty():
    cv = {"name": "J", "skills": ["Py", "SQL"],
          "experience": [{"title": "Dev", "company": "X", "duration": "2y", "description": "built"}],
          "education": [{"degree": "BTech", "institution": "IITK", "year": "2020"}]}
    chunks = matcher.cv_chunks(cv)
    assert len(chunks) == 3 and "Py, SQL" in chunks[0] and "Dev" in chunks[1] and "BTech" in chunks[2]
    assert matcher.cv_chunks({"name": "", "skills": [], "experience": [], "education": []}) == []


def test_similarity_score_takes_best_chunk_and_handles_none():
    assert matcher.similarity_score([1, 0], [("a", [0, 1]), ("b", [1, 0])]) == pytest.approx(100.0)
    assert matcher.similarity_score([1, 0], []) == 0.0


def test_parse_llm_score_variants():
    assert matcher.parse_llm_score('{"score": 85, "explanation": "good"}') == (85.0, "good")
    assert matcher.parse_llm_score('```json\n{"score": "72", "explanation": "ok"}\n```') == (72.0, "ok")
    assert matcher.parse_llm_score('{"score": 250, "explanation": "x"}')[0] == 100.0
    assert matcher.parse_llm_score('{"score": -5, "explanation": "x"}')[0] == 0.0
    with pytest.raises(ValueError):
        matcher.parse_llm_score('{"explanation": "no score"}')
    with pytest.raises(ValueError):
        matcher.parse_llm_score("nonsense")


def _req():
    return {"text": "Python", "type": "must-have"}


def test_score_requirement_combines(monkeypatch):
    monkeypatch.setattr(matcher.ollama_client, "embed", lambda h, t: [1, 0])
    monkeypatch.setattr(matcher.ollama_client, "generate",
                        lambda h, m, p, json_mode=False: '{"score": 80, "explanation": "Has Python"}')
    r = matcher.score_requirement("h", _req(), {"name": "J"}, [("c", [1, 0])])
    assert r["L"] == 80.0 and r["E"] == 100.0
    assert r["S"] == pytest.approx(0.6 * 80 + 0.4 * 100)
    assert r["explanation"] == "Has Python"


def test_score_requirement_survives_bad_llm(monkeypatch):
    monkeypatch.setattr(matcher.ollama_client, "embed", lambda h, t: [1, 0])
    monkeypatch.setattr(matcher.ollama_client, "generate", lambda *a, **k: "garbage")
    r = matcher.score_requirement("h", _req(), {"name": "J"}, [("c", [1, 0])])
    assert r["L"] == 0.0 and r["explanation"] == "scoring failed"
    assert r["S"] == pytest.approx(40.0)


def test_embed_cv_embeds_each_chunk(monkeypatch):
    monkeypatch.setattr(matcher.ollama_client, "embed", lambda h, t: [float(len(t))])
    out = matcher.embed_cv("h", {"name": "", "skills": ["Py"], "experience": [], "education": []})
    assert len(out) == 1 and out[0][0].startswith("Skills")


def test_score_retry_uses_stricter_prompt_and_warns(monkeypatch, capsys):
    prompts = []
    monkeypatch.setattr(matcher.ollama_client, "embed", lambda h, t: [1, 0])
    monkeypatch.setattr(matcher.ollama_client, "generate",
                        lambda h, m, p, json_mode=False: prompts.append(p) or "garbage")
    matcher.score_requirement("h", _req(), {"name": "J"}, [("c", [1, 0])])
    assert len(prompts) == 2 and prompts[0] != prompts[1]
    assert "scoring failed" in capsys.readouterr().err
