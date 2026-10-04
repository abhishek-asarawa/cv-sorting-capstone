import pytest
import extractor
from extractor import extract_cv, extract_jd, parse_json_loose, ExtractionError


def test_parse_json_loose_handles_fences_and_prose():
    assert parse_json_loose('Sure!\n```json\n{"a": 1}\n```\nDone') == {"a": 1}


def test_parse_json_loose_rejects_garbage():
    with pytest.raises(ValueError):
        parse_json_loose("no json here")


def _patch(monkeypatch, replies):
    calls = []

    def fake(host, model, prompt, json_mode=False):
        calls.append((model, prompt))
        return replies[min(len(calls) - 1, len(replies) - 1)]

    monkeypatch.setattr(extractor.ollama_client, "generate", fake)
    return calls


def test_extract_jd_normalizes(monkeypatch):
    _patch(monkeypatch, ['{"role_title":"Dev","requirements":[{"text":"Python","type":"Must-Have"},'
                         '{"text":"Docker","type":"preferred"},{"text":"","type":"must-have"}]}'])
    jd = extract_jd("h", "jd text")
    assert jd["role_title"] == "Dev"
    assert jd["requirements"] == [{"text": "Python", "type": "must-have"},
                                  {"text": "Docker", "type": "nice-to-have"}]


def test_extract_jd_retries_once_then_succeeds(monkeypatch):
    calls = _patch(monkeypatch, ["garbage", '{"role_title":"D","requirements":[{"text":"x","type":"must-have"}]}'])
    assert extract_jd("h", "t")["requirements"][0]["text"] == "x"
    assert len(calls) == 2 and calls[0][1] != calls[1][1]


def test_extract_fails_after_retry(monkeypatch):
    calls = _patch(monkeypatch, ["garbage"])
    with pytest.raises(ExtractionError, match="extraction failed"):
        extract_cv("h", "cv text")
    assert len(calls) == 2


def test_extract_cv_fills_missing_fields(monkeypatch):
    _patch(monkeypatch, ['{"skills": "Python"}'])
    cv = extract_cv("h", "t")
    assert "name" not in cv
    assert cv["skills"] == ["Python"]
    assert cv["experience"] == [] and cv["education"] == []


def test_extract_cv_ignores_non_dict_entries(monkeypatch):
    _patch(monkeypatch, ['{"skills": ["a", 3], "experience": ["oops", {"title": "Eng"}], "education": []}'])
    cv = extract_cv("h", "t")
    assert cv["skills"] == ["a", "3"]
    assert cv["experience"] == [{"title": "Eng", "company": "", "duration": "", "description": ""}]


def test_jd_with_zero_requirements_returns_empty_list(monkeypatch):
    _patch(monkeypatch, ['{"role_title":"D","requirements":[]}'])
    assert extract_jd("h", "t")["requirements"] == []


def test_jd_string_requirements_accepted(monkeypatch):
    _patch(monkeypatch, ['{"role_title":"D","requirements":["Python","SQL"]}'])
    assert extract_jd("h", "t")["requirements"] == [{"text": "Python", "type": "must-have"},
                                                    {"text": "SQL", "type": "must-have"}]


def test_jd_non_list_requirements_does_not_crash(monkeypatch):
    _patch(monkeypatch, ['{"role_title":"D","requirements":5}'])
    assert extract_jd("h", "t")["requirements"] == []


PLACEHOLDER_CV = '{"skills": ["string"], "experience": [{"title": "string", "company": "", "duration": "", "description": ""}], "education": []}'


def test_cv_that_only_echoes_schema_placeholders_fails_after_retry(monkeypatch):
    calls = _patch(monkeypatch, [PLACEHOLDER_CV])
    with pytest.raises(ExtractionError, match="extraction failed"):
        extract_cv("h", "t")
    assert len(calls) == 2


def test_cv_placeholder_retry_can_recover(monkeypatch):
    _patch(monkeypatch, [PLACEHOLDER_CV, '{"skills": ["Python"]}'])
    assert extract_cv("h", "t")["skills"] == ["Python"]


def test_placeholders_dropped_next_to_real_values(monkeypatch):
    _patch(monkeypatch, ['{"skills": ["string", "SQL", "N/A"], "experience": [], "education": []}'])
    assert extract_cv("h", "t")["skills"] == ["SQL"]
