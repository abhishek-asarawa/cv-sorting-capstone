import csv
import os
import main as m
import ollama_client
from file_parser import ParseError


def _fake_pipeline(monkeypatch):
    """Replace every Ollama-touching call with deterministic fakes."""
    monkeypatch.setattr(m.extractor, "extract_jd", lambda h, t: {
        "role_title": "Dev", "requirements": [{"text": "Python", "type": "must-have"},
                                              {"text": "Docker", "type": "nice-to-have"}]})
    monkeypatch.setattr(m.extractor, "extract_cv",
                        lambda h, t: {"skills": ["x"], "experience": [], "education": []})
    monkeypatch.setattr(m.matcher, "embed_cv", lambda h, cv: [])
    monkeypatch.setattr(m.matcher, "embed_requirements", lambda h, reqs: [[1.0] for _ in reqs])
    monkeypatch.setattr(m.matcher, "score_requirement",
                        lambda h, req, cv, vecs, rv=None: {"L": 50.0, "E": 50.0, "S": 50.0, "explanation": "e"})


def _setup(tmp_path):
    cvs = tmp_path / "cvs"
    cvs.mkdir()
    (cvs / "alice.txt").write_text("Alice python")
    (cvs / "bobby.txt").write_text("Bobby python")
    (cvs / "anon.txt").write_text("anon python")
    (cvs / "empty.txt").write_text("")
    (cvs / ".DS_Store").write_bytes(b"x")
    (cvs / "notes.docx").write_text("x")
    (cvs / "subdir").mkdir()
    jd = tmp_path / "jd.txt"
    jd.write_text("Need Python")
    return cvs, jd


def test_end_to_end(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch)
    cvs, jd = _setup(tmp_path)
    out = tmp_path / "r.csv"
    code = m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(out)])
    assert code == 0
    text = capsys.readouterr().out
    assert "bobby" in text and "alice" in text
    assert "empty.txt" in text and "notes.docx" in text
    assert ".DS_Store" not in text and "subdir" not in text
    rows = list(csv.DictReader(open(str(out), newline="")))
    assert sorted(r["file"] for r in rows if r["section"] == "skipped") == ["empty.txt", "notes.docx"]
    names = [r["candidate"] for r in rows if r["section"] == "summary"]
    assert sorted(names) == ["alice", "anon", "bobby"]  # candidate = CV file name


def test_ollama_down_gives_clean_error(tmp_path, monkeypatch, capsys):
    cvs, jd = _setup(tmp_path)

    def boom(h, t):
        raise ollama_client.OllamaError("Cannot reach Ollama at x")

    monkeypatch.setattr(m.extractor, "extract_jd", boom)
    assert m.main(["--cvs", str(cvs), "--jd", str(jd)]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().err


def test_jd_without_requirements_exits_1(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(m.extractor, "extract_jd", lambda h, t: {"role_title": "x", "requirements": []})
    cvs, jd = _setup(tmp_path)
    assert m.main(["--cvs", str(cvs), "--jd", str(jd)]) == 1
    assert "no requirements" in capsys.readouterr().err.lower()


def test_missing_paths_exit_1(tmp_path, capsys):
    assert m.main(["--cvs", str(tmp_path / "nope"), "--jd", str(tmp_path / "jd.txt")]) == 1
    assert "error" in capsys.readouterr().err.lower()


def test_extraction_failure_skips_cv_not_run(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch)

    def flaky(h, t):
        if "Bobby" in t:
            raise m.extractor.ExtractionError("extraction failed")
        return {"skills": ["a"], "experience": [], "education": []}

    monkeypatch.setattr(m.extractor, "extract_cv", flaky)
    cvs, jd = _setup(tmp_path)
    assert m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(tmp_path / "o.csv")]) == 0
    assert "extraction failed" in capsys.readouterr().out


def test_empty_candidate_list_still_writes_csv(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    cvs = tmp_path / "cvs"
    cvs.mkdir()
    jd = tmp_path / "jd.txt"
    jd.write_text("Need Python")
    out = tmp_path / "o.csv"
    assert m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(out)]) == 0
    assert os.path.exists(str(out))


def test_unwritable_output_gives_clean_error(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch)
    cvs, jd = _setup(tmp_path)
    bad = tmp_path / "no" / "such" / "dir" / "o.csv"
    assert m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(bad)]) == 1
    assert "output" in capsys.readouterr().err.lower()


def test_per_cv_ollama_error_skips_that_cv_only(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch)

    def flaky(h, t):
        if "Bobby" in t:
            raise ollama_client.OllamaError("timed out")
        return {"skills": ["a"], "experience": [], "education": []}

    monkeypatch.setattr(m.extractor, "extract_cv", flaky)
    cvs, jd = _setup(tmp_path)
    out = tmp_path / "o.csv"
    assert m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(out)]) == 0
    assert "timed out" in capsys.readouterr().out and os.path.exists(str(out))


def test_all_cvs_failing_with_ollama_error_exits_1(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch)

    def down(h, t):
        raise ollama_client.OllamaError("Cannot reach Ollama")

    monkeypatch.setattr(m.extractor, "extract_cv", down)
    cvs, jd = _setup(tmp_path)
    assert m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(tmp_path / "o.csv")]) == 1
    assert "Cannot reach Ollama" in capsys.readouterr().err


def test_jd_file_inside_cv_folder_is_not_scored_as_cv(tmp_path, monkeypatch, capsys):
    _fake_pipeline(monkeypatch)
    cvs, jd = _setup(tmp_path)
    inside = cvs / "job-description.txt"
    inside.write_text("Need Python")
    out = tmp_path / "o.csv"
    assert m.main(["--cvs", str(cvs), "--jd", str(inside), "--output", str(out)]) == 0
    rows = list(csv.DictReader(open(str(out), newline="")))
    assert "job-description.txt" not in [r["file"] for r in rows]


def test_all_cvs_extracted_before_any_scoring(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    order = []
    monkeypatch.setattr(m.extractor, "extract_cv",
                        lambda h, t: order.append("extract") or {"skills": ["x"], "experience": [], "education": []})
    monkeypatch.setattr(m.matcher, "score_requirement",
                        lambda h, req, cv, vecs, rv=None: order.append("score") or
                        {"L": 1.0, "E": 1.0, "S": 1.0, "explanation": "e"})
    cvs, jd = _setup(tmp_path)
    m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(tmp_path / "o.csv")])
    assert order.index("score") > max(i for i, x in enumerate(order) if x == "extract")


def test_model_flags_override_defaults(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    monkeypatch.setattr(ollama_client, "EXTRACTION_MODEL", ollama_client.EXTRACTION_MODEL)
    monkeypatch.setattr(ollama_client, "SCORING_MODEL", ollama_client.SCORING_MODEL)
    cvs, jd = _setup(tmp_path)
    m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(tmp_path / "o.csv"),
            "--extract-model", "tiny-x", "--score-model", "tiny-s"])
    assert ollama_client.EXTRACTION_MODEL == "tiny-x" and ollama_client.SCORING_MODEL == "tiny-s"


def test_model_defaults_unchanged_without_flags(tmp_path, monkeypatch):
    _fake_pipeline(monkeypatch)
    cvs, jd = _setup(tmp_path)
    m.main(["--cvs", str(cvs), "--jd", str(jd), "--output", str(tmp_path / "o.csv")])
    assert ollama_client.EXTRACTION_MODEL == "llama3.2:3b" and ollama_client.SCORING_MODEL == "qwen2.5:7b-instruct"
