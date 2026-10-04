import csv
import pytest
import ranker


def row(text, typ, S, L=0, E=0, why="why"):
    return {"text": text, "type": typ, "L": L, "E": E, "S": S, "explanation": why}


def test_weights_must_haves_3x():
    assert ranker.weights(["must-have", "nice-to-have"]) == [3, 1]


def test_weights_all_equal_when_no_must_have():
    assert ranker.weights(["nice-to-have", "nice-to-have"]) == [1, 1]


def test_overall_score_weighted():
    rows = [row("a", "must-have", 100), row("b", "nice-to-have", 0)]
    assert ranker.overall_score(rows) == pytest.approx(75.0)  # (3*100+1*0)/4


def test_overall_score_no_must_have_is_plain_mean():
    rows = [row("a", "nice-to-have", 100), row("b", "nice-to-have", 50)]
    assert ranker.overall_score(rows) == pytest.approx(75.0)


def test_overall_score_empty_is_error():
    with pytest.raises(ValueError, match="no requirements"):
        ranker.overall_score([])


def _cands():
    return [
        {"name": "Low", "file": "l.txt", "rows": [row("Py", "must-have", 20)]},
        {"name": "High", "file": "h.txt", "rows": [row("Py", "must-have", 90)]},
    ]


def test_rank_orders_descending_and_adds_fields():
    r = ranker.rank(_cands())
    assert [c["name"] for c in r] == ["High", "Low"]
    assert [c["rank"] for c in r] == [1, 2]
    assert r[0]["overall"] == 90.0 and r[0]["rows"][0]["w"] == 3


def test_rank_with_no_candidates():
    assert ranker.rank([]) == []


def test_console_lists_ranked_and_skipped():
    out = ranker.format_console(ranker.rank(_cands()), [{"file": "bad.pdf", "reason": "corrupt"}])
    assert out.index("High") < out.index("Low")
    assert "bad.pdf" in out and "corrupt" in out


def test_console_when_all_skipped():
    out = ranker.format_console([], [{"file": "bad.pdf", "reason": "corrupt"}])
    assert "No candidates" in out and "bad.pdf" in out


def test_write_csv_sections(tmp_path):
    p = tmp_path / "o.csv"
    ranker.write_csv(str(p), ranker.rank(_cands()), [{"file": "bad.pdf", "reason": "corrupt"}])
    rows = list(csv.DictReader(open(str(p), newline="")))
    sections = [r["section"] for r in rows]
    assert sections.count("summary") == 2 and sections.count("requirement") == 2 and sections.count("skipped") == 1
    top = [r for r in rows if r["section"] == "summary"][0]
    assert top["candidate"] == "High" and top["rank"] == "1" and top["overall_score"] == "90.0"
    sk = [r for r in rows if r["section"] == "skipped"][0]
    assert sk["file"] == "bad.pdf" and sk["explanation_or_reason"] == "corrupt"


def test_csv_quotes_commas_and_newlines(tmp_path):
    c = [{"name": 'A, "B"', "file": "a.txt", "rows": [row("x,y", "must-have", 50, why="line1\nline2")]}]
    p = tmp_path / "o.csv"
    ranker.write_csv(str(p), ranker.rank(c), [])
    rows = list(csv.DictReader(open(str(p), newline="")))
    assert [r for r in rows if r["section"] == "requirement"][0]["explanation_or_reason"] == "line1\nline2"
