import pytest
from file_parser import parse_file, ParseError


def test_reads_txt(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("Jane Doe\nPython")
    assert parse_file(str(p)) == "Jane Doe\nPython"


def test_empty_txt_rejected(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("   \n")
    with pytest.raises(ParseError, match="empty"):
        parse_file(str(p))


def test_unsupported_extension(tmp_path):
    p = tmp_path / "a.docx"
    p.write_text("x")
    with pytest.raises(ParseError, match="unsupported"):
        parse_file(str(p))


def test_corrupt_pdf(tmp_path):
    p = tmp_path / "a.pdf"
    p.write_bytes(b"not a pdf")
    with pytest.raises(ParseError, match="PDF"):
        parse_file(str(p))


def test_pdf_with_no_text_is_empty(tmp_path, monkeypatch):
    import file_parser

    class FakePage:
        def extract_text(self):
            return None

    class FakePdf:
        pages = [FakePage()]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(file_parser.pdfplumber, "open", lambda p: FakePdf())
    p = tmp_path / "scan.pdf"
    p.write_bytes(b"x")
    with pytest.raises(ParseError, match="empty"):
        parse_file(str(p))


def test_txt_non_utf8_falls_back(tmp_path):
    p = tmp_path / "a.txt"
    p.write_bytes("caf\xe9".encode("latin-1"))
    assert "caf" in parse_file(str(p))
