"""Turn a CV/JD file (.pdf or .txt) into plain text."""
import os

import pdfplumber

SUPPORTED_EXTENSIONS = (".pdf", ".txt")


class ParseError(Exception):
    """Raised with a human-readable reason when a file cannot be turned into text."""


def _read_pdf(path: str) -> str:
    """Extract text from every page of a PDF; wrap any library failure in ParseError."""
    try:
        with pdfplumber.open(path) as pdf:
            return "\n".join((page.extract_text() or "") for page in pdf.pages)
    except Exception as exc:
        raise ParseError("could not read PDF (%s)" % type(exc).__name__)


def _read_txt(path: str) -> str:
    """Read a text file as UTF-8, falling back to latin-1 so odd encodings still load."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="latin-1") as f:
            return f.read()
    except OSError as exc:
        raise ParseError("could not read file (%s)" % exc)


def parse_file(path: str) -> str:
    """Return the stripped text of `path`, or raise ParseError (unsupported/corrupt/empty)."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ParseError("unsupported file type '%s'" % ext)
    text = (_read_pdf(path) if ext == ".pdf" else _read_txt(path)).strip()
    if not text:
        raise ParseError("file is empty or has no extractable text")
    return text
