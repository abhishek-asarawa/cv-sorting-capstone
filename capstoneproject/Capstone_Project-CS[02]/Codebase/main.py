"""CLI entry point: rank CVs against a job description using local Ollama LLMs."""
import argparse
import os
import sys
from typing import List, Optional, Tuple

import extractor
import file_parser
import matcher
import ollama_client
import ranker

DEFAULT_HOST = "http://localhost:11434"


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line interface."""
    p = argparse.ArgumentParser(description="Rank CVs against a job description using local LLMs.")
    p.add_argument("--cvs", required=True, help="folder containing CV files (.pdf / .txt)")
    p.add_argument("--jd", required=True, help="job description file (.pdf / .txt)")
    p.add_argument("--output", default="results.csv", help="CSV output path (default: results.csv)")
    p.add_argument("--ollama-host", default=DEFAULT_HOST, help="Ollama base URL")
    return p


def list_cv_files(folder: str) -> Tuple[List[str], List[dict]]:
    """Split a folder's top-level files into CV candidates and skipped (unsupported) files."""
    files, skipped = [], []
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if name.startswith(".") or not os.path.isfile(full):
            continue
        if name.lower().endswith(file_parser.SUPPORTED_EXTENSIONS):
            files.append(full)
        else:
            skipped.append({"file": name, "reason": "unsupported file type"})
    return files, skipped


def process_cv(host: str, path: str, requirements: List[dict]) -> dict:
    """Parse, extract and score one CV; may raise ParseError/ExtractionError (caller skips)."""
    text = file_parser.parse_file(path)
    cv = extractor.extract_cv(host, text)
    if not cv["name"]:
        cv["name"] = os.path.splitext(os.path.basename(path))[0]
    chunk_vecs = matcher.embed_cv(host, cv)
    rows = []
    for req in requirements:
        scored = matcher.score_requirement(host, req, cv, chunk_vecs)
        rows.append(dict(scored, text=req["text"], type=req["type"]))
    return {"name": cv["name"], "file": os.path.basename(path), "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    """Run the pipeline; return a process exit code."""
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.cvs):
        print("Error: CV folder not found: %s" % args.cvs, file=sys.stderr)
        return 1
    try:
        jd = extractor.extract_jd(args.ollama_host, file_parser.parse_file(args.jd))
    except (file_parser.ParseError, extractor.ExtractionError, ollama_client.OllamaError) as exc:
        print("Error: cannot process job description: %s" % exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    if not jd["requirements"]:
        print("Error: the job description has no requirements to score against.", file=sys.stderr)
        return 1

    files, skipped = list_cv_files(args.cvs)
    candidates = []
    for path in files:
        try:
            candidates.append(process_cv(args.ollama_host, path, jd["requirements"]))
        except (file_parser.ParseError, extractor.ExtractionError) as exc:
            print("Warning: skipping %s: %s" % (os.path.basename(path), exc), file=sys.stderr)
            skipped.append({"file": os.path.basename(path), "reason": str(exc)})
        except ollama_client.OllamaError as exc:
            print("Error: %s" % exc, file=sys.stderr)
            return 1
    ranked = ranker.rank(candidates)
    print(ranker.format_console(ranked, skipped))
    ranker.write_csv(args.output, ranked, skipped)
    print("\nFull results written to %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
