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
    p.add_argument("--extract-model", default=ollama_client.EXTRACTION_MODEL,
                   help="Ollama model that structures CV/JD text (default: %(default)s)")
    p.add_argument("--score-model", default=ollama_client.SCORING_MODEL,
                   help="Ollama model that scores each requirement (default: %(default)s)")
    return p


def list_cv_files(folder: str, exclude: Optional[str] = None) -> Tuple[List[str], List[dict]]:
    """Split a folder's top-level files into CV candidates and skipped (unsupported) files.

    `exclude` (the JD path) is left out even if it sits inside the CV folder.
    """
    files, skipped = [], []
    skip_real = os.path.realpath(exclude) if exclude else None
    for name in sorted(os.listdir(folder)):
        full = os.path.join(folder, name)
        if name.startswith(".") or not os.path.isfile(full) or os.path.realpath(full) == skip_real:
            continue
        if name.lower().endswith(file_parser.SUPPORTED_EXTENSIONS):
            files.append(full)
        else:
            skipped.append({"file": name, "reason": "unsupported file type"})
    return files, skipped


def _run_stage(items: list, fn, skipped: List[dict], ollama_errors: List[str]) -> list:
    """Apply fn(item) to each (path, ...) item; a failing CV is recorded as skipped, never aborts the run."""
    out = []
    for item in items:
        name = os.path.basename(item[0])
        try:
            out.append(fn(item))
        except (file_parser.ParseError, extractor.ExtractionError, ollama_client.OllamaError) as exc:
            print("Warning: skipping %s: %s" % (name, exc), file=sys.stderr)
            skipped.append({"file": name, "reason": str(exc)})
            if isinstance(exc, ollama_client.OllamaError):
                ollama_errors.append(str(exc))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """Run the pipeline; return a process exit code.

    Work is staged so each Ollama model is loaded once: extract all CVs (3B), embed all (embedder),
    then score all (7B). The candidate is named after the CV file, not the LLM-extracted name.
    """
    args = build_parser().parse_args(argv)
    if not os.path.isdir(args.cvs):
        print("Error: CV folder not found: %s" % args.cvs, file=sys.stderr)
        return 1
    host = args.ollama_host
    ollama_client.EXTRACTION_MODEL = args.extract_model
    ollama_client.SCORING_MODEL = args.score_model
    try:
        jd = extractor.extract_jd(host, file_parser.parse_file(args.jd))
    except (file_parser.ParseError, extractor.ExtractionError, ollama_client.OllamaError) as exc:
        print("Error: cannot process job description: %s" % exc, file=sys.stderr)
        return 1
    except OSError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    reqs = jd["requirements"]
    if not reqs:
        print("Error: the job description has no requirements to score against.", file=sys.stderr)
        return 1

    files, skipped = list_cv_files(args.cvs, exclude=args.jd)
    ollama_errors = []
    extracted = _run_stage([(p,) for p in files],
                           lambda it: (it[0], extractor.extract_cv(host, file_parser.parse_file(it[0]))),
                           skipped, ollama_errors)
    try:
        req_vecs = matcher.embed_requirements(host, reqs)
    except ollama_client.OllamaError as exc:
        print("Error: %s" % exc, file=sys.stderr)
        return 1
    embedded = _run_stage(extracted, lambda it: (it[0], it[1], matcher.embed_cv(host, it[1])),
                          skipped, ollama_errors)

    def score(item):
        """Score one embedded CV against every requirement."""
        path, cv, chunk_vecs = item
        rows = []
        for req, rv in zip(reqs, req_vecs):
            scored = matcher.score_requirement(host, req, cv, chunk_vecs, rv)
            rows.append(dict(scored, text=req["text"], type=req["type"]))
        base = os.path.basename(path)
        return {"name": os.path.splitext(base)[0], "file": base, "rows": rows}

    candidates = _run_stage(embedded, score, skipped, ollama_errors)
    if ollama_errors and not candidates:
        print("Error: %s" % ollama_errors[0], file=sys.stderr)
        return 1
    ranked = ranker.rank(candidates)
    print(ranker.format_console(ranked, skipped))
    try:
        ranker.write_csv(args.output, ranked, skipped)
    except OSError as exc:
        print("Error: cannot write output file %s: %s" % (args.output, exc), file=sys.stderr)
        return 1
    print("\nFull results written to %s" % args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
