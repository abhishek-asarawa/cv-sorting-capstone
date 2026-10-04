# CV Sorting using LLMs — Capstone Project Design

**Date:** 2026-10-04
**Project ID:** CS02 / HPPCS02 (confirm exact prefix on course portal before final submission)
**Deadline:** 08 October 2026, 23:59 IST

## 1. Objective

Build a Python CLI program that ranks candidate CVs against a job description (JD),
using two local LLMs plus a deterministic embedding-similarity signal, producing a
ranked list with per-candidate, per-requirement explanations.

Satisfies capstone requirements: individual project, ≥2 justified LLMs, Python,
terminal-executable, `main.py` entry point, no GUI.

## 2. Models

All models run locally via Ollama (`http://localhost:11434`), called via plain
`requests` (no heavy SDK). All three are already pulled locally.

| Role | Model | Why |
|---|---|---|
| Extraction LLM | `llama3.2:3b` | Structuring CV/JD text into JSON is pattern-matching, not deep reasoning — a 3B instruction-tuned model handles this reliably. Going smaller (~1B) risks inconsistent JSON formatting across varied resume layouts. |
| Scoring LLM | `qwen2.5:7b-instruct` | Match scoring + explanation needs real reasoning across multiple criteria. 7B instruct-tuned gives strong instruction-following without the overhead of a 14B model or a reasoning/thinking model. |
| Embedding model | `nomic-embed-text` | Lightweight (274MB), used for deterministic cosine-similarity between JD requirement text and CV content — grounds the scoring LLM's judgment, matching real-world hybrid resume-matching systems rather than trusting LLM judgment in isolation. |

`deepseek-r1` variants were considered and rejected for this project: they are
reasoning/thinking models that emit long chain-of-thought tokens before answering,
adding latency/memory cost without benefit for this bounded scoring task.

## 3. Architecture / Pipeline

```
main.py --cvs <cv_folder> --jd <jd_file> [--output results.csv] [--ollama-host http://localhost:11434]
   │
   ├─► 1. Load JD file → Extraction LLM (llama3.2:3b)
   │        → structured JD: {role_title, requirements: [{text, type: must-have|nice-to-have}]}
   │
   ├─► 2. For each CV file in cv_folder:
   │        a. Parse file (PDF via pdfplumber / .txt direct read)
   │        b. If unparseable → log warning, add to "skipped" list, continue to next CV
   │           (never abort the whole run on one bad file — matches real-world ATS/parsing practice)
   │        c. Extraction LLM (llama3.2:3b) → structured CV:
   │           {name, skills: [], experience: [{title, company, duration, description}],
   │            education: [{degree, institution, year}]}
   │        d. For each JD requirement:
   │             - Embedding model (nomic-embed-text) → cosine similarity between
   │               requirement text and candidate's relevant CV section → E_i (0-100, deterministic)
   │             - Scoring LLM (qwen2.5:7b-instruct) → qualitative score L_i (0-100) + explanation,
   │               given the requirement, candidate data, and E_i as context
   │             - Combined score: S_i = 0.6·L_i + 0.4·E_i
   │                 (LLM-weighted: the LLM is the primary scorer per the project brief's explicit
   │                 instruction to "ask the LLM to score the match"; embedding grounds/sanity-checks
   │                 it, matching real-world hybrid systems that blend LLM + embedding signals,
   │                 e.g. 30%/70% embedding/LLM splits seen in production resume matchers.)
   │        e. Deterministic aggregation:
   │             w_i = 3 if requirement is must-have, else 1
   │             overall_score = Σ(w_i · S_i) / Σ(w_i)
   │           (must-have requirements weighted ~3x nice-to-have, matching how production
   │           ATS/resume-matching systems penalize missing required skills far more than
   │           missing preferred ones.)
   │
   ├─► 3. Rank all successfully-scored candidates by overall_score, descending
   │
   └─► 4. Output:
            - Console: ranked table (rank, name, overall_score, top 2-3 reasons)
            - results.csv: full detail per candidate (L_i, E_i, S_i, w_i per requirement + overall_score)
            - Skipped files listed separately (console + CSV section), never silently dropped,
              never scored (a parse failure is a data problem, not a ranking signal)
```

## 4. Edge Cases (deterministic layer)

- Zero must-have requirements in JD → fall back to weighting all requirements equally (w_i = 1 for all)
- Zero nice-to-have requirements → unaffected, must-haves simply dominate naturally
- Empty extracted skill list for a candidate → still scored (low similarity/LLM scores naturally rank them low), not treated as a parse failure
- No division by zero: aggregation guards `Σ(w_i) == 0` (only possible if a JD has zero requirements — treat as a hard error, log and exit, since there's nothing to score against)

## 5. File Structure

Flat layout per capstone rule (no subdirectories inside Codebase; all files reachable via `./`):

```
Capstone_Project-CS[02]/
├── Report/
│   └── report.pdf
└── Codebase/
    ├── main.py           # CLI entry point, orchestrates pipeline, writes console/CSV output
    ├── file_parser.py    # PDF/text file → raw text
    ├── extractor.py      # raw text → structured CV/JD JSON via extraction LLM
    ├── matcher.py        # embedding similarity + scoring LLM calls, produces S_i per requirement
    ├── ranker.py         # weighted aggregation (overall_score) + sorting + CSV/console formatting
    ├── ollama_client.py  # thin wrapper around Ollama REST API, shared by extractor.py/matcher.py
    └── execution.txt     # exact run command(s)
```

## 6. CLI

```
python main.py --cvs <cv_folder> --jd <jd_file> [--output results.csv] [--ollama-host http://localhost:11434]
```

No API key needed (fully local via Ollama) — satisfies capstone's "no hardcoded API key" rule trivially,
though `--ollama-host` is exposed as a parameter in case an evaluator runs Ollama on a non-default host/port.

## 7. Output Formats

- **Console**: ranked table — rank, candidate name, overall_score, 2-3 line summary of top matching/missing requirements
- **results.csv**: one row per (candidate, requirement) with columns: `candidate`, `requirement_text`, `requirement_type`, `L_i`, `E_i`, `S_i`, plus a summary row/section per candidate with `overall_score` and rank; a separate section/rows for skipped files with the reason
- Input/output format is self-defined per FAQ #68 ("you can define your own input and output formats... clearly define them in the report and execution instructions") — documented in both `execution.txt` and the report's Methodology section

## 8. Error Handling

- Unparseable CV (corrupt PDF, empty file, unreadable text) → log warning to console with filename + reason, skip, continue processing remaining CVs, list in a separate "skipped" section of the output
- Malformed LLM JSON output → one retry with a stricter re-prompt; if still malformed, treat that CV as skipped with reason "extraction failed"
- Never abort the entire run on a single bad file — matches real-world resume-parsing pipeline practice (production systems explicitly avoid blocking an entire batch on one bad document)

## 9. Testing Approach

No formal test suite required by the capstone, but before submission:
- **Unit-level** (deterministic logic only — no LLM calls involved, so fully repeatable): aggregation formula correctness, must-have/nice-to-have weighting, the edge cases in §4, CSV/console formatting
- **End-to-end manual run**: 3-4 sample CVs (mix of PDF/txt, including one intentionally corrupt file) + one sample JD — confirm ranked output is sensible, skipped file is reported not silently dropped, CSV matches console output
- No sample/test data shipped in the final submission ZIP (capstone rule — evaluators supply their own test files)

## 10. Capstone Compliance Checklist

- [x] Individual project, Python only
- [x] ≥2 LLMs used, both justified (§2)
- [x] `main.py` entry point
- [x] `execution.txt` with exact run command
- [x] Code comments on every function/class (to be enforced during implementation)
- [x] No subdirectories in Codebase, all files via `./`
- [x] No hardcoded API key (N/A — fully local, but `--ollama-host` documented as a parameter)
- [x] No sample input/output data shipped
- [x] Final code in `.py` (not `.ipynb`)
- [x] Report ≤3 pages, Times New Roman 12pt, 1.5 spacing, text-only (no images), matches supplied template
- [x] Report stays in sync with actual code (methodology section must describe exactly this pipeline)

## 11. Report Alignment

The report's Methodology / System Design section must describe, concisely:
- The two LLMs used and why (table in §2, condensed)
- The role of the embedding model as a deterministic grounding signal (not counted as one of the "2 LLMs" — it's a similarity/representation model, distinct from a generative LLM)
- The scoring formula (§3) and the weighting rationale (must-have > nice-to-have; LLM-weighted blend with embedding)
- Input/output format definition (§7)
- Limitations (e.g., no skill-taxonomy/synonym normalization — noted as future work rather than implemented, to stay within capstone scope)

## 12. Explicitly Out of Scope (YAGNI, noted as future work in report)

- Full skill-taxonomy/synonym ontology (e.g. "JS" = "JavaScript") — real-world systems use this, but it's disproportionate engineering effort for capstone scope; noted as a limitation/future work instead
- GUI/dashboard — capstone explicitly says not required and recommends avoiding it
- Interactive query refinement / re-running with adjusted criteria — the project brief's "optional" extension, not needed for core objective
