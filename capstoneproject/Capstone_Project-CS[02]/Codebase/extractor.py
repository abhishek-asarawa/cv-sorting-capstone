"""Use the extraction LLM to turn raw CV / JD text into structured dicts."""
import json
import re

import ollama_client

JD_SCHEMA = ('{"role_title": "string", "requirements": [{"text": "short requirement", '
             '"type": "must-have or nice-to-have"}]}')
CV_SCHEMA = ('{"skills": ["skill"], "experience": [{"title": "", "company": "", '
             '"duration": "", "description": ""}], "education": [{"degree": "", "institution": "", "year": ""}]}')
STRICT = "\nReturn ONLY one valid JSON object. No markdown, no commentary, no trailing text."
PLACEHOLDERS = ("", "string", "null", "none", "n/a", "na", "unknown")
NICE_WORDS = ("nice", "preferred", "optional", "bonus", "plus")


class ExtractionError(Exception):
    """Raised when the LLM never produced usable JSON (reason: 'extraction failed')."""


def parse_json_loose(text: str) -> dict:
    """Parse a JSON object from LLM output, tolerating ``` fences and surrounding prose."""
    cleaned = re.sub(r"```(?:json)?", "", text)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    obj = json.loads(cleaned[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("JSON is not an object")
    return obj


def _call(host: str, prompt: str, build):
    """Ask the extraction model for JSON and run `build(data)`; retry once with a stricter prompt.

    `build` raises ValueError for unusable content, which counts like malformed JSON.
    After the retry fails, ExtractionError is raised.
    """
    for attempt in range(2):
        full = prompt + (STRICT if attempt else "")
        raw = ollama_client.generate(host, ollama_client.EXTRACTION_MODEL, full, json_mode=True)
        try:
            return build(parse_json_loose(raw))
        except ValueError:
            continue
    raise ExtractionError("extraction failed")


def _s(value) -> str:
    """Coerce any value to a stripped string (None -> '')."""
    return "" if value is None else str(value).strip()


def _clean(value) -> str:
    """Like _s, but schema placeholders the model echoed back ('string', 'N/A', ...) become ''."""
    text = _s(value)
    return "" if text.lower() in PLACEHOLDERS else text


def _dicts(items, keys):
    """Keep dict entries of a list projected onto `keys` (cleaned strings); drop all-empty entries."""
    if not isinstance(items, list):
        return []
    out = [{k: _clean(i.get(k)) for k in keys} for i in items if isinstance(i, dict)]
    return [e for e in out if any(e.values())]


def _build_jd(data: dict) -> dict:
    """Normalise the JD JSON: string or dict requirements, labels folded to must/nice."""
    items = data.get("requirements")
    reqs = []
    for r in items if isinstance(items, list) else []:
        if isinstance(r, str):
            r = {"text": r, "type": "must-have"}
        if not isinstance(r, dict) or not _clean(r.get("text")):
            continue
        kind = "nice-to-have" if any(w in _s(r.get("type")).lower() for w in NICE_WORDS) else "must-have"
        reqs.append({"text": _clean(r["text"]), "type": kind})
    return {"role_title": _clean(data.get("role_title")), "requirements": reqs}


def _build_cv(data: dict) -> dict:
    """Normalise the CV JSON; ValueError if nothing real remains (e.g. only echoed placeholders)."""
    skills = data.get("skills")
    if isinstance(skills, str):
        skills = [skills]
    cv = {
        "skills": [_clean(s) for s in skills if _clean(s)] if isinstance(skills, list) else [],
        "experience": _dicts(data.get("experience"), ("title", "company", "duration", "description")),
        "education": _dicts(data.get("education"), ("degree", "institution", "year")),
    }
    if not (cv["skills"] or cv["experience"] or cv["education"]):
        raise ValueError("no usable CV content")
    return cv


def extract_jd(host: str, text: str) -> dict:
    """Extract {role_title, requirements[{text, type}]} from job-description text."""
    prompt = ("Extract the job title and every requirement from this job description. Mark each "
              "requirement 'must-have' (required/essential) or 'nice-to-have' (preferred/bonus). "
              "Keep each requirement short.\nSchema: %s\n\nJOB DESCRIPTION:\n%s" % (JD_SCHEMA, text))
    return _call(host, prompt, _build_jd)


def extract_cv(host: str, text: str) -> dict:
    """Extract {skills, experience, education} from CV text (the candidate is named by file, not by LLM)."""
    prompt = ("Extract the candidate's skills, work experience and education from this CV. Use only "
              "facts stated in the CV; leave fields empty rather than inventing or copying the schema "
              "words.\nSchema: %s\n\nCV:\n%s" % (CV_SCHEMA, text))
    return _call(host, prompt, _build_cv)
