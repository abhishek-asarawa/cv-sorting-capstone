"""Use the extraction LLM to turn raw CV / JD text into structured dicts."""
import json
import re

import ollama_client

JD_SCHEMA = ('{"role_title": "string", "requirements": [{"text": "short requirement", '
             '"type": "must-have or nice-to-have"}]}')
CV_SCHEMA = ('{"name": "string", "skills": ["string"], "experience": [{"title": "", "company": "", '
             '"duration": "", "description": ""}], "education": [{"degree": "", "institution": "", "year": ""}]}')
STRICT = "\nReturn ONLY one valid JSON object. No markdown, no commentary, no trailing text."
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


def _call(host: str, prompt: str) -> dict:
    """Ask the extraction model for JSON; retry once with a stricter prompt, else ExtractionError."""
    for attempt in range(2):
        full = prompt + (STRICT if attempt else "")
        raw = ollama_client.generate(host, ollama_client.EXTRACTION_MODEL, full, json_mode=True)
        try:
            return parse_json_loose(raw)
        except ValueError:
            continue
    raise ExtractionError("extraction failed")


def _s(value) -> str:
    """Coerce any value to a stripped string (None -> '')."""
    return "" if value is None else str(value).strip()


def _dicts(items, keys):
    """Keep only dict entries of a list, projected onto `keys` with string values."""
    if not isinstance(items, list):
        return []
    return [{k: _s(i.get(k)) for k in keys} for i in items if isinstance(i, dict)]


def extract_jd(host: str, text: str) -> dict:
    """Extract {role_title, requirements[{text, type}]} from job-description text."""
    prompt = ("Extract the job title and every requirement from this job description. Mark each "
              "requirement 'must-have' (required/essential) or 'nice-to-have' (preferred/bonus). "
              "Keep each requirement short.\nSchema: %s\n\nJOB DESCRIPTION:\n%s" % (JD_SCHEMA, text))
    data = _call(host, prompt)
    reqs = []
    items = data.get("requirements")
    for r in items if isinstance(items, list) else []:
        if isinstance(r, str):
            r = {"text": r, "type": "must-have"}
        if not isinstance(r, dict) or not _s(r.get("text")):
            continue
        kind = "nice-to-have" if any(w in _s(r.get("type")).lower() for w in NICE_WORDS) else "must-have"
        reqs.append({"text": _s(r["text"]), "type": kind})
    return {"role_title": _s(data.get("role_title")), "requirements": reqs}


def extract_cv(host: str, text: str) -> dict:
    """Extract {name, skills, experience, education} from CV text."""
    prompt = ("Extract the candidate's details from this CV. Use empty strings/lists for anything "
              "missing; do not invent facts.\nSchema: %s\n\nCV:\n%s" % (CV_SCHEMA, text))
    data = _call(host, prompt)
    skills = data.get("skills")
    if isinstance(skills, str):
        skills = [skills]
    return {
        "name": _s(data.get("name")),
        "skills": [_s(s) for s in skills if _s(s)] if isinstance(skills, list) else [],
        "experience": _dicts(data.get("experience"), ("title", "company", "duration", "description")),
        "education": _dicts(data.get("education"), ("degree", "institution", "year")),
    }
