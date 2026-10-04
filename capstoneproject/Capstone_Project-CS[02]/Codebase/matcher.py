"""Per-requirement matching: embedding similarity (E) + scoring-LLM judgement (L) -> S."""
import math
import sys
from typing import List, Tuple

import ollama_client
from extractor import parse_json_loose

LLM_WEIGHT = 0.6
EMBED_WEIGHT = 0.4
STRICT = "\nReturn ONLY one JSON object with a numeric \"score\" between 0 and 100. No other text."


def cosine(a: List[float], b: List[float]) -> float:
    """Cosine similarity of two vectors (0.0 if either has zero length)."""
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def to_percent(cos: float) -> float:
    """Map a cosine value to a 0-100 score (negatives clamp to 0)."""
    return min(100.0, max(0.0, cos) * 100.0)


def combine(llm_score: float, embed_score: float) -> float:
    """Blend LLM and embedding scores: S = 0.6*L + 0.4*E."""
    return LLM_WEIGHT * llm_score + EMBED_WEIGHT * embed_score


def cv_chunks(cv: dict) -> List[str]:
    """Split a structured CV into text chunks: skills, each job, each degree."""
    chunks = []
    if cv.get("skills"):
        chunks.append("Skills: " + ", ".join(cv["skills"]))
    for e in cv.get("experience", []):
        chunks.append("Experience: %s at %s (%s). %s" % (e["title"], e["company"], e["duration"], e["description"]))
    for d in cv.get("education", []):
        chunks.append("Education: %s, %s (%s)" % (d["degree"], d["institution"], d["year"]))
    return chunks


def embed_cv(host: str, cv: dict) -> List[Tuple[str, List[float]]]:
    """Embed every CV chunk once so it can be reused for all requirements."""
    return [(c, ollama_client.embed(host, c)) for c in cv_chunks(cv)]


def similarity_score(req_vec: List[float], chunk_vecs: List[Tuple[str, List[float]]]) -> float:
    """E_i: best cosine (as 0-100) between the requirement and any CV chunk; 0 if no chunks."""
    if not chunk_vecs:
        return 0.0
    return to_percent(max(cosine(req_vec, v) for _, v in chunk_vecs))


def parse_llm_score(raw: str) -> Tuple[float, str]:
    """Parse the scoring LLM's JSON into (score clamped to 0-100, explanation); ValueError if unusable."""
    data = parse_json_loose(raw)
    if "score" not in data:
        raise ValueError("missing score")
    try:
        score = float(data["score"])
    except (TypeError, ValueError):
        raise ValueError("score is not numeric")
    return min(100.0, max(0.0, score)), str(data.get("explanation", "")).strip()


def _prompt(req: dict, cv: dict, e_score: float) -> str:
    """Build the scoring prompt for one requirement."""
    return ("You are a strict technical recruiter. Rate how well the candidate satisfies ONE job "
            "requirement, using only the evidence given.\n"
            "Requirement (%s): %s\n"
            "Embedding similarity hint (0-100, a rough signal only): %.0f\n"
            "Candidate: %s\n\n"
            'Reply with JSON only: {"score": <0-100 integer>, "explanation": "<one short sentence>"}'
            % (req["type"], req["text"], e_score, "\n".join(cv_chunks(cv)) or "(no details)"))


def score_requirement(host: str, req: dict, cv: dict, chunk_vecs) -> dict:
    """Return {L, E, S, explanation} for one requirement against one candidate."""
    e_score = similarity_score(ollama_client.embed(host, req["text"]), chunk_vecs)
    llm_score, why = 0.0, "scoring failed"
    for attempt in range(2):
        prompt = _prompt(req, cv, e_score) + (STRICT if attempt else "")
        raw = ollama_client.generate(host, ollama_client.SCORING_MODEL, prompt, json_mode=True)
        try:
            llm_score, why = parse_llm_score(raw)
            break
        except ValueError:
            continue
    else:
        print("Warning: scoring failed for requirement '%s'; using L=0" % req["text"], file=sys.stderr)
    return {"L": round(llm_score, 1), "E": round(e_score, 1),
            "S": round(combine(llm_score, e_score), 1), "explanation": why}
