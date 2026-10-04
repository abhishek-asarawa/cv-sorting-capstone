"""Deterministic aggregation, ranking and output formatting."""
import csv
from typing import List

HEADER = ["section", "rank", "candidate", "file", "requirement_text", "requirement_type",
          "w_i", "L_i", "E_i", "S_i", "overall_score", "explanation_or_reason"]


def weights(types: List[str]) -> List[int]:
    """Weight per requirement: must-have=3, nice-to-have=1; all 1 if the JD has no must-have."""
    if "must-have" not in types:
        return [1 for _ in types]
    return [3 if t == "must-have" else 1 for t in types]


def overall_score(rows: List[dict]) -> float:
    """Weighted mean of S_i; raises ValueError if there are no requirements."""
    if not rows:
        raise ValueError("JD has no requirements")
    w = weights([r["type"] for r in rows])
    return round(sum(wi * r["S"] for wi, r in zip(w, rows)) / sum(w), 1)


def rank(candidates: List[dict]) -> List[dict]:
    """Return candidates (copied) with per-row `w`, `overall` and 1-based `rank`, best first."""
    out = []
    for c in candidates:
        w = weights([r["type"] for r in c["rows"]])
        rows = [dict(r, w=wi) for wi, r in zip(w, c["rows"])]
        out.append(dict(c, rows=rows, overall=overall_score(rows)))
    out.sort(key=lambda c: (-c["overall"], c["name"]))
    for i, c in enumerate(out, 1):
        c["rank"] = i
    return out


def reasons(cand: dict, n: int = 3) -> List[str]:
    """Short lines: the strongest two requirement matches plus the weakest one."""
    rows = sorted(cand["rows"], key=lambda r: -r["S"])
    picked = [("+", r) for r in rows[:max(0, n - 1)]]
    if len(rows) >= n:
        picked.append(("-", rows[-1]))
    return ["%s %s (%.0f): %s" % (m, r["text"], r["S"], r["explanation"]) for m, r in picked]


def format_console(ranked: List[dict], skipped: List[dict]) -> str:
    """Render the ranked table and skipped-file list as text."""
    lines = []
    if ranked:
        lines.append("%-5s %-28s %s" % ("Rank", "Candidate", "Score"))
        for c in ranked:
            lines.append("%-5d %-28s %.1f" % (c["rank"], c["name"][:28], c["overall"]))
            lines.extend("        " + s for s in reasons(c))
    else:
        lines.append("No candidates could be scored.")
    if skipped:
        lines.append("")
        lines.append("Skipped files:")
        lines.extend("  %s - %s" % (s["file"], s["reason"]) for s in skipped)
    return "\n".join(lines)


def write_csv(path: str, ranked: List[dict], skipped: List[dict]) -> None:
    """Write requirement rows, per-candidate summary rows and skipped-file rows to a CSV."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(HEADER)
        for c in ranked:
            for r in c["rows"]:
                wr.writerow(["requirement", c["rank"], c["name"], c["file"], r["text"], r["type"],
                             r["w"], r["L"], r["E"], r["S"], "", r["explanation"]])
            wr.writerow(["summary", c["rank"], c["name"], c["file"], "", "", "", "", "", "", c["overall"], ""])
        for s in skipped:
            wr.writerow(["skipped", "", "", s["file"], "", "", "", "", "", "", "", s["reason"]])
