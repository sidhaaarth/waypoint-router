"""Gap 5: prompt-layout lint. CONTRACT:

def lint(system_prompt: str | None, head_chars: int = 2000) -> list[dict]
    Find dynamic content in the cache-relevant head of a system prompt (dates, times, ISO timestamps,
    UUIDs/long ids, emails, template vars like {name} {{x}} $VAR <var>, "today", weekday names).
    Each warning: {"kind": str, "match": str, "pos": int, "message": str}. Static prompt -> [].
def cache_stats(project_id: str) -> dict
    Per route over Super/Ultra requests: {route: {"requests", "prompt_tokens", "cached_tokens", "hit_rate"}}.
"""
import re

from . import config, db

_MON = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?"
# Most specific first; a later match overlapping an earlier one is dropped.
PATTERNS = [
    ("timestamp", r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?"),
    ("uuid", r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    ("email", r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ("date", r"\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b|\b\d{1,2}\s+" + _MON + r",?\s+\d{4}\b"
             r"|\b" + _MON + r"\s+\d{1,2}(?:st|nd|rd|th)?\b(?:,?\s+\d{4})?"),
    ("time", r"\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?\b|\b(?:1[0-2]|0?[1-9])(?::[0-5]\d)?\s?[ap]\.?m\b\.?"),
    ("id", r"\b(?=[0-9a-fA-F]*\d)[0-9a-fA-F]{8,}\b"),
    ("template", r"\{\{[^{}]*\}\}|\$\{[^{}]*\}|\{[A-Za-z_]\w*\}|\$[A-Z_][A-Z0-9_]*\b|<[A-Za-z_][\w-]*>"
                 r"|\[[A-Z][A-Z0-9_]+\]"),
    ("relative", r"\b(?:today|tonight|tomorrow|yesterday|right now|current (?:date|time)"  # bare "now" too noisy
                 r"|(?:mon|tues|wednes|thurs|fri|satur|sun)day)\b"),
]
_RX = [(k, re.compile(p, re.I if k in ("date", "time", "relative") else 0)) for k, p in PATTERNS]
MSG = ("Dynamic {kind} {match!r} at char {pos} is in the cache-relevant head of the system prompt. "
       "Move dynamic content after the static policy text so Super/Ultra can reuse the cached prefix.")
CACHED_MODELS = sorted({config.TIERS["super"], config.TIERS["ultra"], *config.SIBLINGS[config.TIERS["super"]]})


def lint(system_prompt, head_chars=2000):
    head = (system_prompt or "")[:head_chars]
    taken, out = [], []
    for kind, rx in _RX:
        for m in rx.finditer(head):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            taken.append(m.span())
            out.append({"kind": kind, "match": m.group(), "pos": m.start(),
                        "message": MSG.format(kind=kind, match=m.group(), pos=m.start())})
    return sorted(out, key=lambda w: w["pos"])


def cache_stats(project_id):
    marks = ",".join("?" * len(CACHED_MODELS))
    rows = db.q("SELECT route, COUNT(*) AS n, COALESCE(SUM(prompt_tokens), 0) AS p, "
                f"COALESCE(SUM(cached_tokens), 0) AS c FROM requests WHERE project_id=? AND model IN ({marks}) "
                "GROUP BY route", (project_id, *CACHED_MODELS))
    return {r["route"]: {"requests": r["n"], "prompt_tokens": r["p"], "cached_tokens": r["c"],
                         "hit_rate": r["c"] / r["p"] if r["p"] else 0.0} for r in rows}
