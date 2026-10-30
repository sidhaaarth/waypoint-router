"""LOOKS UP: one Tavily search for product facts that the store's FAQ doesn't hold (todo.md T2, DECISIONS.md D32).

Rule (needs_lookup): product questions (subflow boots/shirt/jeans/jacket) always look up; storewide questions
(pricing/membership/timing/policy) look up only when the store FAQ gave no answer. Everything else never does:
store policy is the truth there, not the web (D32).
Search: POST https://api.tavily.com/search, basic depth (1 credit), max 4 results, restricted to the brand's official
sites when the brand is known. Never raises. Traced to LangSmith as a "tool" run under tracing.PARENT.
"""
import os, re, time, uuid

import httpx

from app import config  # noqa: F401  (loads .env)
from app import tracing

URL = "https://api.tavily.com/search"
TIMEOUT = 6.0
PRODUCTS = {"boots", "shirt", "jeans", "jacket"}
STOREWIDE = {"pricing", "membership", "timing", "policy"}
BRAND_DOMAINS: dict[str, list[str]] = {
    "tommy_hilfiger": ["usa.tommy.com"],  # bare tommy.com also matches sg./pt./fi. regional stores,
    "calvin_klein": ["calvinklein.us", "calvinklein.com"],
    "guess": ["guess.com"],
    "michael_kors": ["michaelkors.com"],
}


def needs_lookup(subflow: str, store_answer: str | None, text: str) -> bool:
    if subflow in PRODUCTS:
        return True
    # ponytail: "doesn't address" = empty answer only; add an overlap/LLM check if storewide misses show up in evals
    return subflow in STOREWIDE and not (store_answer or "").strip() and bool(text.strip())


def trusted_domains(subflow: str, store_answer: str | None, text: str, support: dict) -> list[str]:
    """Outside facts only from the store's trusted domains (rules["support"], fix 10): the request type must allow a
    lookup, our own FAQ must have had no answer, and a trusted name (e.g. a brand) must appear in the question.
    [] = no lookup. Never the open web."""
    if subflow not in (support.get("lookup_subflows") or []) or (store_answer or "").strip() or not text.strip():
        return []
    t = " " + re.sub(r"[\W_]+", " ", text.lower()) + " "
    return next((d for name, d in (support.get("trusted_domains") or {}).items() if f" {name.lower()} " in t), [])


def _brand(*texts) -> str | None:
    s = " ".join(t for t in texts if t).lower().replace("_", " ").replace("-", " ")
    return next((b for b in BRAND_DOMAINS if b.replace("_", " ") in s), None)


async def lookup(question: str, product: str | None, brand: str | None, key: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, domains: list[str] | None = None) -> dict:
    """domains: the trusted sites to search (from trusted_domains); None = infer from BRAND_DOMAINS (legacy)."""
    brand = None if domains else _brand(brand) or _brand(product, question)
    domains = domains or BRAND_DOMAINS.get(brand, [])
    prefix = " ".join(w for w in [(brand or "").replace("_", " "), product or ""] if w)
    query = re.sub(r"\s+", " ", f"{prefix} {question}").strip()[:380]  # Tavily caps queries at 400 chars
    out = {"query": query, "domains": domains, "answer": None, "sources": [], "credits": 0, "ok": False}
    key = key or os.environ.get("TAVILY_API_KEY")
    started = time.time()
    if not key:
        out["error"] = "no TAVILY_API_KEY"
    else:
        body = {"query": query, "search_depth": "basic", "max_results": 4, "include_answer": "basic",
                "include_usage": True, **({"include_domains": domains} if domains else {})}
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT, transport=transport) as c:
                r = await c.post(URL, json=body, headers={"Authorization": f"Bearer {key}"})
            data = r.json() if r.content else {}
            if r.status_code != 200:
                out["error"] = f"tavily {r.status_code}: {str(data)[:200]}"
            else:
                out.update(ok=True, answer=data.get("answer") or None,
                           credits=(data.get("usage") or {}).get("credits", 1),
                           sources=[{"title": x.get("title", ""), "url": x.get("url", ""),
                                     "snippet": (x.get("content") or "")[:400]} for x in data.get("results") or []])
        except (httpx.HTTPError, ValueError) as e:
            out["error"] = f"{type(e).__name__}: {e}"
    _trace(out, started)
    return out


def _trace(out, started):
    share = tracing.SHARE_TEXT.get()
    run = {"id": str(uuid.uuid4()), "name": "tavily_search", "run_type": "tool", "session_name": tracing.PROJECT,
           "start_time": tracing._iso(started), "end_time": tracing._iso(time.time()),
           "inputs": {"domains": out["domains"], **({"query": out["query"]} if share else {})},
           "outputs": {"ok": out["ok"], "credits": out["credits"], "n_sources": len(out["sources"]),
                       **({"answer": out["answer"], "sources": out["sources"]} if share else {})}}
    if out.get("error"):
        run["error"] = out["error"]
    if tracing.PARENT.get():
        run["parent_run_id"] = tracing.PARENT.get()
    tracing._post(run)


def as_brief(result: dict) -> str:
    """Short block for the reply writer. Empty when the lookup failed or found nothing."""
    if not result.get("ok") or not (result.get("answer") or result.get("sources")):
        return ""
    lines = ["WEB LOOKUP (from the brand's website, not store policy; don't promise anything based on it):"]
    # Tavily's answer often just says the pages don't cover it; that's not a fact, so drop it
    if result.get("answer") and not re.search(r"(do|does) not (contain|provide|mention|address|include)", result["answer"]):
        lines.append(f"Summary: {result['answer']}")
    lines += [f"- {s['snippet'][:250]} (source: {s['url']})" for s in result["sources"][:3]]
    return "\n".join(lines)
