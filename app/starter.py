"""Starter pack: routing hints for a business type from public help-centre pages (Tavily) + Nemotron Super.

Tavily: 2 basic searches (1 credit each) + 1 basic extract of the top 5 pages (1 credit per 5 URLs) = ~3 credits.
Super (thinking off, json_schema, via limits.call) proposes request types, example customer messages, topics for a
person, and a "what we help with" paragraph. Routing hints only: no policy facts, prices, windows or answers from
other companies. Public pages, used internally, never republished. A person accepts or discards the pack.
"""
import json, os

import httpx

from . import config, limits

SEARCH, EXTRACT = "https://api.tavily.com/search", "https://api.tavily.com/extract"
QUERIES = ("{b} customer support help centre FAQ", "{b} common customer service questions contact us")
PAGES, PAGE_CHARS = 5, 3000

SCHEMA = {"type": "json_schema", "json_schema": {"name": "starter_pack", "strict": True, "schema": {
    "type": "object", "additionalProperties": False, "required": ["request_types", "human_topics", "scope"],
    "properties": {
        "request_types": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["name", "description", "examples"],
            "properties": {"name": {"type": "string"}, "description": {"type": "string"},
                           "examples": {"type": "array", "items": {"type": "string"}}}}},
        "human_topics": {"type": "array", "items": {"type": "string"}},
        "scope": {"type": "string"}}}}}

PROMPT = """You design the request types for a customer-support router in this business: {b}.
From the help-centre pages below, list what customers typically ask for. Return:
- request_types: 10 to 25 types. name: short snake_case. description: one line. examples: 5 to 10 realistic customer
  messages each, paraphrased in your own words.
- human_topics: topics that must go to a person (safety, legal, complaints, vulnerable customers, fraud, ...).
- scope: one paragraph "We help with ..." describing what this kind of business's support covers.
STRICT: routing hints only. No company or brand names, no personal data, no prices, fees, time windows, limits,
eligibility rules or answers. Never copy text from the pages."""


async def tavily(url, body, key, transport=None):
    async with httpx.AsyncClient(timeout=30, transport=transport) as c:
        r = await c.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
    if r.status_code != 200:
        raise RuntimeError(f"tavily {r.status_code}: {r.text[:200]}")
    return r.json()


async def build(tf_key, business, tavily_key=None, transport=None):
    """-> starter pack dict (request_types, human_topics, scope, sources, credits, usage). Raises on failure."""
    key = tavily_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise RuntimeError("no TAVILY_API_KEY on the server")
    hits, credits = {}, 0
    for q in QUERIES:
        d = await tavily(SEARCH, {"query": q.format(b=business), "search_depth": "basic", "max_results": 5,
                                  "include_usage": True}, key, transport)
        credits += (d.get("usage") or {}).get("credits", 1)
        for x in d.get("results") or []:
            hits.setdefault(x.get("url"), x)
    top = sorted(hits.values(), key=lambda x: -(x.get("score") or 0))[:PAGES]
    pages = {x["url"]: x.get("content") or "" for x in top}
    if top:
        d = await tavily(EXTRACT, {"urls": list(pages), "extract_depth": "basic", "include_usage": True}, key, transport)
        credits += (d.get("usage") or {}).get("credits", 1)
        for x in d.get("results") or []:
            pages[x.get("url")] = x.get("raw_content") or pages.get(x.get("url"), "")
    text = "\n\n".join(f"--- page {i}\n{t[:PAGE_CHARS]}" for i, t in enumerate(pages.values(), 1))
    body = {"model": config.TIERS["super"], "temperature": 0.2, "max_tokens": 8000, **config.NO_THINK,
            "response_format": SCHEMA,
            "messages": [{"role": "system", "content": PROMPT.format(b=business)},
                         {"role": "user", "content": text or "(no pages found: use general knowledge of this business)"}]}
    status, data, *_ = await limits.call(tf_key, body, priority="background")
    if status != 200:
        raise RuntimeError(f"Super failed: {status} {str(data)[:200]}")
    pack = json.loads(data["choices"][0]["message"]["content"])
    pack["request_types"] = pack["request_types"][:25]
    for t in pack["request_types"]:
        t["examples"] = t["examples"][:10]
    return {**pack, "business_type": business, "credits": credits, "usage": data.get("usage"),
            "sources": [{"url": x["url"], "title": x.get("title", "")} for x in top],
            "note": "Routing hints only: shapes how requests are understood; never used as your answers or rules."}
