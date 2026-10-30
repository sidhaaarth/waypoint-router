"""Thin async client for Token Factory. No retries here: limits.py owns pacing, retry and fallback.
Every call is traced to LangSmith (tracing.llm_run); that's a no-op without LANGSMITH_API_KEY."""
import json, time
from contextlib import asynccontextmanager

import httpx

from . import config, tracing

client = httpx.AsyncClient(base_url=config.TF_BASE, timeout=httpx.Timeout(300, connect=10),
                           limits=httpx.Limits(max_connections=200))


class TFError(Exception):
    def __init__(self, status, body, headers):
        super().__init__(f"Token Factory {status}: {str(body)[:200]}")
        self.status, self.body, self.headers = status, body, headers


def _h(key):
    return {"Authorization": f"Bearer {key}"}


def _body(r):
    try:
        return r.json()
    except ValueError:
        return r.text[:1000]


async def chat(key, body):
    """POST /chat/completions (non-streaming). Returns (status, json_or_text, headers)."""
    t = time.time()
    try:
        r = await client.post("/chat/completions", json=body, headers=_h(key))
    except httpx.TransportError as e:
        tracing.llm_run(body, 0, repr(e), t, time.time())
        raise
    data = _body(r)
    tracing.llm_run(body, r.status_code, data, t, time.time())
    return r.status_code, data, dict(r.headers)


@asynccontextmanager
async def stream(key, body):
    """Streaming chat. Yields (headers, async iterator of parsed chunk dicts). Raises TFError on non-200.
    Traced once the stream closes, with the accumulated text and the final usage chunk."""
    t, acc = time.time(), {"text": [], "usage": None}
    async with client.stream("POST", "/chat/completions", json={**body, "stream": True}, headers=_h(key)) as r:
        if r.status_code != 200:
            raw = (await r.aread()).decode(errors="replace")
            tracing.llm_run(body, r.status_code, raw[:1000], t, time.time(), stream=True)
            raise TFError(r.status_code, raw[:1000], dict(r.headers))

        async def chunks():
            async for line in r.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:].strip()
                    if data == "[DONE]":
                        return
                    c = json.loads(data)
                    acc["usage"] = c.get("usage") or acc["usage"]
                    for ch in c.get("choices") or []:
                        acc["text"].append((ch.get("delta") or {}).get("content") or "")
                    yield c

        try:
            yield dict(r.headers), chunks()
        finally:
            tracing.llm_run(body, 200, {"usage": acc["usage"], "choices": [{"message": {"content": "".join(acc["text"])}}]},
                            t, time.time(), stream=True)


async def models(key):
    """GET /models?verbose=true. Returns (status, json_or_text)."""
    r = await client.get("/models", params={"verbose": "true"}, headers=_h(key))
    return r.status_code, _body(r)
