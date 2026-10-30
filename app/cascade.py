"""Gap 1: confidence cascade. CONTRACT (main.py calls exactly these):

route: dict with keys policy ("cascade"|"direct"), tier ("nano"|"super"|"ultra", used when direct),
       threshold (float 0..1), first_n (int), escalate_to ("super"|"ultra", used when cascade)

def confidence(lp_content: list[dict] | None, first_n: int) -> float
    Geometric-mean token probability over the first `first_n` answer tokens = exp(mean(logprob)).
    None/empty -> 0.0 (no evidence => escalate).

async def complete(key: str, body: dict, route: dict, priority="interactive") -> tuple[dict, dict]
    Non-streaming. Returns (openai_response_json, meta).
async def complete_stream(key: str, body: dict, route: dict, priority="interactive") -> tuple[AsyncIterator[bytes], dict]
    Returns (sse_byte_generator, meta). meta is filled in while the generator runs; final once it ends.

meta = {"tier": str, "model": str, "confidence": float|None, "escalated": bool, "fallback": bool,
        "status": int, "attempts": [{"tier", "model", "usage": dict|None, "confidence": float|None}]}
"""
import json, math

from . import config, limits


def confidence(lp_content, first_n):
    lps = [t["logprob"] for t in (lp_content or [])[:first_n] if t.get("logprob") is not None]
    return math.exp(sum(lps) / len(lps)) if lps else 0.0


DONE = b"data: [DONE]\n\n"


def _sse(obj):
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _route(route):
    r = route or {}
    return (r.get("policy") or "cascade", r.get("tier") or "super",
            config.DEFAULT_THRESHOLD if r.get("threshold") is None else r["threshold"],
            r.get("first_n") or config.DEFAULT_FIRST_N, r.get("escalate_to") or "super")


def _nano_body(body):
    """Nano scoring call: thinking off (else logprobs score hidden reasoning), logprobs forced."""
    b = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
    b["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), **config.NO_THINK["chat_template_kwargs"]}
    return {**b, "model": config.TIERS["nano"], "logprobs": True}


def _direct_body(body, tier):
    """Direct route. Nano runs thinking-off, matching how calibration measured it (and cheaper)."""
    b = {**body, "model": config.TIERS[tier]}
    if tier == "nano":
        b["chat_template_kwargs"] = {**(body.get("chat_template_kwargs") or {}), **config.NO_THINK["chat_template_kwargs"]}
    return b


def _with_usage(body):
    return {**body, "stream": True, "stream_options": {**(body.get("stream_options") or {}), "include_usage": True}}


def _strip_lp(data):
    if isinstance(data, dict) and data.get("choices"):
        return {**data, "choices": [{**c, "logprobs": None} for c in data["choices"]]}
    return data


def _meta():
    return {"tier": None, "model": None, "confidence": None, "escalated": False, "fallback": False,
            "status": 200, "attempts": []}


def _attempt(meta, tier, model, usage=None, c=None):
    a = {"tier": tier, "model": model, "usage": usage, "confidence": c}
    meta["attempts"].append(a)
    return a


async def _call(key, body, tier, meta, priority):
    status, data, _, model, fb = await limits.call(key, body, priority)
    ok = status == 200 and isinstance(data, dict)
    _attempt(meta, tier, model, data.get("usage") if ok else None)
    meta.update(tier=tier, model=model, fallback=fb, status=status)
    return status, data if isinstance(data, dict) else {"error": {"message": str(data), "code": status}}


async def complete(key, body, route, priority="interactive"):
    policy, tier, threshold, first_n, esc = _route(route)
    body = {k: v for k, v in body.items() if k not in ("stream", "stream_options")}
    meta = _meta()
    if policy == "direct":
        _, data = await _call(key, _direct_body(body, tier), tier, meta, priority)
        return data, meta

    status, data = await _call(key, _nano_body(body), "nano", meta, priority)
    if status == 200:
        # A tool_calls answer has no content logprobs -> confidence 0 -> always escalates.
        # ponytail: scores the first N tokens only, like streaming; scoring the whole answer is an option here.
        c = confidence((data["choices"][0].get("logprobs") or {}).get("content"), first_n)
        meta["confidence"] = meta["attempts"][0]["confidence"] = c
        if c >= threshold:
            return (data if body.get("logprobs") else _strip_lp(data)), meta
    meta["escalated"] = True
    _, data = await _call(key, {**body, "model": config.TIERS[esc]}, esc, meta, priority)
    return (data if body.get("logprobs") else _strip_lp(data)), meta


async def complete_stream(key, body, route, priority="interactive"):
    policy, tier, threshold, first_n, esc = _route(route)
    meta = _meta()
    want_lp = bool(body.get("logprobs"))
    want_usage = bool((body.get("stream_options") or {}).get("include_usage"))

    def out(ch):
        if not ch.get("choices"):  # usage-only chunk: we always request it, the caller may not have
            return _sse(ch) if want_usage else None
        return _sse(ch if want_lp else _strip_lp(ch))

    async def pipe(tier, b):
        async with limits.call_stream(key, b, priority) as (_, chunks, model, fb):
            a = _attempt(meta, tier, model)
            meta.update(tier=tier, model=model, fallback=fb)
            async for ch in chunks:
                a["usage"] = ch.get("usage") or a["usage"]
                if o := out(ch):
                    yield o

    async def gen():
        try:
            if policy == "direct":
                async for o in pipe(tier, _with_usage(_direct_body(body, tier))):
                    yield o
            else:
                sent, nano, n_seen = False, None, 0
                try:
                    async with limits.call_stream(key, _with_usage(_nano_body(body)), priority) as (_, chunks, model, fb):
                        nano = _attempt(meta, "nano", model)
                        it, buf, lps = aiter(chunks), [], []
                        async for ch in it:
                            buf.append(ch)
                            nano["usage"] = ch.get("usage") or nano["usage"]
                            for c in ch.get("choices") or []:
                                lps += (c.get("logprobs") or {}).get("content") or []
                            if len(lps) >= first_n:
                                break
                        n_seen = len(lps)
                        nano["confidence"] = meta["confidence"] = c = confidence(lps, first_n)
                        if c >= threshold:
                            # ponytail: confidence dropping after the first N tokens isn't caught (PLAN §5.4).
                            meta.update(tier="nano", model=model, fallback=fb)
                            sent = True
                            for ch in buf:
                                if o := out(ch):
                                    yield o
                            async for ch in it:
                                nano["usage"] = ch.get("usage") or nano["usage"]
                                if o := out(ch):
                                    yield o
                except Exception:
                    if sent:
                        raise
                    nano = nano or _attempt(meta, "nano", config.TIERS["nano"])
                if not sent:  # low confidence, or Nano failed before we sent anything: stop Nano, stream escalation live
                    meta["escalated"] = True
                    async for o in pipe(esc, _with_usage({**body, "model": config.TIERS[esc]})):
                        yield o
                    if nano and not nano["usage"]:
                        # ponytail: abandoned Nano stream has no usage chunk; estimate from the escalation's prompt.
                        p = (meta["attempts"][-1]["usage"] or {}).get("prompt_tokens", 0)
                        nano["usage"] = {"prompt_tokens": p, "completion_tokens": n_seen, "total_tokens": p + n_seen}
        except Exception as e:
            meta["status"] = getattr(e, "status", 502)
            yield _sse({"error": {"message": str(e), "code": meta["status"]}})
        yield DONE

    return gen(), meta
