"""Gap 2: rate-limit pacing + fallback. CONTRACT (main.py and cascade.py call exactly these):

async def call(key: str, body: dict, priority: str = "interactive") -> tuple[int, dict|str, dict, str, bool]
    Non-streaming chat through tf.chat. Returns (status, data, headers, model_used, fell_back).
    - await pacing gate for this key before sending (background waits when near the limit; interactive only waits on hard 0-remaining)
    - observe() the response headers
    - on 429/529/5xx: honor Retry-After (cap ~5s for interactive), retry once same model, then walk fallback_chain(model)
    - 4xx other than 429 -> return as-is, no retry

@asynccontextmanager
async def call_stream(key: str, body: dict, priority: str = "interactive")
    -> yields (headers: dict, chunks: AsyncIterator[dict], model_used: str, fell_back: bool)
    Same pacing/fallback, but fallback only applies to errors BEFORE the first chunk (tf.stream raises tf.TFError).
    If everything fails, raise the last tf.TFError.

def fallback_chain(model: str) -> list[str]     # [model, *config.SIBLINGS.get(model, [])]
def observe(key: str, status: int, headers: dict) -> None
def state(key: str) -> dict                    # for dashboard: remaining, limits, dynamic usage %, scale, last_429_ts, counts
"""
import hashlib, re, time
from asyncio import sleep
from contextlib import asynccontextmanager

import httpx

from . import config, tf

INTERACTIVE_MAX_WAIT = 5.0   # s; cap on any pause (gate or Retry-After) for customer-facing calls
BACKGROUND_MAX_WAIT = 30.0   # s; same cap for background work
BG_MIN_REMAINING = 0.20      # background pauses when remaining-requests < 20% of the limit
BG_MAX_USAGE_PCT = 85.0      # ...or dynamic-period usage above this (keeps us in the 50-85% band)
BG_PACE = 1.0                # s; minimum background pause when the gate trips
DEFAULT_RETRY_AFTER = 0.5    # s; backoff before the same-model retry when no Retry-After came back

_states = {}  # ponytail: in-memory, per-process; move to Redis if the router runs >1 replica


def _st(key):
    h = hashlib.sha256(key.encode()).hexdigest()
    if h not in _states:
        _states[h] = {"limits": {}, "remaining": {}, "reset_s": {}, "dynamic": {}, "over_limit": False,
                      "retry_after_s": None, "seen_at": None, "last_429_ts": None,
                      "counts": {"requests": 0, "retries": 0, "fallbacks": 0, "rate_limited": 0, "errors": 0}}
    return _states[h]


def _secs(v):
    """'1s' / '500ms' / '1m30s' / '2' (Retry-After) -> seconds, or None."""
    if v is None:
        return None
    v = v.strip()
    try:
        return float(v)
    except ValueError:
        parts = re.findall(r"([\d.]+)(ms|h|m|s)", v)
        return sum(float(n) * {"ms": .001, "s": 1, "m": 60, "h": 3600}[u] for n, u in parts) if parts else None


def _num(v):
    try:
        return float(v.strip().rstrip("%s")) if v is not None else None
    except ValueError:
        return None


def fallback_chain(model):
    return [model, *config.SIBLINGS.get(model, [])]


def observe(key, status, headers):
    st, h = _st(key), {k.lower(): v for k, v in (headers or {}).items()}
    p = "x-ratelimit-"
    for kind in ("requests", "tokens"):
        for field, name, conv in (("limits", "limit", _num), ("remaining", "remaining", _num), ("reset_s", "reset", _secs)):
            val = conv(h.get(f"{p}{name}-{kind}"))
            if val is not None:
                st[field][kind] = int(val) if field != "reset_s" else val
        for name, field in (("scale", "scale_"), ("period-usage", "usage_pct_")):
            val = _num(h.get(f"{p}dynamic-{name}-{kind}"))
            if val is not None:
                st["dynamic"][field + kind] = val
    if (val := _secs(h.get(f"{p}dynamic-period-remaining"))) is not None:
        st["dynamic"]["period_remaining_s"] = val
    if any(k.startswith(p) for k in h):
        st["over_limit"] = h.get(f"{p}over-limit", "").lower() == "yes"
        st["seen_at"] = time.monotonic()
    st["retry_after_s"] = _secs(h.get("retry-after"))
    if status in (429, 529):
        st["counts"]["rate_limited"] += 1
        st["last_429_ts"] = time.time()
    elif status >= 400:
        st["counts"]["errors"] += 1


def state(key):
    st = _st(key)
    return {k: v for k, v in st.items() if k != "seen_at"}


def _cap(priority):
    return INTERACTIVE_MAX_WAIT if priority == "interactive" else BACKGROUND_MAX_WAIT


def _gate_wait(st, priority):
    """Seconds to pause before sending, from the last-seen headers (stale once their reset window passes)."""
    if st["seen_at"] is None:
        return 0.0
    age = time.monotonic() - st["seen_at"]
    rem, lim, reset = st["remaining"], st["limits"], st["reset_s"]
    left = {k: max(0.0, reset.get(k, 1.0) - age) for k in ("requests", "tokens")}
    exhausted = [k for k in left if rem.get(k) == 0 and left[k] > 0]
    if exhausted:
        return min(max(left[k] for k in exhausted), _cap(priority))
    if priority == "interactive":
        return 0.0
    low = lim.get("requests") and rem.get("requests") is not None and rem["requests"] < BG_MIN_REMAINING * lim["requests"] and left["requests"] > 0
    usage = max(st["dynamic"].get("usage_pct_requests", 0), st["dynamic"].get("usage_pct_tokens", 0))
    if low or st["over_limit"] or usage > BG_MAX_USAGE_PCT:
        return min(max(left["requests"], BG_PACE), _cap(priority))
    return 0.0


def _attempts(model):
    """(model, is_same_model_retry): first model twice, then each sibling once."""
    chain = fallback_chain(model)
    return [(chain[0], False), (chain[0], True)] + [(m, False) for m in chain[1:]]


def _retryable(status):
    return status == 429 or status >= 500


async def _before(st, m, retry, origin, priority):
    if retry:
        st["counts"]["retries"] += 1
        await sleep(min(st["retry_after_s"] or DEFAULT_RETRY_AFTER, _cap(priority)))
    elif m != origin:
        st["counts"]["fallbacks"] += 1
    if (w := _gate_wait(st, priority)) > 0:
        await sleep(w)
    st["counts"]["requests"] += 1


async def call(key, body, priority="interactive"):
    st, origin = _st(key), body["model"]
    for m, retry in _attempts(origin):
        await _before(st, m, retry, origin, priority)
        b = body if m == origin else {**body, "model": m}
        try:
            status, data, headers = await tf.chat(key, b)
        except httpx.TransportError as e:
            status, data, headers = 502, str(e), {}
        observe(key, status, headers)
        if not _retryable(status):
            break
    return status, data, headers, m, m != origin


@asynccontextmanager
async def call_stream(key, body, priority="interactive"):
    st, origin, last, yielded = _st(key), body["model"], None, False
    for m, retry in _attempts(origin):
        await _before(st, m, retry, origin, priority)
        b = body if m == origin else {**body, "model": m}
        try:
            async with tf.stream(key, b) as (headers, chunks):
                observe(key, 200, headers)
                yielded = True
                yield headers, chunks, m, m != origin
                return
        except httpx.TransportError as e:
            if yielded:
                raise
            last = tf.TFError(502, str(e), {})
        except tf.TFError as e:
            if yielded:
                raise
            last = e
        observe(key, last.status, last.headers)
        if not _retryable(last.status):
            break
    raise last
