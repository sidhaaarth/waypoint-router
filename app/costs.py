"""Gap 6: cost per request and per customer. CONTRACT:

PRICES: dict[model_id -> {"prompt": $/token, "completion": $/token, "cache_read": $/token | None}]
    Seeded with DEFAULT_PRICES (from probe, PLAN.md §8a); load() overwrites from the live catalog.
def load(models_payload: dict) -> None          # payload of GET /models?verbose=true
def cost(model: str, usage: dict | None) -> float
    prompt_tokens split into cached (usage.prompt_tokens_details.cached_tokens) at cache_read price
    (falls back to prompt price if None) and uncached at prompt price; + completion_tokens * completion.
def ultra_equivalent(usage: dict | None) -> float   # what this usage would cost on Ultra, uncached
def spend(project_id: str, user: str) -> float      # SUM(cost_usd) from requests
def over_budget(project_id: str, user: str | None, frac: float = 0.9) -> bool   # vs customers.budget_usd
"""
from . import config, db

_NANO = {"prompt": 6e-08, "completion": 2.4e-07, "cache_read": None}
_SUPER = {"prompt": 3e-07, "completion": 9e-07, "cache_read": 6e-08}
DEFAULT_PRICES = {  # $/token, live catalog GET /models?verbose=true, 2026-10-27
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": _NANO,
    "nvidia/Nemotron-3_5-Lightning": _NANO,
    "nvidia/nemotron-3-super-120b-a12b": _SUPER,
    "nvidia/Nemotron-3.5-Super-VL": _SUPER,
    "nvidia/Nemotron-3-Ultra-550b-a55b": {"prompt": 1e-06, "completion": 3e-06, "cache_read": 1.2e-07},
}
PRICES = {m: dict(p) for m, p in DEFAULT_PRICES.items()}


def _f(v):
    return float(v) if v not in (None, "") else None


def load(models_payload):
    for m in (models_payload or {}).get("data") or []:
        p = m.get("pricing")
        if m.get("id") and p:
            PRICES[m["id"]] = {"prompt": _f(p.get("prompt")) or 0.0, "completion": _f(p.get("completion")) or 0.0,
                               "cache_read": _f(p.get("input_cache_read"))}


def _cost(p, usage, cache=True):
    u = usage or {}
    prompt = u.get("prompt_tokens") or 0
    cached = ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0) if cache else 0
    cached = min(cached, prompt)
    cache_price = p["prompt"] if p["cache_read"] is None else p["cache_read"]
    return cached * cache_price + (prompt - cached) * p["prompt"] + (u.get("completion_tokens") or 0) * p["completion"]


def cost(model, usage):
    p = PRICES.get(model)
    return _cost(p, usage) if p else 0.0


def ultra_equivalent(usage):
    return _cost(PRICES[config.TIERS["ultra"]], usage, cache=False)


def spend(project_id, user):
    return db.one("SELECT COALESCE(SUM(cost_usd), 0) AS s FROM requests WHERE project_id=? AND user=?",
                  (project_id, user))["s"]


def over_budget(project_id, user, frac=0.9):
    if user is None:
        return False
    row = db.one("SELECT budget_usd FROM customers WHERE project_id=? AND user=?", (project_id, user))
    if not row or row["budget_usd"] is None:
        return False
    return spend(project_id, user) >= frac * row["budget_usd"]
