"""Settings and the model map. Model IDs and prices were verified against the live Token Factory catalog (PLAN.md §8a)."""
import os

if os.path.exists(".env"):
    for line in open(".env"):
        k, sep, v = line.strip().partition("=")
        if sep and not k.startswith("#"):
            os.environ.setdefault(k.strip(), v.strip().strip('"\''))

TF_BASE = os.environ.get("TF_BASE", "https://api.tokenfactory.nebius.com/v1")
DB_PATH = os.environ.get("DB_PATH", "router.db")  # point at a mounted volume in prod (/data/router.db)
ROUTER_SECRET = os.environ.get("ROUTER_SECRET")  # Fernet key; generated next to the DB if unset
PORT = int(os.environ.get("PORT", "8888"))
SELF_HOSTED = os.environ.get("SELF_HOSTED") == "1"  # downloaded copy: no marketing site or showcase (Dockerfile sets it)

TIERS = {
    "nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "super": "nvidia/nemotron-3-super-120b-a12b",
    "ultra": "nvidia/Nemotron-3-Ultra-550b-a55b",
}
ORDER = ["nano", "super", "ultra"]
MODEL_TIER = {m: t for t, m in TIERS.items()}

# Fallback on 429/529/5xx. No -fast variants exist, so siblings are different-but-similar models.
SIBLINGS = {
    TIERS["nano"]: ["nvidia/Nemotron-3_5-Lightning"],
    TIERS["super"]: ["nvidia/Nemotron-3.5-Super-VL"],
    TIERS["ultra"]: [TIERS["super"]],
}

# Nemotron 3 thinks by default; logprobs then score the reasoning, not the answer. Verified fix:
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}

# Cascade defaults for a route that hasn't been calibrated yet (conservative: escalate often).
DEFAULT_THRESHOLD = 0.90
DEFAULT_FIRST_N = 24
