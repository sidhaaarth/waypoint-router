"""Setup + calibration (C15–C18). CONTRACT:

def check(output: str, expected: str | None, kind: str) -> bool | None
    kind in {"exact", "contains", "json", "regex", "judge"}; returns None when it can't decide (-> use judge).
async def judge(key: str, task: str, input_text: str, expected: str | None, a: str, b: str, context: str | None = None) -> float
    Blind pairwise with Ultra, run twice with order swapped. Returns P(a at least as good as b) in {0, .5, 1}.
def choose_threshold(rows: list[dict], tolerance=0.02, min_coverage=0.2) -> dict
    rows: [{"c": nano_confidence, "nano_ok": bool, "ultra_ok": bool}]
    -> {"policy": "cascade"|"direct", "tier": str, "threshold": float|None, "coverage": float,
        "nano_acc_above": float, "ultra_acc": float, "n": int}
async def generate_prompt(key: str, description: str) -> str      # Super writes the route system prompt
async def run(key: str, project_id: str, job_id: int, name: str, description: str,
              examples: list[dict], kind: str, system_prompt: str | None) -> dict
    examples: [{"input": str, "expected": str | None}]. Upserts routes row, updates the job, returns result.
"""
import asyncio, json, re, time

from . import cascade, config, db, limits

MAX_EXAMPLES = 50
CONCURRENCY = 4
NANO_MAX_TOKENS = 512
BIG_MAX_TOKENS = 1024
TOLERANCE = 0.02
SMALL_N = 20  # below this, nudge the threshold up one distinct c (§6a: start conservative, recalibration lowers it)

PROMPT_WRITER = ("You write system prompts for LLM-backed API routes. Given a task description, write a short, "
                 "practical system prompt (under 150 words) that makes a model do the task well: role, output format, "
                 "key rules. Reply with the prompt text only, no preamble or quotes.")

JUDGE = ("You are a strict, impartial grader. Compare two answers to the same request and decide which is better "
         "for the task (correctness first, then completeness and format). Ignore length and style unless the task "
         'asks for them. Reply in JSON: {"winner": "A"|"B"|"tie"}.')
JUDGE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": {
    "type": "object", "properties": {"winner": {"type": "string", "enum": ["A", "B", "tie"]}}, "required": ["winner"]}}}


def check(output, expected, kind):
    out = (output or "").strip()
    if kind == "json":
        try:
            got = json.loads(out)
        except ValueError:
            return False
        if expected is None:
            return True
        try:
            return got == json.loads(expected)
        except ValueError:
            return None
    if expected is None:
        return None
    if kind == "exact":
        return out.casefold() == expected.strip().casefold()
    if kind == "contains":
        return expected.strip().casefold() in out.casefold()
    if kind == "regex":
        return re.search(expected, out) is not None
    return None  # "judge" or unknown


async def _ask(key, tier, messages, **kw):
    """One chat call via limits. Returns (answer_text, logprobs_content). Raises on non-200."""
    status, data, _, _, _ = await limits.call(key, {"model": config.TIERS[tier], "messages": messages, **kw},
                                              priority="background")
    if status != 200 or not isinstance(data, dict):
        raise RuntimeError(f"{tier} call failed: {status} {str(data)[:200]}")
    choice = data["choices"][0]
    text = re.sub(r"(?s)<think>.*?</think>", "", choice["message"].get("content") or "").strip()
    return text, (choice.get("logprobs") or {}).get("content")


async def judge(key, task, input_text, expected, a, b, context=None):
    # context = the route's system prompt/policy; without it the judge grades style, not facts (PLAN §8b)
    rules = f"\n\nInstructions/policy the answers must follow:\n{context}" if context else ""
    ref = f"\n\nReference answer (known good):\n{expected}" if expected else ""

    async def once(first, second):
        user = f"Task: {task}{rules}\n\nRequest:\n{input_text}{ref}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}"
        text, _ = await _ask(key, "ultra", [{"role": "system", "content": JUDGE}, {"role": "user", "content": user}],
                             max_tokens=50, temperature=0, response_format=JUDGE_SCHEMA, **config.NO_THINK)
        try:
            return json.loads(text).get("winner", "tie")
        except (ValueError, AttributeError):
            return "tie"

    w1, w2 = await asyncio.gather(once(a, b), once(b, a))
    return ((w1 in ("A", "tie")) + (w2 in ("B", "tie"))) / 2


def choose_threshold(rows, tolerance=TOLERANCE, min_coverage=0.2, escalate_to="super"):
    """`ultra_ok` is the reference tier's grade (run() passes the escalate_to tier's)."""
    n = len(rows)
    if not n:
        return {"policy": "cascade", "tier": "nano", "threshold": config.DEFAULT_THRESHOLD, "coverage": 0.0,
                "nano_acc_above": 0.0, "ultra_acc": 0.0, "n": 0}
    ultra_acc = sum(r["ultra_ok"] for r in rows) / n
    bar = ultra_acc - tolerance - 1e-9
    cs = sorted({r["c"] for r in rows}, reverse=True)

    def above(t):
        sel = [r for r in rows if r["c"] >= t]
        return sum(r["nano_ok"] for r in sel) / len(sel), len(sel) / n

    base = {"ultra_acc": ultra_acc, "n": n}
    direct = {**base, "policy": "direct", "tier": escalate_to, "threshold": None}
    acc_all, _ = above(cs[-1])
    if acc_all >= bar:
        return {**base, "policy": "direct", "tier": "nano", "threshold": None, "coverage": 1.0, "nano_acc_above": acc_all}
    good = [t for t in cs if above(t)[0] >= bar]
    if not good:
        return {**direct, "coverage": 0.0, "nano_acc_above": 0.0}
    i = cs.index(min(good))
    if n < SMALL_N and i > 0:  # tiny sample: one step more conservative
        i -= 1
    acc, cov = above(cs[i])
    if cov < min_coverage:
        return {**direct, "coverage": cov, "nano_acc_above": acc}
    return {**base, "policy": "cascade", "tier": "nano", "threshold": cs[i], "coverage": cov, "nano_acc_above": acc}


async def generate_prompt(key, description):
    text, _ = await _ask(key, "super", [{"role": "system", "content": PROMPT_WRITER},
                                        {"role": "user", "content": description}], max_tokens=400, **config.NO_THINK)
    return text


def _p(s):
    return None if s is None else s[:120]


async def _example(key, description, system_prompt, kind, ex):
    msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": ex["input"]}]
    (nano, lp), (sup, _), (ult, _) = await asyncio.gather(
        _ask(key, "nano", msgs, max_tokens=NANO_MAX_TOKENS, logprobs=True, **config.NO_THINK),
        _ask(key, "super", msgs, max_tokens=BIG_MAX_TOKENS),
        _ask(key, "ultra", msgs, max_tokens=BIG_MAX_TOKENS))
    exp = ex.get("expected")
    ref = exp if exp is not None else ult  # no reference reply -> Ultra's answer is the reference

    async def grade(ans, is_ref=False):
        ok = check(ans, exp, kind)
        if ok is None:
            ok = is_ref or await judge(key, description, ex["input"], None, ans, ref, system_prompt) >= 0.5
        return bool(ok)

    n_ok, s_ok, u_ok = await asyncio.gather(grade(nano), grade(sup), grade(ult, is_ref=exp is None))
    return {"c": cascade.confidence(lp, config.DEFAULT_FIRST_N), "nano_ok": n_ok, "super_ok": s_ok, "ultra_ok": u_ok,
            "input": _p(ex["input"]), "expected": _p(exp), "nano": _p(nano), "super": _p(sup), "ultra": _p(ult)}


async def run(key, project_id, job_id, name, description, examples, kind, system_prompt):
    try:
        system_prompt = system_prompt or await generate_prompt(key, description)
        sem = asyncio.Semaphore(CONCURRENCY)

        async def one(ex):
            async with sem:
                return await _example(key, description, system_prompt, kind, ex)

        done = await asyncio.gather(*(one(ex) for ex in examples[:MAX_EXAMPLES]), return_exceptions=True)
        rows = [r for r in done if isinstance(r, dict)]
        errors = [str(e)[:200] for e in done if not isinstance(e, dict)]
        if examples and not rows:
            raise RuntimeError(f"all examples failed: {errors[0]}")
        n = len(rows) or 1
        acc = {t: sum(r[f"{t}_ok"] for r in rows) / n for t in config.ORDER}
        escalate_to = "super" if acc["super"] >= acc["ultra"] - TOLERANCE - 1e-9 else "ultra"
        # Nano is held to the bar of the tier it would escalate to.
        pick = choose_threshold([{"c": r["c"], "nano_ok": r["nano_ok"], "ultra_ok": r[f"{escalate_to}_ok"]}
                                 for r in rows], escalate_to=escalate_to)
        calibration = {**pick, "escalate_to": escalate_to, "accuracies": acc, "errors": errors, "rows": rows}
        db.x("""INSERT INTO routes(project_id, name, description, system_prompt, policy, tier, threshold, first_n,
                  escalate_to, calibration, examples, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(project_id, name) DO UPDATE SET description=excluded.description,
                  system_prompt=excluded.system_prompt, policy=excluded.policy, tier=excluded.tier,
                  threshold=excluded.threshold, first_n=excluded.first_n, escalate_to=excluded.escalate_to,
                  calibration=excluded.calibration, examples=excluded.examples, created_at=excluded.created_at""",
             (project_id, name, description, system_prompt, pick["policy"], pick["tier"], pick["threshold"],
              config.DEFAULT_FIRST_N, escalate_to, json.dumps(calibration),
              json.dumps([{"input": e["input"], "expected": e.get("expected")} for e in examples]), time.time()))
        result = {**calibration, "system_prompt": system_prompt}
        db.job_update(job_id, "done", result)
        return result
    except Exception as e:
        err = {"error": f"{type(e).__name__}: {e}"[:500]}
        db.job_update(job_id, "failed", err)
        return err
