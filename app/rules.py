"""Store rules as data + a checker in plain code (the "check the rules" step). CONTRACT:

Rules file: rules/trailhead.json  (built from ABCD's data/abcd/guidelines.json + kb.json; store = "Trailhead Outfitters")
  {"store": "Trailhead Outfitters",
   "subflows": {
     "<subflow>": {                                   # the 55 ABCD subflows, e.g. "return_size", "recover_password"
        "flow": "<flow>",                             # one of the 10 ABCD flows
        "description": str,                           # one plain sentence
        "required_fields": [field, ...],              # what we must know first, e.g. ["full_name","username","email","order_id"]
        "actions": [action, ...],                     # ordered tool/action names from kb.json, e.g. ["pull-up-account","validate-purchase",...]
        "conditions": [{"when": [{"field": f, "op": "eq|neq|in|not_in|gt|gte|lt|lte|exists|missing", "value": v}, ...],
                        "then": "allow|deny|escalate", "say": str}],   # evaluated in order; first match wins
        "policy_text": str}}}                         # the original guideline sentences (for the reply writer)

def load(path="rules/trailhead.json") -> dict
def subflows() -> list[str]
def required(subflow: str) -> list[str]
def plan(subflow: str, facts: dict) -> dict
    -> {"status": "ok"|"need_info"|"escalate"|"deny", "missing": [field...], "actions": [action...],
        "matched": {"then", "say"} | None, "policy_text": str}
    need_info when any required field is missing from facts; otherwise first matching condition decides; default ok.
    Plain code only; no model calls; no eval().
async def draft(key: str, policy_text: str, store: str) -> dict
    Super drafts a rules dict (same shape) from a store's prose policy. A person reviews it before use.

Details (compatible additions):
- The file also carries "facts": {field: type} -- every field name required_fields/conditions may use (FACTS below).
- required_fields entry "a|b" (or "a|b|c") = any one of them satisfies it; plan() lists it as "a|b" in missing.
- A fact counts as missing when absent, None or "". Every op except "missing" is False on a missing fact
  (so "neq"/"not_in" never match an unknown value). Strings compare case-insensitively; bools accept yes/no/true/false.
- A clause's "when" is AND; OR = several conditions with the same "then". A condition with empty "when" always matches.
- company_error is the ask-the-oracle answer (true = the store made the mistake). Rules that need it escalate while
  it's unknown; call plan() again once the oracle answered.
- plan() always returns the full ordered actions; the caller acts only on status "ok".
- Unknown subflow -> ValueError naming it.
"""
import ast, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = os.path.join(ROOT, "rules", "trailhead.json")

MEMBERSHIP = ["guest", "bronze", "silver", "gold"]
SHIPPING = ["order received", "in transit", "out for delivery", "delivered"]
PAYMENT = ["credit card", "debit card", "paypal"]
FACTS = {  # field -> "str" | "num" | "bool" | [allowed values]
    "full_name": "str", "account_id": "str", "username": "str", "email": "str", "order_id": "str",
    "phone": "str", "zip_code": "str", "pin_number": "str", "security_answer": "str", "address": "str",
    "new_value": "str", "product": "str", "reason": "str",
    "membership": MEMBERSHIP, "shipping_status": SHIPPING, "payment_method": PAYMENT,
    "refund_method": ["gift card", "add value", "paper check", "credit card"],
    "return_method": ["by mail", "in store", "drop off center"],
    "heard_from": ["email", "spouse", "news"],
    "credit_source": ["subscription refund", "previous purchase", "promotional package"],
    "amount": "num", "days_since_purchase": "num", "days_waiting": "num", "days_late": "num", "promo_days_old": "num",
    "has_receipt": "bool", "original_packaging": "bool", "company_error": "bool", "item_in_stock": "bool",
    "already_paid": "bool",
}
OPS = {"eq", "neq", "in", "not_in", "gt", "gte", "lt", "lte", "exists", "missing"}
THEN = {"allow": "ok", "deny": "deny", "escalate": "escalate"}

_rules = None


def load(path=DEFAULT):
    global _rules
    with open(path if os.path.isabs(path) else os.path.join(ROOT, path), encoding="utf-8") as f:
        _rules = json.load(f)
    return _rules


def _sub(subflow):
    r = _rules or load()
    if subflow not in r["subflows"]:
        raise ValueError(f"unknown subflow {subflow!r} (store {r['store']!r} has {len(r['subflows'])} subflows)")
    return r["subflows"][subflow]


def subflows():
    return list((_rules or load())["subflows"])


def required(subflow):
    return list(_sub(subflow)["required_fields"])


# ---------------------------------------------------------------- evaluator (plain code, no eval)
def _missing(v):
    return v is None or (isinstance(v, str) and not v.strip())


def _norm(field, v):
    """Fact or rule value -> comparable value; raises ValueError if it doesn't fit the field's type."""
    t = FACTS.get(field, "str")
    if t == "num":
        if isinstance(v, bool):
            raise ValueError(v)
        return float(str(v).replace("$", "").replace(",", "").strip()) if isinstance(v, str) else float(v)
    if t == "bool":
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
        raise ValueError(v)
    return str(v).strip().lower()


def test(clause, facts):
    f, op, want = clause["field"], clause["op"], clause.get("value")
    v = facts.get(f)
    if op == "missing":
        return _missing(v)
    if _missing(v):
        return False
    if op == "exists":
        return True
    try:
        v = _norm(f, v)
        if op in ("in", "not_in"):
            hit = v in [_norm(f, w) for w in want]
            return hit if op == "in" else not hit
        w = _norm(f, want)
    except (ValueError, TypeError):
        return False
    if op == "eq":
        return v == w
    if op == "neq":
        return v != w
    if isinstance(v, str) or isinstance(v, bool):  # ordering only for numbers
        return False
    return {"gt": v > w, "gte": v >= w, "lt": v < w, "lte": v <= w}[op]


def _have(req, facts):
    return any(not _missing(facts.get(f)) for f in req.split("|"))


def plan(subflow, facts):
    s = _sub(subflow)
    facts = facts or {}
    out = {"status": "ok", "missing": [r for r in s["required_fields"] if not _have(r, facts)],
           "actions": list(s["actions"]), "matched": None, "policy_text": s["policy_text"]}
    if out["missing"]:
        out["status"] = "need_info"
        return out
    for c in s["conditions"]:
        if all(test(cl, facts) for cl in c["when"]):
            out["status"], out["matched"] = THEN[c["then"]], {"then": c["then"], "say": c["say"]}
            break
    return out


# ---------------------------------------------------------------- validation (for drafted / hand-written rules)
def check_condition(c):
    """Returns a cleaned condition or raises ValueError saying what's wrong."""
    if c.get("then") not in THEN:
        raise ValueError(f"bad then {c.get('then')!r}")
    if not isinstance(c.get("say"), str) or not c["say"].strip():
        raise ValueError("say must be a non-empty string")
    when = []
    for cl in c.get("when") or []:
        f, op, v = cl.get("field"), cl.get("op"), cl.get("value")
        if f not in FACTS:
            raise ValueError(f"unknown field {f!r}")
        if op not in OPS:
            raise ValueError(f"unknown op {op!r}")
        t = FACTS[f]
        if op in ("exists", "missing"):
            v = None
        elif op in ("in", "not_in"):
            if not isinstance(v, list) or not v:
                raise ValueError(f"{f} {op} needs a non-empty list")
            v = [_norm(f, x) for x in v]
        else:
            if op in ("gt", "gte", "lt", "lte") and t != "num":
                raise ValueError(f"{op} on non-number field {f}")
            v = _norm(f, v)
        if isinstance(t, list) and v is not None and any(x not in t for x in (v if isinstance(v, list) else [v])):
            raise ValueError(f"{f}: {v!r} not in {t}")
        if t == "num" and isinstance(v, float) and v.is_integer():
            v = int(v)
        when.append({"field": f, "op": op, "value": v})
    return {"when": when, "then": c["then"], "say": c["say"].strip()}


def check_required(fields):
    for r in fields:
        bad = [f for f in r.split("|") if f not in FACTS]
        if bad:
            raise ValueError(f"unknown required field(s) {bad}")
    return list(fields)


# ---------------------------------------------------------------- drafting with Super (builder + store-pasted policy)
PROMPT = """You turn a store's customer-support policy into machine-checkable rules.
For each subflow listed, write the decision conditions the policy states (who may get what: membership tiers, time
windows, amount limits, shipping status, whether the company made the error, ...). Rules are checked in order; the
first whose "when" clauses ALL hold decides. Put the most specific / most generous allowed cases first, then denials.
"then": allow (go ahead), deny (policy says no), escalate (a manager/human must decide or a system check is needed).
"say": one short sentence explaining the decision to the customer. Use ONLY these fact fields:
{facts}
Ops: eq, neq, in, not_in, gt, gte, lt, lte, exists, missing. value_json is the JSON value (string, number, true/false,
or a list for in/not_in; null for exists/missing). Durations are in days (6 months = 180). Lower-case enum values.
If the policy states no condition for a subflow, return an empty conditions list. Don't invent rules the text lacks.
required_fields: facts we must know before deciding; write "a|b" when either one is enough."""


def _schema(ids):
    clause = {"type": "object", "additionalProperties": False, "required": ["field", "op", "value_json"],
              "properties": {"field": {"type": "string", "enum": list(FACTS)}, "op": {"type": "string", "enum": sorted(OPS)},
                             "value_json": {"type": "string"}}}
    cond = {"type": "object", "additionalProperties": False, "required": ["when", "then", "say"],
            "properties": {"when": {"type": "array", "items": clause}, "then": {"type": "string", "enum": list(THEN)},
                           "say": {"type": "string"}}}
    sub = {"type": "object", "additionalProperties": False,
           "required": ["subflow", "description", "required_fields", "conditions"],
           "properties": {"subflow": {"type": "string", "enum": ids}, "description": {"type": "string"},
                          "required_fields": {"type": "array", "items": {"type": "string"}},
                          "conditions": {"type": "array", "items": cond}}}
    return {"type": "json_schema", "json_schema": {"name": "rules", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["subflows"],
        "properties": {"subflows": {"type": "array", "items": sub}}}}}


def _facts_doc():
    return "\n".join(f"- {f}: {t if isinstance(t, str) else 'one of ' + ', '.join(t)}" for f, t in FACTS.items())


async def ask(key, policy_text, ids, max_tokens=4000):
    """One Super call. Returns (raw: {subflow: {...}} with conditions still in value_json form, usage)."""
    from . import config, tf
    body = {"model": config.TIERS["super"], "temperature": 0, "max_tokens": max_tokens, **config.NO_THINK,
            "response_format": _schema(ids),
            "messages": [{"role": "system", "content": PROMPT.format(facts=_facts_doc())},
                         {"role": "user", "content": f"Subflows: {', '.join(ids)}\n\nPolicy:\n{policy_text}"}]}
    status, data, _ = await tf.chat(key, body)
    if status != 200:
        raise RuntimeError(f"Super draft failed: {status} {str(data)[:200]}")
    raw = json.loads(data["choices"][0]["message"]["content"])
    return {s.pop("subflow"): s for s in raw["subflows"] if s.get("subflow") in ids}, data.get("usage")


def _value(op, s):
    """Model's value_json -> value. Lenient: JSON, else a Python literal (ast, never eval), else comma list / raw text."""
    if op in ("exists", "missing") or s in (None, ""):
        return None
    for parse in (json.loads, ast.literal_eval):
        try:
            return parse(s)
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return [x.strip() for x in s.split(",")] if op in ("in", "not_in") else s


def validate(raw_sub):
    """Drafted subflow -> (conditions, required_fields, dropped: [reason]). Invalid conditions are dropped whole,
    and so is a drafted condition with no clauses (it would match everything)."""
    conds, dropped = [], []
    for c in raw_sub.get("conditions") or []:
        try:
            when = [{"field": cl["field"], "op": cl["op"], "value": _value(cl["op"], cl.get("value_json"))}
                    for cl in c.get("when") or []]
            if not when:
                raise ValueError("empty when")
            conds.append(check_condition({**c, "when": when}))
        except (ValueError, KeyError, TypeError) as e:
            dropped.append(f"{e}: {json.dumps(c)[:160]}")
    req = []
    for r in raw_sub.get("required_fields") or []:
        try:
            req += check_required([r.replace(" ", "")])
        except ValueError as e:
            dropped.append(str(e))
    return conds, req, dropped


async def draft(key, policy_text, store):
    """Super drafts rules for the ABCD subflows a store's prose policy covers; actions/flow come from kb/ontology."""
    with open(os.path.join(ROOT, "data", "abcd", "ontology.json"), encoding="utf-8") as f:
        flows = json.load(f)["intents"]["subflows"]
    with open(os.path.join(ROOT, "data", "abcd", "kb.json"), encoding="utf-8") as f:
        kb = json.load(f)
    flow_of = {s: fl for fl, subs in flows.items() for s in subs}
    raw, _ = await ask(key, policy_text, list(flow_of), max_tokens=16000)
    out = {}
    for s, d in raw.items():
        conds, req, _ = validate(d)
        out[s] = {"flow": flow_of[s], "description": d.get("description", ""), "required_fields": req,
                  "actions": kb.get(s, []), "conditions": conds, "policy_text": policy_text}
    return {"store": store, "facts": FACTS, "subflows": out}
