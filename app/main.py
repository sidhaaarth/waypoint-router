"""FastAPI app: OpenAI-compatible router (/v1) + dashboard API (/api) + dashboard page."""
import asyncio, csv, html, io, json, re, statistics, time
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse

from . import calibrate, cascade, config, costs, db, keys, limits, lint, pipeline, rules, starter, tf, tracing

app = FastAPI(title="Waypoint", docs_url=None, redoc_url=None)  # /docs is the product docs page
STATIC = Path(__file__).resolve().parent.parent / "static"
_prices_loaded = False
_tasks = set()  # keep background task refs alive


async def ensure_prices(tf_key):
    global _prices_loaded
    if not _prices_loaded:
        status, data = await tf.models(tf_key)
        if status == 200:
            costs.load(data)
            _prices_loaded = True


def resolve(project_id, model):
    """model -> ("route", route_dict) | ("model", model_id). 'auto' uses route 'default' if it exists."""
    name = "default" if model in (None, "", "auto") else model
    row = db.one("SELECT * FROM routes WHERE project_id=? AND name=?", (project_id, name))
    if row:
        return "route", row
    if name == "default":  # uncalibrated: conservative cascade
        return "route", {"name": "default", "policy": "cascade", "threshold": config.DEFAULT_THRESHOLD,
                         "first_n": config.DEFAULT_FIRST_N, "escalate_to": "super", "system_prompt": None}
    return "model", model


def with_system(body, system_prompt):
    """Prepend the route's static system prompt (cache-friendly: static first, caller's messages after)."""
    if not system_prompt:
        return body
    return {**body, "messages": [{"role": "system", "content": system_prompt}, *body.get("messages", [])]}


def decision_record(meta, with_values):
    """D55: the per-reply decision record stored with each request (one JSON column, no new store).
    Without the project's opt-in, customer values (names, emails, addresses, tool args) are left out: we keep WHAT was
    found, by whom and how sure, which is what analytics needs. With opt-in, values are kept (distillation data)."""
    pm = meta.get("pipeline_meta") or {}
    calls = [{"tier": a.get("tier"), "model": a.get("model"), "usage": a.get("usage"), "confidence": a.get("confidence")}
             for a in meta.get("attempts", [])]
    rec = {"tier": meta.get("tier"), "escalated": bool(meta.get("escalated")), "calls": calls}
    if not pm:  # cascade / passthrough request
        return rec
    intent, plan, vis, look = pm.get("intent") or {}, pm.get("plan") or {}, pm.get("vision") or {}, pm.get("lookup") or {}
    rec.update({
        "channel": pm.get("channel"), "turn": pm.get("turn"), "open_task": pm.get("open_task"),
        "vision": {k: vis.get(k) for k in ("label", "confidence", "model", "skipped", "match")} if vis else None,
        "intent": {"subflow": intent.get("subflow"), "flow": intent.get("flow"), "confidence": intent.get("confidence"),
                   "by": intent.get("by"), "steps": [{k: s.get(k) for k in ("step", "label", "confidence", "by")}
                                                     for s in intent.get("steps") or []],
                   "memory": intent.get("memory")},  # past cases shown to Nano: label + similarity only, no text
        "fields": {f: {"by": v.get("by"), "confidence": v.get("confidence"), "source": v.get("source"),
                       **({"value": v.get("value")} if with_values else {})}
                   for f, v in (pm.get("fields") or {}).items()},
        "plan": {"status": plan.get("status"), "missing": plan.get("missing"), "actions": plan.get("actions"),
                 "rule": (plan.get("matched") or {}).get("then")},
        "tools": [{"action": t.get("action"), "ok": t.get("ok"), **({"args": t.get("args")} if with_values else {})}
                  for t in pm.get("tool_calls") or []],
        "lookup": {"found": look.get("found"), "credits": look.get("credits")} if look else None,
        "writer": pm.get("writer"), "safety": {"ok": (pm.get("safety") or {}).get("ok"),
                                               "issues": len((pm.get("safety") or {}).get("issues") or [])}})
    return rec


def log_request(project, route_name, user, meta, started, stream, messages=None):
    usage_all = [a.get("usage") for a in meta.get("attempts", [])]
    final = usage_all[-1] if usage_all else None
    cost = sum(costs.cost(a["model"], a.get("usage")) for a in meta.get("attempts", []))
    # "What would Ultra have cost?" Cascade: the answer alone on Ultra. Pipeline: every step's tokens on Ultra (the same
    # work, all on Ultra), otherwise a 6-call pipeline is compared to one writer call and shows fake negative savings.
    if meta.get("pipeline"):
        ultra = sum(costs.ultra_equivalent(a.get("usage")) for a in meta.get("attempts", []))
    else:
        ultra = costs.ultra_equivalent(final)
    meta["ultra_cost_usd"] = ultra
    cached = ((final or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    db.x("""INSERT INTO requests(project_id, route, user, ts, tier, model, confidence, escalated, fallback, status,
            latency_ms, stream, prompt_tokens, completion_tokens, cached_tokens, cost_usd, ultra_cost_usd, decision)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
         (project["id"], route_name, user, time.time(), meta.get("tier"), meta.get("model"), meta.get("confidence"),
          int(bool(meta.get("escalated"))), int(bool(meta.get("fallback"))), meta.get("status", 200),
          int((time.time() - started) * 1000), int(stream), (final or {}).get("prompt_tokens"),
          (final or {}).get("completion_tokens"), cached, cost, ultra,
          json.dumps(decision_record(meta, bool(project.get("log_opt_in"))), default=str)))
    rid = tracing.PARENT.get()
    if rid:
        tracing.end_parent(rid, started, f"router:{route_name or 'passthrough'}",
                           {"route": route_name, "project": project["id"], "stream": stream},
                           {**{k: meta.get(k) for k in ("tier", "model", "confidence", "escalated", "fallback", "status")},
                            "cost_usd": cost})
    return cost


def router_headers(meta, cost=None):
    h = {"x-router-tier": str(meta.get("tier")), "x-router-model": str(meta.get("model")),
         "x-router-escalated": str(bool(meta.get("escalated"))).lower(),
         "x-router-fallback": str(bool(meta.get("fallback"))).lower()}
    if meta.get("confidence") is not None:
        h["x-router-confidence"] = f"{meta['confidence']:.4f}"
    if cost is not None:
        h["x-router-cost-usd"] = f"{cost:.8f}"
    if meta.get("ultra_cost_usd") is not None:
        h["x-router-ultra-cost-usd"] = f"{meta['ultra_cost_usd']:.8f}"
    return h


# ---------------------------------------------------------------- OpenAI-compatible
@app.post("/v1/chat/completions")
async def chat_completions(request: Request, project=Depends(keys.auth)):
    body = await request.json()
    started, key, user = time.time(), project["tf_key"], body.get("user")
    tracing.start_parent()  # every Token Factory call below nests under this request's trace
    tracing.SHARE_TEXT.set(bool(project.get("log_opt_in")))  # user text leaves only with opt-in
    await ensure_prices(key)
    kind, target = resolve(project["id"], body.get("model"))
    stream = bool(body.get("stream"))
    priority = request.headers.get("x-router-priority", "interactive")

    if kind == "model":  # explicit real model id: pass through (still paced, logged, costed)
        route_name, route = None, {"policy": "direct", "model": target}
    else:
        route_name, route = target["name"], dict(target)
        if route.get("policy") == "pipeline":  # extract -> rules -> act -> reply; builds its own prompts
            return await pipeline_response(project, key, body, route, user, started, stream, priority,
                                           request.headers.get("x-router-channel"))
        body = with_system(body, route.get("system_prompt"))
        if costs.over_budget(project["id"], user):  # Gap 6: customer near budget -> Nano only
            route = {**route, "policy": "direct", "tier": "nano"}
    body = {k: v for k, v in body.items() if k != "model"}

    if route.get("model"):  # passthrough
        body["model"] = route["model"]
        tier = config.MODEL_TIER.get(route["model"], "custom")
        if stream:
            return await passthrough_stream(project, key, body, tier, user, started, priority)
        status, data, _, model_used, fell_back = await limits.call(key, body, priority)
        meta = {"tier": tier, "model": model_used, "confidence": None, "escalated": False, "fallback": fell_back,
                "status": status, "attempts": [{"tier": tier, "model": model_used,
                                                "usage": data.get("usage") if isinstance(data, dict) else None}]}
        cost = log_request(project, None, user, meta, started, False, body.get("messages"))
        return JSONResponse(data if isinstance(data, dict) else {"error": {"message": data}}, status,
                            headers=router_headers(meta, cost))

    if stream:
        gen, meta = await cascade.complete_stream(key, body, route, priority)

        async def relay():
            try:
                async for b in gen:
                    yield b
            finally:
                maybe_log_training(project, route_name, body, meta, None)
                log_request(project, route_name, user, meta, started, True, body.get("messages"))
        # tier/model may change after the first chunk; headers reflect the decision point
        return StreamingResponse(relay(), media_type="text/event-stream")

    data, meta = await cascade.complete(key, body, route, priority)
    cost = log_request(project, route_name, user, meta, started, False, body.get("messages"))
    maybe_log_training(project, route_name, body, meta, data)
    return JSONResponse(data, meta.get("status", 200), headers=router_headers(meta, cost))


async def pipeline_response(project, key, body, route, user, started, stream, priority, channel):
    """Route policy "pipeline". Channel: x-router-channel header, else chat if the thread has >1 customer turn."""
    msgs = body.get("messages", [])
    channel = channel or ("chat" if sum(m.get("role") == "user" for m in msgs) > 1 else "email")
    reply, pm = await pipeline.run(key, msgs, route, channel=channel, priority=priority, store_note=body.get("store_note"))
    tiers = [a["tier"] for a in pm.get("attempts", [])]
    meta = {"tier": "ultra" if "ultra" in tiers else "nano", "model": f"pipeline:{pm.get('writer')}",
            "confidence": (pm.get("intent") or {}).get("confidence"), "escalated": "ultra" in tiers,
            "fallback": False, "status": pm.get("status", 200), "attempts": pm.get("attempts", []), "pipeline": True,
            "pipeline_meta": pm}
    cost = log_request(project, route["name"], user, meta, started, stream, msgs)
    info = {k: pm.get(k) for k in ("channel", "vision", "turn", "intent", "open_task", "fields", "plan", "tool_calls", "lookup", "writer", "safety")}
    resp = {"id": f"pipe-{int(started * 1000)}", "object": "chat.completion", "created": int(started),
            "model": meta["model"], "router": info,  # extra key: OpenAI clients ignore it, dashboards read it
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}
    maybe_log_training(project, route["name"], {"messages": msgs}, meta, resp)  # opt-in only: feeds nightly memory growth
    headers = {**router_headers(meta, cost), "x-router-subflow": str((pm.get("intent") or {}).get("subflow")),
               "x-router-plan": str((pm.get("plan") or {}).get("status"))}
    if not stream:
        return JSONResponse(resp, meta["status"], headers=headers)

    async def one_chunk():  # ponytail: pipeline decides before writing, so it streams the finished reply at once
        chunk = {**resp, "object": "chat.completion.chunk",
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}
        yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
        yield b"data: [DONE]\n\n"
    return StreamingResponse(one_chunk(), media_type="text/event-stream", headers=headers)


async def passthrough_stream(project, key, body, tier, user, started, priority):
    meta = {"tier": tier, "model": body["model"], "confidence": None, "escalated": False, "fallback": False,
            "status": 200, "attempts": [{"tier": tier, "model": body["model"], "usage": None}]}

    async def gen():
        try:
            async with limits.call_stream(key, {**body, "stream_options": {"include_usage": True}}, priority) as (_, chunks, model_used, fell_back):
                meta.update(model=model_used, fallback=fell_back)
                meta["attempts"][0]["model"] = model_used
                async for c in chunks:
                    if c.get("usage"):
                        meta["attempts"][0]["usage"] = c["usage"]
                    yield b"data: " + json.dumps(c).encode() + b"\n\n"
        except tf.TFError as e:
            meta["status"] = e.status
            yield b"data: " + json.dumps({"error": {"message": str(e.body)[:500], "code": e.status}}).encode() + b"\n\n"
        finally:
            log_request(project, None, user, meta, started, True, body.get("messages"))
        yield b"data: [DONE]\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


def maybe_log_training(project, route_name, body, meta, data):
    """Gap 3 data: opt-in only. Keep escalated (teacher) answers; those are the distillation targets."""
    if not project.get("log_opt_in") or not data or meta.get("status", 200) != 200:
        return
    try:
        answer = data["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError):
        return
    db.x("INSERT INTO training(project_id, route, messages, answer, tier, confidence, ts) VALUES (?,?,?,?,?,?,?)",
         (project["id"], route_name, json.dumps(body.get("messages")), answer, meta.get("tier"),
          meta.get("confidence"), time.time()))


@app.get("/v1/models")
async def list_models(project=Depends(keys.auth)):
    routes = [r["name"] for r in db.q("SELECT name FROM routes WHERE project_id=?", (project["id"],))]
    ids = ["auto", *routes, *config.TIERS.values()]
    return {"object": "list", "data": [{"id": i, "object": "model", "owned_by": "router"} for i in dict.fromkeys(ids)]}


# ---------------------------------------------------------------- dashboard API
EMAIL_RE = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
USERNAME_RE = re.compile(r"[A-Za-z0-9_.\-]{3,32}")
USE_CASES = ("email", "chat", "both")


def signup_profile(payload):
    """Validated signup fields (site /signup) -> dict, or HTTPException(400) naming the first problem."""
    p = {f: str(payload.get(f) or "").strip() for f in db.PROFILE}
    for f, v in p.items():
        if not v:
            raise HTTPException(400, f"{f.replace('_', ' ')} is required")
        if len(v) > 200:
            raise HTTPException(400, f"{f.replace('_', ' ')} is too long")
    if not EMAIL_RE.fullmatch(p["email"]):
        raise HTTPException(400, "email doesn't look like an email address")
    if not USERNAME_RE.fullmatch(p["username"]):
        raise HTTPException(400, "username must be 3-32 letters, numbers, '.', '_' or '-'")
    if p["use_case"] not in USE_CASES:
        raise HTTPException(400, "use case must be email, chat or both")
    if db.one("SELECT 1 FROM projects WHERE username=? COLLATE NOCASE", (p["username"],)):
        raise HTTPException(409, "that username is taken")
    return p


@app.post("/api/projects")
async def create_project(payload: dict):
    tf_key = (payload.get("tf_key") or "").strip()
    if not tf_key:
        raise HTTPException(400, "Token Factory API key is required")
    profile = signup_profile(payload)
    status, _ = await tf.models(tf_key)  # validate before storing
    if status != 200:
        raise HTTPException(400, f"Token Factory rejected this key (HTTP {status})")
    pid, rk = keys.create_project(payload.get("name") or profile["company"], tf_key)  # TF key Fernet-encrypted there
    db.x(f"UPDATE projects SET {', '.join(f + '=?' for f in profile)} WHERE id=?", (*profile.values(), pid))
    return {"project_id": pid, "router_key": rk}


@app.get("/api/me")
def me(project=Depends(keys.auth)):
    return {"project_id": project["id"], "name": project["name"], "log_opt_in": bool(project["log_opt_in"]),
            **{f: project.get(f) for f in db.PROFILE}}


@app.post("/api/session")
def login(request: Request, project=Depends(keys.auth)):
    """Log in: keep the router key in an HttpOnly cookie until /api/session is deleted (Log out)."""
    rk = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    resp = JSONResponse({"ok": True})
    resp.set_cookie(keys.SESSION, rk, max_age=30 * 86400, httponly=True, samesite="lax",
                    secure=request.url.scheme == "https")
    return resp


@app.get("/api/session")
def session(request: Request):
    return {"signed_in": _signed_in(request), "self_hosted": config.SELF_HOSTED}


@app.delete("/api/session")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(keys.SESSION)
    return resp


def _signed_in(request):
    rk = request.cookies.get(keys.SESSION, "")
    return rk.startswith("rk-") and keys.project_for(rk) is not None


@app.put("/api/settings")
def settings(payload: dict, project=Depends(keys.auth)):
    db.x("UPDATE projects SET log_opt_in=? WHERE id=?", (int(bool(payload.get("log_opt_in"))), project["id"]))
    return me({**project, "log_opt_in": bool(payload.get("log_opt_in"))})


@app.get("/api/overview")
def overview(hours: float = 24, project=Depends(keys.auth)):
    since = time.time() - hours * 3600
    rows = db.q("SELECT * FROM requests WHERE project_id=? AND ts>=? ORDER BY ts", (project["id"], since))
    n = len(rows)
    cost = sum(r["cost_usd"] or 0 for r in rows)
    ultra = sum(r["ultra_cost_usd"] or 0 for r in rows)
    mix = {t: sum(r["tier"] == t for r in rows) for t in config.ORDER}
    buckets = {}
    for r in rows:
        b = buckets.setdefault(int(r["ts"] // 3600 * 3600), {"requests": 0, "cost_usd": 0.0, **{t: 0 for t in config.ORDER}})
        b["requests"] += 1
        b["cost_usd"] += r["cost_usd"] or 0
        if r["tier"] in config.ORDER:
            b[r["tier"]] += 1
    confs = [r["confidence"] for r in rows if r["confidence"] is not None]
    return {"requests": n, "cost_usd": cost, "ultra_cost_usd": ultra,
            "savings_pct": round(100 * (1 - cost / ultra), 1) if ultra else None, "tier_mix": mix,
            "escalation_rate": sum(r["escalated"] or 0 for r in rows) / n if n else 0,
            "fallbacks": sum(r["fallback"] or 0 for r in rows),
            "errors": sum((r["status"] or 200) >= 400 for r in rows),
            "p50_latency_ms": statistics.median([r["latency_ms"] for r in rows]) if rows else None,
            "avg_confidence": sum(confs) / len(confs) if confs else None,
            "timeseries": [{"t": t, **v} for t, v in sorted(buckets.items())]}


@app.get("/api/insights")
def insights(hours: float = 24, project=Depends(keys.auth)):
    """D55: what the decision records say, aggregated (no customer values involved)."""
    from collections import Counter
    rows = db.q("SELECT decision FROM requests WHERE project_id=? AND ts>=? AND decision IS NOT NULL",
                (project["id"], time.time() - hours * 3600))
    recs = [json.loads(r["decision"]) for r in rows]
    pipe = [r for r in recs if r.get("plan")]
    c = lambda it: dict(Counter(x for x in it if x is not None).most_common())  # noqa: E731
    tools = [t for r in pipe for t in r.get("tools") or []]
    tool_ok = {}
    for t in tools:
        s = tool_ok.setdefault(t["action"], {"calls": 0, "ok": 0})
        s["calls"] += 1
        s["ok"] += bool(t.get("ok"))
    fields = [f for r in pipe for f in (r.get("fields") or {}).values()]
    calls = [k for r in recs for k in r.get("calls") or []]
    confs = [r["intent"]["confidence"] for r in pipe if (r.get("intent") or {}).get("confidence") is not None]
    looks = [r["lookup"] for r in pipe if r.get("lookup")]
    mem = [r["intent"] for r in pipe if (r.get("intent") or {}).get("memory")]  # memory of past cases (RAG step)
    return {
        "replies": len(recs), "pipeline_replies": len(pipe),
        "subflows": c((r.get("intent") or {}).get("subflow") for r in pipe),
        "plan_status": c((r.get("plan") or {}).get("status") for r in pipe),
        "missing_asked": c(m for r in pipe for m in (r.get("plan") or {}).get("missing") or []),
        "intent_by": c((r.get("intent") or {}).get("by") for r in pipe),
        "intent_conf_avg": sum(confs) / len(confs) if confs else None,
        "turn_types": c((r.get("turn") or {}).get("type") for r in pipe),
        "writer": c(r.get("writer") if r.get("writer") not in (None, "none") else "template" for r in pipe),  # template: ready-made reply
        "field_sources": c(f.get("by") for f in fields),
        "tools": tool_ok,
        "tool_success": sum(s["ok"] for s in tool_ok.values()) / len(tools) if tools else None,
        "calls_by_tier": c(k.get("tier") for k in calls),
        "vision": c((r.get("vision") or {}).get("label") or ("off" if (r.get("vision") or {}).get("skipped") else None)
                    for r in pipe),
        "lookups": {"n": len(looks), "found": sum(bool(x.get("found")) for x in looks)},
        "safety_pass": sum(bool((r.get("safety") or {}).get("ok")) for r in pipe) / len(pipe) if pipe else None,
        "open_tasks": sum(bool(r.get("open_task")) for r in pipe),
        "memory": {"used": len(mem), "agree": sum(i["memory"][0]["label"] == i.get("subflow") for i in mem),
                   "avg_top": sum(i["memory"][0]["score"] for i in mem) / len(mem) if mem else None}}


@app.get("/api/customers")
def customers(project=Depends(keys.auth)):
    rows = db.q("""SELECT r.user, COUNT(*) requests, SUM(r.cost_usd) cost_usd, c.budget_usd FROM requests r
                   LEFT JOIN customers c ON c.project_id=r.project_id AND c.user=r.user
                   WHERE r.project_id=? AND r.user IS NOT NULL GROUP BY r.user ORDER BY cost_usd DESC""", (project["id"],))
    budget_only = db.q("""SELECT user, budget_usd FROM customers WHERE project_id=? AND user NOT IN
                          (SELECT DISTINCT user FROM requests WHERE project_id=? AND user IS NOT NULL)""",
                       (project["id"], project["id"]))
    rows += [{"user": b["user"], "requests": 0, "cost_usd": 0.0, "budget_usd": b["budget_usd"]} for b in budget_only]
    for r in rows:
        r["pct"] = round(100 * (r["cost_usd"] or 0) / r["budget_usd"], 1) if r["budget_usd"] else None
    return rows


@app.put("/api/customers/{user}")
def set_budget(user: str, payload: dict, project=Depends(keys.auth)):
    budget = payload.get("budget_usd")
    if budget in (None, ""):
        db.x("DELETE FROM customers WHERE project_id=? AND user=?", (project["id"], user))
    else:
        db.x("INSERT INTO customers(project_id, user, budget_usd) VALUES (?,?,?) "
             "ON CONFLICT(project_id, user) DO UPDATE SET budget_usd=excluded.budget_usd", (project["id"], user, float(budget)))
    return {"user": user, "budget_usd": budget}


@app.get("/api/routes")
def routes(project=Depends(keys.auth)):
    stats = lint.cache_stats(project["id"])
    out = []
    for r in db.q("SELECT * FROM routes WHERE project_id=? ORDER BY name", (project["id"],)):
        r["calibration"] = json.loads(r["calibration"]) if r["calibration"] else None
        r["examples"] = json.loads(r["examples"]) if r.get("examples") else None
        if r["examples"] is None and r["calibration"]:  # tuned before examples were kept: only 120-char excerpts survive
            r["examples"] = [{"input": x.get("input"), "expected": x.get("expected")} for x in r["calibration"].get("rows", [])]
            r["examples_partial"] = True
        if r["calibration"]:
            r["calibration"].pop("rows", None)  # per-example model answers: not needed by the dashboard
        r["lint"] = lint.lint(r["system_prompt"])
        r["cache"] = stats.get(r["name"])
        out.append(r)
    return out


@app.post("/api/routes")
async def create_route(payload: dict, project=Depends(keys.auth)):
    name = (payload.get("name") or "").strip()
    examples = [e for e in payload.get("examples") or [] if (e.get("input") or "").strip()]
    if not name or name == "auto" or name in config.TIERS.values():
        raise HTTPException(400, "route name required (not 'auto' or a model id)")
    if payload.get("policy") == "pipeline":  # extract -> rules -> act -> reply: uses the store's rules, no calibration run
        vision_on = int(payload.get("vision", True) not in (False, 0, "false", "off"))
        db.x("""INSERT INTO routes(project_id, name, description, policy, threshold, store_url, vision, created_at)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(project_id, name) DO UPDATE SET description=excluded.description,
                policy=excluded.policy, threshold=excluded.threshold, store_url=excluded.store_url, vision=excluded.vision""",
             (project["id"], name, payload.get("description") or "", "pipeline",
              float(payload.get("threshold") or config.DEFAULT_THRESHOLD), payload.get("store_url"), vision_on, time.time()))
        return {"route": name, "policy": "pipeline", "vision": bool(vision_on)}
    if not examples:
        raise HTTPException(400, "at least one example with an input is required")
    job_id = db.job_create(project["id"], "calibrate")
    rid, t0 = tracing.start_parent()  # the task copies this context: its calls nest under one trace
    tracing.SHARE_TEXT.set(bool(project.get("log_opt_in")))
    tracing.end_parent(rid, t0, f"calibrate:{name}", {"route": name, "examples": len(examples)}, {"job_id": job_id})
    task = asyncio.create_task(calibrate.run(project["tf_key"], project["id"], job_id, name,
                                             payload.get("description") or "", examples,
                                             payload.get("check") or "judge", payload.get("system_prompt")))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return {"job_id": job_id}


@app.put("/api/routes/{name}")
def edit_route(name: str, payload: dict, project=Depends(keys.auth)):
    """Edit a connection without re-tuning: description, system prompt; pipeline: store URL and Nano cut-off.
    Changed reference conversations need a re-tune instead (POST /api/routes with the same name)."""
    row = db.one("SELECT policy FROM routes WHERE project_id=? AND name=?", (project["id"], name))
    if not row:
        raise HTTPException(404, f"no route {name!r}")
    sets = {}
    if "description" in payload:
        sets["description"] = str(payload["description"] or "")[:4000]
    if "system_prompt" in payload and row["policy"] != "pipeline":  # the pipeline builds its own prompts
        sets["system_prompt"] = str(payload["system_prompt"] or "").strip()[:20000] or None
    if row["policy"] == "pipeline" and "threshold" in payload:
        try:
            t = float(payload["threshold"])
        except (TypeError, ValueError):
            raise HTTPException(400, "threshold must be a number")
        if not 0 < t < 1:
            raise HTTPException(400, "threshold must be between 0 and 1")
        sets["threshold"] = t
    if row["policy"] == "pipeline" and "store_url" in payload:
        url = str(payload["store_url"] or "").strip().rstrip("/")
        if url and not re.fullmatch(r"https?://[^\s/]+(/\S*)?", url):
            raise HTTPException(400, "store URL must start with http:// or https://")
        sets["store_url"] = url or None
    if sets:
        db.x(f"UPDATE routes SET {', '.join(k + '=?' for k in sets)} WHERE project_id=? AND name=?",
             (*sets.values(), project["id"], name))
    return {"route": name, "updated": sorted(sets)}


@app.put("/api/routes/{name}/settings")
def route_settings(name: str, payload: dict, project=Depends(keys.auth)):
    """Per-route switches. Today: {"vision": bool} turns photo analysis on/off (text-only stores never pay for it)."""
    if not db.one("SELECT id FROM routes WHERE project_id=? AND name=?", (project["id"], name)):
        raise HTTPException(404, f"no route {name!r}")
    if "vision" in payload:
        db.x("UPDATE routes SET vision=? WHERE project_id=? AND name=?", (int(bool(payload["vision"])), project["id"], name))
    return {"route": name, "vision": bool(db.one("SELECT vision FROM routes WHERE project_id=? AND name=?",
                                                 (project["id"], name))["vision"])}


@app.get("/api/jobs/{job_id}")
def job(job_id: int, project=Depends(keys.auth)):
    row = db.one("SELECT * FROM jobs WHERE id=? AND project_id=?", (job_id, project["id"]))
    if not row:
        raise HTTPException(404, "job not found")
    row["result"] = json.loads(row["result"]) if row["result"] else None
    return row


@app.get("/api/requests")
def recent(limit: int = 50, project=Depends(keys.auth)):
    return db.q("""SELECT ts, route, user, tier, model, confidence, escalated, fallback, status, latency_ms,
                   cost_usd, stream FROM requests WHERE project_id=? ORDER BY ts DESC LIMIT ?""",
                (project["id"], min(limit, 500)))


@app.get("/api/limits")
def limit_state(project=Depends(keys.auth)):
    s = limits.state(project["tf_key"])
    d = s.get("dynamic", {})  # dashboard contract names
    s["dynamic"] = {**d, "usage_requests_pct": d.get("usage_pct_requests"), "usage_tokens_pct": d.get("usage_pct_tokens"),
                    "scale_requests": d.get("scale_requests"), "period_remaining_s": d.get("period_remaining_s")}
    return s


# ---------------------------------------------------------------- knowledge ("Teach your router", /teach)
KNOWLEDGE = {  # uploadable kind -> what it will be used for (shown back to the uploader)
    "policy": "drafted into rules by Super; a person reviews them before they go live (D41)",
    "faq": "stored answers the reply writer can quote exactly",
    "good": "memory of past cases: examples the model copies",
    "bad": "test suite: replies are checked against these, never copied",
    "urls": "trusted lookup sites",
    "triggers": "safety override: these always go to a person",
    "scope": "scope gate: what the router helps with (anything else is handed off)",
}
KNOWLEDGE_NOTE = "Stored in your project. Wiring into the pipeline (memory, tests, scope gate) happens in the next step."
KNOWLEDGE_MAX = 2_000_000  # bytes per upload


def _records(text):
    """JSONL (one object per line) or CSV with a header row -> [(line_no, dict | None)]."""
    if text.startswith("{"):
        out = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    out.append((n, json.loads(line)))
                except ValueError:
                    out.append((n, None))
        return out
    return [(n, {(k or "").strip().lower(): v.strip() if isinstance(v, str) else v for k, v in r.items()})
            for n, r in enumerate(csv.DictReader(io.StringIO(text)), 2)]


def parse_knowledge(kind, text):
    """-> (items, errors). One parse for the page's preview (dry_run) and the stored count."""
    text = text.strip()
    if not text:
        return [], ["nothing to upload"]
    items, errs = [], []
    if kind in ("good", "bad"):
        need = ("request_type", "reply") + (("what_went_wrong",) if kind == "bad" else ())
        for n, r in _records(text):
            if not isinstance(r, dict):
                errs.append(f"line {n}: not valid JSON")
                continue
            miss = ([] if r.get("message") or r.get("messages") else ["message"]) + [
                f for f in need if not str(r.get(f) or "").strip()]
            if miss:
                errs.append(f"row {n}: missing {', '.join(miss)}")
            else:
                items.append(r)
    elif kind == "faq":
        head = text.splitlines()[0].lower()
        if "question" in head and "answer" in head:  # CSV question,answer
            for n, r in _records(text):
                q, a = (r or {}).get("question"), (r or {}).get("answer")
                if q and a:
                    items.append({"question": q, "answer": a})
                else:
                    errs.append(f"row {n}: needs a question and an answer")
        else:  # "Q: ..." / "A: ..." text; extra lines continue the current part
            for line in text.splitlines():
                s = line.strip()
                if s[:2].lower() == "q:":
                    items.append({"question": s[2:].strip(), "answer": ""})
                elif items and s[:2].lower() == "a:":
                    items[-1]["answer"] = s[2:].strip()
                elif items and s:
                    k = "answer" if items[-1]["answer"] else "question"
                    items[-1][k] = (items[-1][k] + " " + s).strip()
            errs = [f"question {i}: no answer" for i, it in enumerate(items, 1) if not it["answer"]]
            if not items:
                errs = ['no questions found: use "Q: ..." / "A: ..." lines or a question,answer CSV']
    elif kind in ("urls", "triggers"):
        items = [s for s in (l.strip() for l in text.splitlines()) if s and not s.startswith("#")]
        if kind == "urls":
            errs = [f"line {i}: not a web address: {u[:80]}" for i, u in enumerate(items, 1)
                    if not re.match(r"https?://[^\s/]+\.\S+$", u)]
    else:  # policy (counted in lines) / scope (one paragraph)
        items = [l for l in text.splitlines() if l.strip()] if kind == "policy" else [text]
    return items, errs


def _save_knowledge(project_id, kind, filename, content, items):
    return db.x("INSERT INTO knowledge(project_id, kind, filename, content, items, created_at) VALUES (?,?,?,?,?,?)",
                (project_id, kind, filename, content, items, time.time()))


async def _draft_rules(project, job_id, text):
    """Background: Super drafts rules from the policy; stored as kind rules_draft for a person to review (D41)."""
    try:
        d = await rules.draft(project["tf_key"], text, project.get("name") or "")
        n = len(d.get("subflows") or {})
        kid = _save_knowledge(project["id"], "rules_draft", "rules_draft.json", json.dumps(d), n)
        db.job_update(job_id, "done", {"knowledge_id": kid, "subflows": n, "needs_review": True})
    except Exception as e:  # the draft is optional: report it, don't crash
        db.job_update(job_id, "failed", {"error": str(e)[:300]})


@app.post("/api/knowledge")
def add_knowledge(payload: dict, background: BackgroundTasks, dry_run: bool = False, draft_rules: bool = False,
                  project=Depends(keys.auth)):
    kind, content = payload.get("kind"), payload.get("content")
    filename = str(payload.get("filename") or "")[:200]
    if kind not in KNOWLEDGE:
        raise HTTPException(400, f"kind must be one of {', '.join(KNOWLEDGE)}")
    if not isinstance(content, str):
        raise HTTPException(400, "content must be text")
    if len(content.encode()) > KNOWLEDGE_MAX:
        raise HTTPException(413, "upload is over 2 MB: split it into smaller files")
    if kind == "policy" and (filename.lower().endswith((".html", ".htm")) or content.lstrip().startswith("<")):
        content = html.unescape(re.sub(r"(?is)<(script|style)\b.*?</\1>|<[^>]+>", " ", content))
        content = re.sub(r"[ \t]+", " ", re.sub(r"\n\s*\n+", "\n\n", content))
    items, errors = parse_knowledge(kind, content)
    out = {"kind": kind, "items": len(items), "errors": errors[:20], "error_count": len(errors),
           "preview": items[:3], "used_for": KNOWLEDGE[kind], "note": KNOWLEDGE_NOTE}
    if dry_run:
        return out
    if errors:
        raise HTTPException(400, out)
    out["id"] = _save_knowledge(project["id"], kind, filename, content.strip(), len(items))
    if draft_rules and kind == "policy":
        out["job_id"] = db.job_create(project["id"], "draft_rules")
        background.add_task(_draft_rules, project, out["job_id"], content.strip())
    return out


async def _starter_pack(project, job_id, business):
    try:
        pack = await starter.build(project["tf_key"], business)
        kid = _save_knowledge(project["id"], "starter_pack", "starter_pack.json", json.dumps(pack),
                              len(pack["request_types"]))
        db.job_update(job_id, "done", {"knowledge_id": kid, "request_types": len(pack["request_types"]),
                                       "credits": pack["credits"], "needs_review": True})
    except Exception as e:
        db.job_update(job_id, "failed", {"error": str(e)[:300]})


@app.post("/api/knowledge/starter_pack")
def start_starter_pack(background: BackgroundTasks, payload: dict | None = None, project=Depends(keys.auth)):
    """Background: public help-centre pages (Tavily) + Super -> routing hints for review. Never answers or rules."""
    business = str((payload or {}).get("business_type") or project.get("business_type") or "").strip()[:200]
    if not business:
        raise HTTPException(400, "set a business type first")
    job_id = db.job_create(project["id"], "starter_pack")
    background.add_task(_starter_pack, project, job_id, business)
    return {"job_id": job_id, "business_type": business}


@app.post("/api/knowledge/{kid}/accept")
def accept_starter_pack(kid: int, payload: dict | None = None, project=Depends(keys.auth)):
    """Copy a reviewed starter pack into approved knowledge (request types, triggers, scope), then drop the pack.
    The page may send edited request_types / human_topics / scope; anything left out keeps the pack's value."""
    row = db.one("SELECT * FROM knowledge WHERE id=? AND project_id=? AND kind='starter_pack'", (kid, project["id"]))
    if not row:
        raise HTTPException(404, "starter pack not found")
    pack = {**json.loads(row["content"]), **{k: v for k, v in (payload or {}).items()
                                             if k in ("request_types", "human_topics", "scope")}}
    types, topics = pack.get("request_types") or [], [str(t).strip() for t in pack.get("human_topics") or [] if str(t).strip()]
    ids = {"request_types": _save_knowledge(project["id"], "request_types", "starter_pack",
                                            json.dumps({"request_types": types, "sources": pack.get("sources", [])}),
                                            len(types))}
    if topics:
        ids["triggers"] = _save_knowledge(project["id"], "triggers", "starter_pack", "\n".join(topics), len(topics))
    if str(pack.get("scope") or "").strip():
        ids["scope"] = _save_knowledge(project["id"], "scope", "starter_pack", pack["scope"].strip(), 1)
    db.x("DELETE FROM knowledge WHERE id=?", (kid,))
    return {"accepted": ids, "note": KNOWLEDGE_NOTE}


@app.get("/api/knowledge")
def list_knowledge(project=Depends(keys.auth)):
    return db.q("""SELECT id, kind, filename, items, length(content) AS chars, created_at FROM knowledge
                   WHERE project_id=? ORDER BY created_at DESC, id DESC""", (project["id"],))


@app.get("/api/knowledge/{kid}")
def get_knowledge(kid: int, project=Depends(keys.auth)):
    row = db.one("SELECT * FROM knowledge WHERE id=? AND project_id=?", (kid, project["id"]))
    if not row:
        raise HTTPException(404, "not found")
    return row


@app.delete("/api/knowledge/{kid}")
def delete_knowledge(kid: int, project=Depends(keys.auth)):
    if not db.one("SELECT id FROM knowledge WHERE id=? AND project_id=?", (kid, project["id"])):
        raise HTTPException(404, "not found")
    db.x("DELETE FROM knowledge WHERE id=?", (kid,))
    return {"deleted": kid}


# ---------------------------------------------------------------- pages
BUNDLE = ["Dockerfile", "docker-compose.yml", ".dockerignore", "requirements.txt", "SELF_HOST.md",
          "app", "static"]  # allow-list: never secrets, databases or demo data
BUNDLE_SKIP = {"static/replay.html", *(f"static/site/{p}.html" for p in ("home", "how", "results", "models"))}  # showcase


@app.get("/download/waypoint.zip", include_in_schema=False)
def download_bundle():
    """Everything needed to build and run Waypoint in your own network (docker compose up --build). Public."""
    import zipfile
    if config.SELF_HOSTED:  # the image has no Dockerfile or compose file to hand out
        raise HTTPException(404, "not found")
    root, buf = STATIC.parent, io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in BUNDLE:
            p = root / name
            for f in sorted([p] if p.is_file() else p.rglob("*")):
                rel = f.relative_to(root).as_posix()
                if f.is_file() and "__pycache__" not in f.parts and rel not in BUNDLE_SKIP:
                    z.write(f, "waypoint/" + ("README.md" if rel == "SELF_HOST.md" else rel))
    return StreamingResponse(iter([buf.getvalue()]), media_type="application/zip",
                             headers={"Content-Disposition": 'attachment; filename="waypoint.zip"'})


@app.get("/healthz")
def healthz():
    db.conn()
    return {"ok": True}


SITE = STATIC / "site"
SITE_PAGES = {"/": "home.html", "/how": "how.html", "/results": "results.html", "/models": "models.html",
              "/signup": "signup.html", "/login": "login.html", "/teach": "teach.html",
              "/onboarding": "onboarding.html", "/docs": "docs.html"}
SIGNED_OUT_ONLY = {"/", "/how", "/results", "/models", "/signup", "/login"}  # signed in -> dashboard
SIGNED_IN_ONLY = {"/dashboard", "/chat", "/onboarding"}  # signed out -> login
SHOWCASE = {"/", "/how", "/results", "/models", "/embed/demo"}  # not in the self-hosted copy


def _page(name):
    return lambda: FileResponse(SITE / name)


for _path, _file in SITE_PAGES.items():  # signed-out marketing site + signup/login
    app.add_api_route(_path, _page(_file), methods=["GET"], include_in_schema=False)


BRAND = STATIC / "brand"


@app.get("/brand/{name}", include_in_schema=False)
def brand(name: str):
    path = (BRAND / name).resolve()
    if path.parent != BRAND.resolve() or path.suffix != ".png" or not path.is_file():
        raise HTTPException(404, "not found")
    return FileResponse(path)


@app.middleware("http")
async def page_guard(request, call_next):
    """Signed-in visitors never see the signed-out site; signed-out visitors can't open the app (?demo=1 aside)."""
    path = request.url.path.rstrip("/") or "/"
    if config.SELF_HOSTED and path in SHOWCASE:
        return RedirectResponse("/dashboard" if _signed_in(request) else "/login", 303)
    if request.method == "GET" and path in SIGNED_OUT_ONLY | SIGNED_IN_ONLY:
        signed_in = _signed_in(request)
        if signed_in and path in SIGNED_OUT_ONLY:
            return RedirectResponse("/dashboard", 303)
        if not signed_in and path in SIGNED_IN_ONLY and request.query_params.get("demo") != "1":
            return RedirectResponse("/login", 303)
    return await call_next(request)


@app.middleware("http")
async def no_stale_pages(request, call_next):
    """Pages and scripts change often during the demo: make browsers re-check instead of showing old copies."""
    resp = await call_next(request)
    if request.method == "GET" and not request.url.path.startswith(("/api/", "/v1/")):
        resp.headers.setdefault("Cache-Control", "no-cache")
    return resp


@app.get("/site/site.js", include_in_schema=False)
def site_js():
    return FileResponse(SITE / "site.js", media_type="text/javascript")


@app.get("/site/samples/{kind}/{name}", include_in_schema=False)
def site_sample(kind: str, name: str):
    path = (SITE / "samples" / kind / name).resolve()
    if path.parent.parent != (SITE / "samples").resolve() or not path.is_file():  # only samples/<type>/<file>
        raise HTTPException(404, "sample not found")
    return FileResponse(path, media_type="text/plain; charset=utf-8")


@app.get("/dashboard")
def dashboard():
    return FileResponse(STATIC / "index.html")


@app.get("/chat")
def chat_page():
    return FileResponse(STATIC / "chat.html")


@app.get("/embed/demo")
def demo_embed():
    """The looping customer-chat video embedded on the home page (kiosk mode only; no stand-alone page)."""
    return FileResponse(STATIC / "replay.html")


ROOT = STATIC.parent
RESULTS, PHOTOS = ROOT / "data" / "published", ROOT / "data" / "photos"  # published = the result snapshot the site shows


@app.get("/api/conversations")
def conversations(run: str | None = None):
    """Recorded conversations (data/published/<run>/conversations.jsonl), newest run by default.
    ponytail: no auth, it's demo test data; put it behind keys.auth before showing real customer traffic."""
    runs = sorted((p.parent.name for p in RESULTS.glob("*/conversations.jsonl")), reverse=True)
    if run is not None and run not in runs:
        raise HTTPException(404, f"no conversations for run {run!r}")
    pick = run or (runs[0] if runs else None)
    items = []
    if pick:
        with open(RESULTS / pick / "conversations.jsonl", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    return {"run": pick, "runs": runs, "items": items}


def _latest(pattern):
    """Newest data/published/<run>/<file> matching pattern, parsed, or None."""
    hits = sorted(RESULTS.glob(pattern), key=lambda p: p.parent.name, reverse=True)
    return json.loads(hits[0].read_text(encoding="utf-8")) if hits else None


@app.get("/api/benchmark")
def benchmark():
    """Results page data: the recorded benchmark results (no auth: aggregate test numbers only)."""
    latest = RESULTS / "latest.json"
    return {"classification": json.loads(latest.read_text(encoding="utf-8")) if latest.exists() else None,
            "pipeline": _latest("*/pipeline_summary.json"), "photos": _latest("*/photo_summary.json"),
            "intent": _latest("*/intent_summary.json")}


@app.get("/media/photos/{name}")
def photo(name: str):
    path = (PHOTOS / name).resolve()
    if path.parent != PHOTOS.resolve() or not path.is_file():  # no path tricks, only files in data/photos
        raise HTTPException(404, "photo not found")
    return FileResponse(path)
