"""LangSmith tracing for EVERY Token Factory call (hooked in tf.py), nested under a parent run when one is set.

- tf.chat / tf.stream -> one "llm" run each (model, usage, latency, status).
- Router requests open a parent "chain" run (main.py) so their model calls nest under it.
- Message text: included only when SHARE_TEXT is true in the current context. Router requests set it from the
  project's opt-in; operator tools (bench, probes) can leave the default (True) since it's our own data.
Operator-level: needs LANGSMITH_API_KEY in the router's env. Never raises.
"""
import asyncio, os, time, uuid
from contextvars import ContextVar
from datetime import datetime, timezone

import httpx

KEY = os.environ.get("LANGSMITH_API_KEY")
URL = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/") + "/api/v1/runs"
PROJECT = os.environ.get("LANGSMITH_PROJECT", "nemotron-router")
PARENT: ContextVar = ContextVar("ls_parent", default=None)
SHARE_TEXT: ContextVar = ContextVar("ls_share_text", default=True)
_tasks = set()


def _iso(ts):
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


def _post(run):
    if not KEY:
        return
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    t = loop.create_task(_send(run))
    _tasks.add(t)
    t.add_done_callback(_tasks.discard)


async def _send(run):
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(URL, json=run, headers={"x-api-key": KEY})
    except httpx.HTTPError:
        pass  # tracing must never break serving


def llm_run(body, status, data, started, ended, stream=False):
    """Called by tf.py after every model call."""
    usage = data.get("usage") if isinstance(data, dict) else None
    inputs = {"model": body.get("model"), "stream": stream,
              **{k: body[k] for k in ("max_tokens", "temperature", "logprobs", "chat_template_kwargs") if k in body}}
    outputs = {"status": status, "usage": usage}
    if SHARE_TEXT.get():
        inputs["messages"] = body.get("messages")
        if isinstance(data, dict) and data.get("choices"):
            outputs["content"] = (data["choices"][0].get("message") or {}).get("content")
    run = {"id": str(uuid.uuid4()), "name": body.get("model", "model").split("/")[-1], "run_type": "llm",
           "session_name": PROJECT, "start_time": _iso(started), "end_time": _iso(ended), "inputs": inputs,
           "outputs": outputs, "extra": {"metadata": {"ls_model_name": body.get("model"),
                                                       "ls_provider": "nebius-token-factory"}}}
    if status != 200:
        run["error"] = str(data)[:500]
    if PARENT.get():
        run["parent_run_id"] = PARENT.get()
    _post(run)


def start_parent():
    """Open a parent run for a router request; model calls made after this nest under it."""
    rid = str(uuid.uuid4())
    PARENT.set(rid)
    return rid, time.time()


def end_parent(rid, started, name, inputs, outputs):
    _post({"id": rid, "name": name, "run_type": "chain", "session_name": PROJECT, "start_time": _iso(started),
           "end_time": _iso(time.time()), "inputs": inputs, "outputs": outputs})


if __name__ == "__main__":
    sent = []
    _post = lambda run: sent.append(run)  # noqa: E731
    SHARE_TEXT.set(False)
    llm_run({"model": "nvidia/x", "messages": [{"role": "user", "content": "secret"}]}, 200,
            {"usage": {"total_tokens": 3}, "choices": [{"message": {"content": "hi"}}]}, 0, 1)
    assert "messages" not in sent[0]["inputs"] and "content" not in sent[0]["outputs"]
    rid, _ = start_parent()
    SHARE_TEXT.set(True)
    llm_run({"model": "nvidia/x", "messages": []}, 529, "overloaded", 0, 1)
    assert sent[1]["parent_run_id"] == rid and sent[1]["error"] == "overloaded" and "messages" in sent[1]["inputs"]
    print("tracing ok")
