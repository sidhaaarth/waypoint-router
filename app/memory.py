"""Memory of past cases for the INTENT step: nearest labelled ABCD TRAIN conversations by embedding similarity.
Built by scripts/build_memory_index.py. One embedding call per query (~$0.000002) + an in-memory cosine top-k.

    cases = await memory.nearest(key, text, k=3, project_id=pid)   # [{"label", "text", "score"}]
    system_prompt += "\n\n" + memory.as_examples(cases)

Growth: a store's own confirmed cases go to data/memory/<project_id>/ (scripts/nightly_memory.py -> add_cases);
nearest() searches that project index plus the ABCD base index.
"""
import json, os, re
from pathlib import Path

import numpy as np

from . import tf

MODEL, DIMS, CHARS = "Qwen/Qwen3-Embedding-8B", 1024, 800  # Matryoshka: the API truncates to `dimensions`
PRICE = 1e-08  # $/token, live catalog 2026-10-30
INDEX = Path(__file__).resolve().parent.parent / "data" / "memory" / "base" / "abcd_index.npy"
PROJECTS = Path(os.environ.get("MEMORY_DIR") or INDEX.parent.parent)  # per-project indexes; a mounted volume in Docker
_loaded = {}  # ponytail: in-memory numpy scan, fine to ~1M rows; a vector DB / FAISS beyond that


def load(path=INDEX):
    """-> (matrix [n, DIMS] float32 L2-normalized, [{"label", "text"}]). Cached per path."""
    path = Path(path)
    if path not in _loaded and not path.exists():  # a project with no cases yet
        return np.zeros((0, DIMS), np.float32), []
    if path not in _loaded:
        labels = json.loads(_labels_path(path).read_text(encoding="utf-8"))
        _loaded[path] = (np.load(path), labels)
    return _loaded[path]


async def embed(key, texts):
    """-> (float32 [len(texts), DIMS] L2-normalized, usage). Raises tf.TFError on a non-200."""
    r = await tf.client.post("/embeddings", json={"model": MODEL, "input": texts, "dimensions": DIMS}, headers=tf._h(key))
    if r.status_code != 200:
        raise tf.TFError(r.status_code, r.text[:300], dict(r.headers))
    j = r.json()
    v = np.array([d["embedding"] for d in sorted(j["data"], key=lambda d: d["index"])], dtype=np.float32)[:, :DIMS]
    return v / np.linalg.norm(v, axis=1, keepdims=True), j.get("usage") or {}


def top(matrix, labels, q, k, exclude_text=None):
    """Pure numpy top-k by cosine (rows and q are normalized). exclude_text skips that exact past case."""
    scores = matrix @ q
    out = []
    for i in np.argsort(-scores):
        if exclude_text is not None and labels[i]["text"] == exclude_text.strip()[:200]:
            continue
        out.append({"label": labels[i]["label"], "text": labels[i]["text"], "score": float(scores[i])})
        if len(out) == k:
            break
    return out


def _labels_path(path):
    return path.with_name(path.name.replace("_index.npy", "_labels.json"))


def index_for(project_id):
    """Per-project index path (MEMORY_DIR, default data/memory, /<project_id>/cases_index.npy)."""
    if not re.fullmatch(r"[\w-]+", str(project_id)):
        raise ValueError("bad project id")
    return PROJECTS / project_id / "cases_index.npy"


async def nearest(key, text, k=3, exclude_text=None, project_id=None, path=INDEX):
    """Top-k past cases over the base index (+ the project's own index when project_id is given)."""
    q, _ = await embed(key, [text.strip()[:CHARS]])
    paths = [path] + ([index_for(project_id)] if project_id else [])
    hits = [c for p in paths for c in top(*load(p), q[0], k, exclude_text)]
    return sorted(hits, key=lambda c: -c["score"])[:k]


async def add_cases(key, cases, path=INDEX, dedup=0.98):
    """Embed [{"text", "label"}], append to the index at `path` (created if missing) and save. A case whose cosine to
    an existing or earlier new case is > dedup is skipped. -> number added."""
    path = Path(path)
    matrix, labels = load(path)
    labels, new = list(labels), []
    for i in range(0, len(cases), 128):
        chunk = cases[i:i + 128]
        vecs, _ = await embed(key, [c["text"].strip()[:CHARS] for c in chunk])
        for c, v in zip(chunk, vecs):
            if (len(matrix) and float((matrix @ v).max()) > dedup) or any(float(n @ v) > dedup for n in new):
                continue
            new.append(v)
            labels.append({"label": c["label"], "text": c["text"].strip()[:200]})
    added = len(new)
    if added:
        matrix = np.vstack([matrix, *new])
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{path}.tmp", "wb") as f:  # write-then-rename: a crash never leaves a half-written index
            np.save(f, matrix.astype(np.float32))
        os.replace(f"{path}.tmp", path)
        _labels_path(path).write_text(json.dumps(labels), encoding="utf-8")
        _loaded[path] = (matrix, labels)
    return added


def as_examples(cases):
    """Prompt block of past cases, most similar first."""
    lines = [f'Past case: "{" ".join(c["text"].split())}" -> {c["label"]}' for c in cases]
    return "Similar past cases from this store, with their correct labels:\n" + "\n".join(lines) if lines else ""
