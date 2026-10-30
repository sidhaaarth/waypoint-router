"""SQLite storage. Modules write their own SQL through q()/x()."""
import json, sqlite3, threading, time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
  id TEXT PRIMARY KEY, name TEXT, tf_key_enc BLOB NOT NULL, log_opt_in INTEGER DEFAULT 0, created_at REAL);
CREATE TABLE IF NOT EXISTS router_keys (
  hash TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id), prefix TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS routes (
  id INTEGER PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL, description TEXT,
  system_prompt TEXT, policy TEXT DEFAULT 'cascade', tier TEXT, threshold REAL, first_n INTEGER,
  escalate_to TEXT DEFAULT 'super', store_url TEXT, vision INTEGER DEFAULT 1,
  calibration TEXT, created_at REAL, UNIQUE(project_id, name));
CREATE TABLE IF NOT EXISTS requests (
  id INTEGER PRIMARY KEY, project_id TEXT, route TEXT, user TEXT, ts REAL, tier TEXT, model TEXT,
  confidence REAL, escalated INTEGER, fallback INTEGER, status INTEGER, latency_ms INTEGER, stream INTEGER,
  prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER, cost_usd REAL, ultra_cost_usd REAL,
  decision TEXT);
CREATE INDEX IF NOT EXISTS requests_project_ts ON requests(project_id, ts);
CREATE TABLE IF NOT EXISTS customers (
  project_id TEXT, user TEXT, budget_usd REAL, PRIMARY KEY(project_id, user));
CREATE TABLE IF NOT EXISTS training (
  id INTEGER PRIMARY KEY, project_id TEXT, route TEXT, messages TEXT, answer TEXT, tier TEXT,
  confidence REAL, score REAL, ts REAL);
CREATE TABLE IF NOT EXISTS jobs (
  id INTEGER PRIMARY KEY, project_id TEXT, kind TEXT, status TEXT, result TEXT, created_at REAL, updated_at REAL);
CREATE TABLE IF NOT EXISTS knowledge (
  id INTEGER PRIMARY KEY, project_id TEXT NOT NULL, kind TEXT NOT NULL, filename TEXT, content TEXT,
  items INTEGER, created_at REAL);
CREATE INDEX IF NOT EXISTS knowledge_project ON knowledge(project_id, kind);
"""

PROFILE = ("first_name", "last_name", "username", "email", "company", "use_case", "business_type")

_lock = threading.Lock()  # ponytail: one connection + global lock; move to Postgres if write contention shows up
_conn = None


def conn():
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.executescript(SCHEMA)
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(routes)")}
        if "store_url" not in cols:  # databases created before the pipeline route existed
            _conn.execute("ALTER TABLE routes ADD COLUMN store_url TEXT")
        if "vision" not in cols:  # per-route photo analysis switch (D51)
            _conn.execute("ALTER TABLE routes ADD COLUMN vision INTEGER DEFAULT 1")
        if "examples" not in cols:  # reference conversations (JSON list), kept so the connection can be edited
            _conn.execute("ALTER TABLE routes ADD COLUMN examples TEXT")
        if "decision" not in {r[1] for r in _conn.execute("PRAGMA table_info(requests)")}:  # D55 decision record (JSON)
            _conn.execute("ALTER TABLE requests ADD COLUMN decision TEXT")
        pcols = {r[1] for r in _conn.execute("PRAGMA table_info(projects)")}
        for c in PROFILE:  # signup profile (site /signup); older projects keep NULLs
            if c not in pcols:
                _conn.execute(f"ALTER TABLE projects ADD COLUMN {c} TEXT")
        # NULLs don't collide, so projects created before signup existed are fine
        _conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS projects_username ON projects(username COLLATE NOCASE)")
    return _conn


def q(sql, args=()):
    with _lock:
        return [dict(r) for r in conn().execute(sql, args).fetchall()]


def one(sql, args=()):
    rows = q(sql, args)
    return rows[0] if rows else None


def x(sql, args=()):
    with _lock:
        c = conn()
        cur = c.execute(sql, args)
        c.commit()
        return cur.lastrowid


def job_create(project_id, kind):
    return x("INSERT INTO jobs(project_id, kind, status, created_at, updated_at) VALUES (?,?,?,?,?)",
             (project_id, kind, "running", time.time(), time.time()))


def job_update(job_id, status, result=None):
    x("UPDATE jobs SET status=?, result=?, updated_at=? WHERE id=?",
      (status, json.dumps(result) if result is not None else None, time.time(), job_id))


if __name__ == "__main__":
    config.DB_PATH = ":memory:"
    x("INSERT INTO projects(id, name, tf_key_enc, created_at) VALUES ('p1','t',?,0)", (b"k",))
    assert one("SELECT name FROM projects WHERE id='p1'")["name"] == "t"
    j = job_create("p1", "calibrate"); job_update(j, "done", {"t": 0.8})
    assert json.loads(one("SELECT result FROM jobs WHERE id=?", (j,))["result"]) == {"t": 0.8}
    print("db ok")
