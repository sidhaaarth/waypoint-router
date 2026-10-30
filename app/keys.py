"""Router keys (rk-…, stored hashed) and encrypted bring-your-own Token Factory keys."""
import hashlib, os, secrets, time

from cryptography.fernet import Fernet
from fastapi import Cookie, Header, HTTPException

from . import config, db

_fernet = None


def fernet():
    global _fernet
    if _fernet is None:
        key = config.ROUTER_SECRET
        if not key:  # self-host convenience: persist a generated key next to the DB
            path = os.path.join(os.path.dirname(os.path.abspath(config.DB_PATH)), "router.secret")
            if not os.path.exists(path):
                with open(path, "wb") as f:
                    f.write(Fernet.generate_key())
            key = open(path, "rb").read()
        _fernet = Fernet(key)
    return _fernet


def hash_key(rk):
    return hashlib.sha256(rk.encode()).hexdigest()


def create_project(name, tf_key):
    """Returns (project_id, router_key). The router key is shown once; only its hash is stored."""
    pid = "proj_" + secrets.token_hex(6)
    rk = "rk-" + secrets.token_urlsafe(24)
    now = time.time()
    db.x("INSERT INTO projects(id, name, tf_key_enc, created_at) VALUES (?,?,?,?)",
         (pid, name, fernet().encrypt(tf_key.encode()), now))
    db.x("INSERT INTO router_keys(hash, project_id, prefix, created_at) VALUES (?,?,?,?)",
         (hash_key(rk), pid, rk[:7], now))
    return pid, rk


def project_for(rk):
    row = db.one("SELECT p.* FROM router_keys k JOIN projects p ON p.id = k.project_id WHERE k.hash = ?",
                 (hash_key(rk),))
    if not row:
        return None
    row["tf_key"] = fernet().decrypt(row.pop("tf_key_enc")).decode()
    return row


SESSION = "wp_session"  # HttpOnly login cookie holding the router key; set by POST /api/session


def auth(authorization: str = Header(default=""), wp_session: str = Cookie(default="")):
    """FastAPI dependency: Bearer rk-… (or the login cookie) -> project dict (with decrypted tf_key)."""
    rk = authorization.removeprefix("Bearer ").strip()
    if not rk.startswith("rk-"):
        rk = wp_session
    project = project_for(rk) if rk.startswith("rk-") else None
    if not project:
        raise HTTPException(401, {"error": {"message": "invalid router key", "type": "invalid_request_error"}})
    return project


if __name__ == "__main__":
    import tempfile
    config.DB_PATH = os.path.join(tempfile.mkdtemp(), "t.db")
    pid, rk = create_project("demo", "tf-secret")
    assert project_for(rk)["tf_key"] == "tf-secret" and project_for(rk)["id"] == pid
    assert project_for("rk-wrong") is None
    assert db.one("SELECT tf_key_enc FROM projects")["tf_key_enc"] != b"tf-secret"
    try:
        auth("Bearer nope", ""); raise AssertionError("should 401")
    except HTTPException as e:
        assert e.status_code == 401
    assert auth("", rk)["id"] == pid  # cookie alone signs in
    print("keys ok")
