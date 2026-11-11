"""Web API tests against the docker Postgres (skipped if it isn't up). No network, no models:
the tokenizer, embeddings, the API-key check and the answering model are faked."""

import numpy as np
import pymupdf
import pytest

psycopg = pytest.importorskip("psycopg")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import api.app as api_app  # noqa: E402
import notebook.ask as ask_module  # noqa: E402
from api import accounts  # noqa: E402
from common.config import load_config  # noqa: E402
from ingest.embed_load import connect, database_url  # noqa: E402

KEY = "sk-ant-api03-" + "x" * 40
HTML = b"<html><body><h1>Grading</h1><p>" + b"Homework is worth forty percent of the grade. " * 5 + b"</p></body></html>"
FAKES = {"count_tokens": lambda t: len(t.split()), "embed_fn": lambda texts: np.ones((len(texts), 384), np.float32)}


@pytest.fixture
def env(tmp_path, monkeypatch):
    config = load_config("configs/app-dev.toml")
    try:
        conn = connect(database_url(config))
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; run `docker compose up -d --wait`")
    config["notebook"]["storage_dir"] = str(tmp_path / "uploads")
    config["paths"]["cache_db"] = str(tmp_path / "cache.sqlite")
    config["app"].update(questions_per_day=2, max_notebooks=2, server_key_daily_usd=0.01)
    monkeypatch.setenv("APP_SECRET_KEY", "test-secret-" + "s" * 40)
    monkeypatch.setattr(api_app, "validate_api_key", lambda key, model: None)
    calls = []

    def fake_answer(question, chunks, **kw):
        calls.append(kw.get("api_key"))
        return {"data": {"found": True, "answer": "Forty percent.", "citations": [chunks[0]["chunk_key"]]},
                "usage": {"input_tokens": 1000, "output_tokens": 100}, "cached": False}

    monkeypatch.setattr(ask_module, "answer_question", fake_answer)

    def cleanup():
        conn.execute("DELETE FROM users WHERE username LIKE 'pytest%'")  # cascades to notebooks, questions
        conn.execute("DELETE FROM invites WHERE note = 'pytest'")

    cleanup()
    app = api_app.create_app(config, ingest_kwargs=FAKES)
    with TestClient(app) as a, TestClient(app) as b:
        yield {"conn": conn, "a": a, "b": b, "calls": calls}
    cleanup()
    conn.close()


def signup(env, client, username):
    code = accounts.create_invite(env["conn"], "pytest")
    r = client.post("/api/auth/signup", json={"invite": code, "username": username, "password": "correct horse 1"})
    assert r.status_code == 201, r.text
    return code


def test_signup_login_logout(env):
    a = env["a"]
    assert a.post("/api/auth/signup", json={"invite": "nope-nope-nope", "username": "pytest_a",
                                            "password": "correct horse 1"}).status_code == 400
    code = signup(env, a, "pytest_a")
    assert a.get("/api/me").json()["username"] == "pytest_a"
    reuse = env["b"].post("/api/auth/signup", json={"invite": code, "username": "pytest_b", "password": "x" * 12})
    assert reuse.status_code == 400 and "already been used" in reuse.json()["detail"]

    a.post("/api/auth/logout")
    assert a.get("/api/me").status_code == 401
    assert a.post("/api/auth/login", json={"username": "pytest_a", "password": "wrong password"}).status_code == 401
    assert a.post("/api/auth/login", json={"username": "pytest_a", "password": "correct horse 1"}).status_code == 200
    assert a.get("/api/me").status_code == 200


def test_short_password_and_bad_username_are_rejected(env):
    code = accounts.create_invite(env["conn"], "pytest")
    r = env["a"].post("/api/auth/signup", json={"invite": code, "username": "pytest_a", "password": "short"})
    assert r.status_code == 400 and "at least 8" in r.json()["detail"]
    r = env["a"].post("/api/auth/signup", json={"invite": code, "username": "Py Test!", "password": "x" * 12})
    assert r.status_code == 400 and "usernames" in r.json()["detail"]


def test_notebooks_are_private_and_limited(env):
    a, b = env["a"], env["b"]
    signup(env, a, "pytest_a")
    signup(env, b, "pytest_b")
    assert a.post("/api/notebooks", json={"name": "papers"}).status_code == 201
    assert b.get("/api/notebooks/papers/documents").status_code == 404  # A's notebook is invisible to B
    assert b.post("/api/notebooks", json={"name": "papers"}).status_code == 201  # same name, own notebook
    assert [n["name"] for n in a.get("/api/notebooks").json()] == ["papers"]
    assert a.post("/api/notebooks", json={"name": "two"}).status_code == 201
    r = a.post("/api/notebooks", json={"name": "three"})
    assert r.status_code == 429 and "maximum" in r.json()["detail"]


def test_upload_processes_in_background_with_status(env):
    a = env["a"]
    signup(env, a, "pytest_a")
    a.post("/api/notebooks", json={"name": "nb"})

    r = a.post("/api/notebooks/nb/documents", files={"file": ("grading.html", HTML, "text/html")})
    assert r.status_code == 202 and r.json() == {"name": "grading.html", "status": "processing"}
    [doc] = a.get("/api/notebooks/nb/documents").json()  # TestClient runs background tasks before returning
    assert doc["status"] == "ready" and doc["n_chunks"] >= 1

    again = a.post("/api/notebooks/nb/documents", files={"file": ("copy.html", HTML, "text/html")}).json()
    assert again["status"] == "skipped"

    scan = pymupdf.open()
    scan.new_page()
    a.post("/api/notebooks/nb/documents", files={"file": ("scan.pdf", scan.tobytes(), "application/pdf")})
    failed = [d for d in a.get("/api/notebooks/nb/documents").json() if d["name"] == "scan.pdf"][0]
    assert failed["status"] == "failed" and "OCR" in failed["error"]
    retry = a.post("/api/notebooks/nb/documents", files={"file": ("scan.pdf", scan.tobytes(), "application/pdf")})
    assert retry.json()["status"] == "processing"  # a failed upload is not a "duplicate"

    r = a.post("/api/notebooks/nb/documents", files={"file": ("notes.txt", b"plain text", "text/plain")})
    assert r.status_code == 400 and "only PDF and HTML" in r.json()["detail"]

    assert a.delete("/api/notebooks/nb/documents/grading.html").status_code == 204
    assert a.delete("/api/notebooks/nb/documents/grading.html").status_code == 404


def test_ask_needs_a_key_then_answers_with_citations_and_limits(env):
    a, conn = env["a"], env["conn"]
    signup(env, a, "pytest_a")
    a.post("/api/notebooks", json={"name": "nb"})
    a.post("/api/notebooks/nb/documents", files={"file": ("grading.html", HTML, "text/html")})

    r = a.post("/api/notebooks/nb/ask", json={"question": "How much is homework worth?"})
    assert r.status_code == 403 and "API key" in r.json()["detail"]

    me = a.put("/api/me/api-key", json={"api_key": KEY}).json()
    assert me["has_api_key"] and me["api_key_hint"] == KEY[:7] + "…" + KEY[-4:]
    stored = conn.execute("SELECT api_key_enc FROM users WHERE username = 'pytest_a'").fetchone()[0]
    assert KEY not in stored  # encrypted at rest
    assert KEY not in a.get("/api/me").text  # never sent back

    r = a.post("/api/notebooks/nb/ask", json={"question": "How much is homework worth?"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["answer"] == "Forty percent." and body["key_source"] == "user"
    assert body["citations"][0]["label"].startswith("grading.html") and "forty percent" in body["citations"][0]["text"]
    assert env["calls"] == [KEY]  # the user's own key was used
    assert len(a.get("/api/notebooks/nb/questions").json()) == 1

    a.post("/api/notebooks/nb/ask", json={"question": "Again?"})
    r = a.post("/api/notebooks/nb/ask", json={"question": "Third?"})
    assert r.status_code == 429 and "daily limit" in r.json()["detail"]


def test_server_key_accounts_and_the_spend_cap(env, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-server-" + "k" * 30)  # never called: answers are faked
    a, conn = env["a"], env["conn"]
    signup(env, a, "pytest_a")
    accounts.update_user(conn, "pytest_a", use_server_key=True)
    a.post("/api/notebooks", json={"name": "nb"})
    a.post("/api/notebooks/nb/documents", files={"file": ("grading.html", HTML, "text/html")})

    r = a.post("/api/notebooks/nb/ask", json={"question": "How much is homework worth?"})
    assert r.status_code == 200 and r.json()["key_source"] == "server"
    assert env["calls"] == [None]  # None = the server's key
    # Haiku pricing: 1000 in + 100 out = $0.00015; the test cap is $0.01. Push spend over it:
    conn.execute("UPDATE questions SET usd = 1 WHERE key_source = 'server' AND user_id = "
                 "(SELECT id FROM users WHERE username = 'pytest_a')")
    r = a.post("/api/notebooks/nb/ask", json={"question": "Again?"})
    assert r.status_code == 429 and "spending cap" in r.json()["detail"]


def test_admin_section_is_admin_only(env):
    a, b, conn = env["a"], env["b"], env["conn"]
    signup(env, a, "pytest_admin")
    signup(env, b, "pytest_b")
    assert b.get("/api/admin/users").status_code == 403
    assert b.post("/api/admin/invites", json={}).status_code == 403

    accounts.update_user(conn, "pytest_admin", is_admin=True)
    assert a.get("/api/me").json()["is_admin"] is True
    code = a.post("/api/admin/invites", json={"note": "pytest"}).json()["code"]
    with TestClient(b.app) as c:  # the code works for a new sign-up
        r = c.post("/api/auth/signup", json={"invite": code, "username": "pytest_c", "password": "x" * 12})
        assert r.status_code == 201

    users = a.patch("/api/admin/users/pytest_b", json={"use_server_key": True, "daily_question_limit": 5}).json()
    row = [u for u in users if u["username"] == "pytest_b"][0]
    assert row["use_server_key"] is True and row["daily_question_limit"] == 5
    users = a.patch("/api/admin/users/pytest_b", json={"daily_question_limit": None}).json()
    assert [u for u in users if u["username"] == "pytest_b"][0]["daily_question_limit"] is None
    assert a.patch("/api/admin/users/nobody", json={"use_server_key": True}).status_code == 400


def test_admin_account_is_created_from_env_at_startup(env, monkeypatch):
    from api.security import hash_password

    monkeypatch.setenv("ADMIN_USERNAME", "pytest_boss")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", hash_password("the boss password"))
    with TestClient(api_app.create_app(load_config("configs/app-dev.toml"), ingest_kwargs=FAKES)) as c:
        r = c.post("/api/auth/login", json={"username": "pytest_boss", "password": "the boss password"})
        assert r.status_code == 200 and r.json()["is_admin"] and r.json()["uses_server_key"]


def test_server_key_access_without_a_server_key_asks_for_your_own(env, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    a, conn = env["a"], env["conn"]
    signup(env, a, "pytest_a")
    accounts.update_user(conn, "pytest_a", use_server_key=True)
    assert a.get("/api/me").json()["uses_server_key"] is False  # the UI then asks for a key
    a.post("/api/notebooks", json={"name": "nb"})
    a.post("/api/notebooks/nb/documents", files={"file": ("grading.html", HTML, "text/html")})
    r = a.post("/api/notebooks/nb/ask", json={"question": "How much is homework worth?"})
    assert r.status_code == 403 and "add your Anthropic API key" in r.json()["detail"]


def test_security_headers_on_every_response(env):
    for path in ("/api/health", "/api/me", "/no-such-page"):
        h = env["a"].get(path).headers
        assert h["x-frame-options"] == "DENY" and h["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in h["content-security-policy"]
        assert "strict-transport-security" not in h  # dev config: plain http, no HSTS


def test_api_docs_only_when_enabled(env):
    config = load_config("configs/app-dev.toml")
    assert env["a"].get("/openapi.json").status_code == 200  # dev: on
    config["app"].update(api_docs=False, secure_cookies=True)  # production settings
    with TestClient(api_app.create_app(config, ingest_kwargs=FAKES)) as c:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert c.get(path).status_code == 404, path
        assert c.get("/api/health").headers["strict-transport-security"] == "max-age=31536000"


def test_a_bad_admin_env_var_does_not_stop_startup(env, monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "pytest_boss")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", "ADMIN_PASSWORD_HASH=scrypt$pasted$wrong")  # a typical paste mistake
    with TestClient(api_app.create_app(load_config("configs/app-dev.toml"), ingest_kwargs=FAKES)) as c:
        assert c.get("/api/health").status_code == 200


def test_an_unreachable_database_fails_with_a_clear_error():
    with pytest.raises(RuntimeError, match="never became reachable"):
        api_app.open_database("postgresql://nobody:x@127.0.0.1:1/nothing", timeout=1)
