"""The web API (FastAPI). Run it with:

    APP_SECRET_KEY=... uvicorn api.app:create_app --factory --port 8000

Settings: APP_CONFIG (default configs/app.toml). APP_SECRET_KEY (required) encrypts
users' Anthropic keys. Every route except signup/login/health needs a session cookie.
ADMIN_USERNAME + ADMIN_PASSWORD_HASH (optional, from `python -m api.admin export-login`)
create the first admin at startup; admins get /api/admin/* (invites, users), shown in Settings.

Notebooks: a user's notebook "ml-papers" is stored as collection "u<user id>--ml-papers",
and every route checks that the notebook belongs to the logged-in user.

Uploads: the fast checks run during the request; parsing and embedding run in the
background (status 'processing' -> 'ready' or 'failed'), and the UI polls the document list.

Asking: uses the user's own Anthropic key. Accounts the admin flagged (use_server_key)
may use the server's key, until [app] server_key_daily_usd is spent for the day.
Each user may ask [app] questions_per_day questions per day (CPU on the server).

If web/dist exists (the built React UI), it is served at /.
"""

import logging
import os
import shutil
import threading
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path

import anthropic
import psycopg
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg import register_vector
from pydantic import BaseModel

from api import accounts
from api.accounts import AccountError
from api.security import KeyBox, key_hint
from common.cache import Cache
from common.embedding import get_model
from common.config import load_config
from common.llm import LLMError, client, cost_usd
from ingest.embed_load import connect, database_url
from notebook import store
from notebook.ask import ask, cite_label
from notebook.fetch import FetchError, fetch
from notebook.ingest import token_counter, build, name_from_url, precheck, save_original, sniff_type
from notebook.store import NAME_RE, NotebookError, NotFoundError
from retrieval.search import get_reranker

log = logging.getLogger("api")
COOKIE = "session"
WEB_DIST = Path("web/dist")


class LimitError(Exception):
    """A usage limit was reached (HTTP 429)."""


class KeyMissing(Exception):
    """The user has no usable API key (HTTP 403)."""


# ---------- request bodies ----------

class SignupBody(BaseModel):
    invite: str
    username: str
    password: str


class LoginBody(BaseModel):
    username: str
    password: str


class ApiKeyBody(BaseModel):
    api_key: str


class NotebookBody(BaseModel):
    name: str


class UrlBody(BaseModel):
    url: str


class AskBody(BaseModel):
    question: str


class InviteBody(BaseModel):
    note: str | None = None


class UserPatch(BaseModel):
    use_server_key: bool | None = None
    daily_question_limit: int | None = None  # null = back to the default (when sent)


# ---------- helpers ----------

def validate_api_key(api_key: str, model: str) -> None:
    """Check a key with a free token-count call. Raises AccountError with a readable reason."""
    if not api_key.startswith("sk-ant-") or len(api_key) < 30:
        raise AccountError("that doesn't look like an Anthropic API key (they start with sk-ant-)")
    try:
        client(api_key).messages.count_tokens(model=model, messages=[{"role": "user", "content": "hi"}])
    except anthropic.AuthenticationError:
        raise AccountError("Anthropic rejected this key: check that you copied all of it")
    except anthropic.PermissionDeniedError:
        raise AccountError("this key isn't allowed to use the API")
    except anthropic.RateLimitError:
        pass  # the key works; it's just busy
    except anthropic.APIConnectionError:
        raise AccountError("couldn't reach Anthropic to check the key; try again")


def llm_http_error(e: Exception) -> HTTPException:
    """Turn an Anthropic error during an answer into a message for the user."""
    if isinstance(e, anthropic.AuthenticationError):
        return HTTPException(400, "Anthropic rejected your API key. Update it in Settings.")
    if isinstance(e, anthropic.RateLimitError):
        return HTTPException(429, "Anthropic is rate-limiting your key. Try again in a minute.")
    if isinstance(e, anthropic.BadRequestError) and "credit balance" in str(e).lower():
        return HTTPException(402, "Your Anthropic account is out of credit. Add credit at console.anthropic.com.")
    if isinstance(e, anthropic.APIConnectionError):
        return HTTPException(502, "Couldn't reach Anthropic. Try again.")
    if isinstance(e, LLMError):
        return HTTPException(502, f"The model couldn't answer: {e}")
    return HTTPException(502, "The answer service failed. Try again.")


def internal_name(user_id: int, name: str) -> str:
    return f"u{user_id}--{name}"


def read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload without ever holding more than max_bytes + 1 MB in memory."""
    data = bytearray()
    while piece := upload.file.read(1024 * 1024):
        data += piece
        if len(data) > max_bytes:
            raise NotebookError(f"the file is larger than the {max_bytes // 1_000_000} MB limit")
    return bytes(data)


class LoginThrottle:
    """At most `limit` failed logins per username per `window` seconds (in memory, per process)."""

    def __init__(self, limit: int = 10, window: int = 900):
        self.limit, self.window, self.failures = limit, window, defaultdict(deque)

    def check(self, username: str) -> None:
        q = self.failures[username]
        while q and q[0] < time.monotonic() - self.window:
            q.popleft()
        if len(q) >= self.limit:
            raise LimitError("too many failed logins for this username; try again in 15 minutes")

    def failed(self, username: str) -> None:
        self.failures[username].append(time.monotonic())


# ---------- the app ----------

def create_app(config: dict | None = None, *, ingest_kwargs: dict | None = None) -> FastAPI:
    """config: defaults to load_config($APP_CONFIG or configs/app.toml).
    ingest_kwargs: passed to build() (tests use it to fake the tokenizer and embeddings)."""
    config = config or load_config(os.environ.get("APP_CONFIG", "configs/app.toml"))
    settings, limits = config["app"], config["notebook"]
    keybox = KeyBox(os.environ.get("APP_SECRET_KEY"))
    url = database_url(config)
    cache = Cache(config["paths"]["cache_db"])
    throttle = LoginThrottle()
    warm_up = not ingest_kwargs  # tests pass fakes and need no models
    ingest_kwargs = ingest_kwargs or {}

    def load_models() -> None:
        """Load the tokenizer, embedder and reranker once, right after startup, so the first
        request doesn't wait for them. In a thread: the health check answers meanwhile."""
        started = time.monotonic()
        try:
            token_counter(config["embedding"]["model"])
            get_model(config["embedding"]["model"])
            if config["retrieval"].get("rerank_model"):
                get_reranker(config["retrieval"]["rerank_model"])
            log.warning("models loaded in %.1fs", time.monotonic() - started)
        except Exception:
            log.exception("loading models failed; they will load on first use")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        with connect(url) as conn:  # applies db/schema.sql
            failed = store.fail_interrupted(conn)
            if failed:
                log.warning("marked %d interrupted uploads as failed", failed)
            admin, admin_hash = os.environ.get("ADMIN_USERNAME"), os.environ.get("ADMIN_PASSWORD_HASH")
            if admin and admin_hash:
                log.warning("admin account %s: %s", admin, accounts.ensure_admin(conn, admin, admin_hash))
        if warm_up:
            threading.Thread(target=load_models, daemon=True, name="warm-up").start()
        yield

    app = FastAPI(title="rag-app", lifespan=lifespan)

    # ----- errors -> HTTP status + {"detail": message}
    for exc, status in [(NotFoundError, 404), (NotebookError, 400), (FetchError, 400), (AccountError, 400),
                        (LimitError, 429), (KeyMissing, 403)]:
        app.add_exception_handler(exc, lambda request, e, status=status: JSONResponse({"detail": str(e)}, status))

    # ----- dependencies
    def get_conn():
        conn = psycopg.connect(url, autocommit=True)
        register_vector(conn)
        try:
            yield conn
        finally:
            conn.close()

    def current_user(request: Request, conn=Depends(get_conn)) -> int:
        user_id = accounts.user_for_session(conn, request.cookies.get(COOKIE))
        if user_id is None:
            raise HTTPException(401, "please log in")
        return user_id

    def current_admin(user_id=Depends(current_user), conn=Depends(get_conn)) -> int:
        if not accounts.get_user(conn, user_id)["is_admin"]:
            raise HTTPException(403, "admins only")
        return user_id

    def owned_notebook(name: str, user_id: int, conn) -> tuple[str, int]:
        full = internal_name(user_id, name)
        return full, store.get_notebook(conn, full, user_id)

    def set_session(response: Response, conn, user_id: int) -> None:
        token = accounts.create_session(conn, user_id, settings["session_days"])
        response.set_cookie(COOKIE, token, httponly=True, samesite="lax", secure=settings["secure_cookies"],
                            max_age=settings["session_days"] * 86400, path="/")

    def me(conn, user_id: int) -> dict:
        u = accounts.get_user(conn, user_id)
        return {
            "username": u["username"], "has_api_key": u["has_api_key"], "api_key_hint": u["api_key_hint"],
            "uses_server_key": u["use_server_key"], "is_admin": u["is_admin"],
            "usage": {
                "questions_today": accounts.questions_today(conn, user_id),
                "questions_per_day": u["daily_question_limit"] or settings["questions_per_day"],
                "notebooks": store.count_notebooks(conn, user_id), "max_notebooks": settings["max_notebooks"],
                "upload_mb": round(store.user_upload_bytes(conn, user_id) / 1e6, 1),
                "max_upload_mb": settings["max_upload_mb_total"],
            },
        }

    # ----- auth
    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/auth/signup", status_code=201)
    def signup(body: SignupBody, response: Response, conn=Depends(get_conn)):
        user_id = accounts.signup(conn, body.invite, body.username, body.password)
        set_session(response, conn, user_id)
        return me(conn, user_id)

    @app.post("/api/auth/login")
    def login(body: LoginBody, response: Response, conn=Depends(get_conn)):
        username = body.username.strip().lower()
        throttle.check(username)
        user_id = accounts.authenticate(conn, username, body.password)
        if user_id is None:
            throttle.failed(username)
            raise HTTPException(401, "wrong username or password")
        set_session(response, conn, user_id)
        return me(conn, user_id)

    @app.post("/api/auth/logout", status_code=204)
    def logout(request: Request, response: Response, conn=Depends(get_conn)):
        if token := request.cookies.get(COOKIE):
            accounts.delete_session(conn, token)
        response.delete_cookie(COOKIE, path="/")

    @app.get("/api/me")
    def get_me(user_id=Depends(current_user), conn=Depends(get_conn)):
        return me(conn, user_id)

    @app.put("/api/me/api-key")
    def put_api_key(body: ApiKeyBody, user_id=Depends(current_user), conn=Depends(get_conn)):
        key = body.api_key.strip()
        validate_api_key(key, settings["key_check_model"])
        accounts.set_api_key(conn, user_id, keybox.encrypt(key), key_hint(key))
        return me(conn, user_id)

    @app.delete("/api/me/api-key")
    def delete_api_key(user_id=Depends(current_user), conn=Depends(get_conn)):
        accounts.set_api_key(conn, user_id, None, None)
        return me(conn, user_id)

    # ----- notebooks
    @app.get("/api/notebooks")
    def list_notebooks(user_id=Depends(current_user), conn=Depends(get_conn)):
        prefix = internal_name(user_id, "")
        return [{"name": name.removeprefix(prefix), "documents": n_docs, "tokens": tokens, "created_at": created}
                for name, n_docs, tokens, created in store.list_notebooks(conn, user_id)]

    @app.post("/api/notebooks", status_code=201)
    def create_notebook(body: NotebookBody, user_id=Depends(current_user), conn=Depends(get_conn)):
        name = body.name.strip()
        if not NAME_RE.match(name) or len(name) > 50:
            raise NotebookError("notebook names use lowercase letters, digits and dashes (max 50), e.g. ml-papers")
        if store.count_notebooks(conn, user_id) >= settings["max_notebooks"]:
            raise LimitError(f"you have {settings['max_notebooks']} notebooks, the maximum; delete one first")
        store.create_notebook(conn, internal_name(user_id, name), user_id)
        return {"name": name}

    @app.delete("/api/notebooks/{name}", status_code=204)
    def delete_notebook(name: str, user_id=Depends(current_user), conn=Depends(get_conn)):
        full, notebook_id = owned_notebook(name, user_id, conn)
        store.delete_notebook(conn, notebook_id)
        shutil.rmtree(Path(limits["storage_dir"]) / full, ignore_errors=True)

    # ----- documents
    @app.get("/api/notebooks/{name}/documents")
    def list_documents(name: str, user_id=Depends(current_user), conn=Depends(get_conn)):
        _, notebook_id = owned_notebook(name, user_id, conn)
        keys = ["name", "source", "doc_type", "n_bytes", "n_chunks", "n_tokens", "added_at", "status", "error"]
        return [dict(zip(keys, row)) for row in store.list_documents(conn, notebook_id)]

    def process_document(full: str, notebook_id: int, meta: dict, data: bytes) -> None:
        """Background task: parse, chunk, embed, store. Opens its own connection."""
        with psycopg.connect(url, autocommit=True) as conn:
            register_vector(conn)
            try:
                chunks, embeddings = build(meta, data, config, cache, **ingest_kwargs)
                store.add_document(conn, notebook_id, meta, chunks, embeddings)
                if limits.get("keep_originals", True):  # off in production: no persistent disk there
                    save_original(limits, full, meta["name"], data)
            except NotebookError as e:
                store.mark_failed(conn, notebook_id, meta["name"], str(e))
            except Exception:
                log.exception("processing %s in %s failed", meta["name"], full)
                store.mark_failed(conn, notebook_id, meta["name"], "processing failed unexpectedly")

    def accept_document(conn, user_id: int, full: str, notebook_id: int, data: bytes, name: str, source: str,
                        background: BackgroundTasks) -> dict:
        used = store.user_upload_bytes(conn, user_id)
        if used + len(data) > settings["max_upload_mb_total"] * 1_000_000:
            raise LimitError(f"this would take you past your {settings['max_upload_mb_total']} MB storage limit "
                             f"({used / 1e6:.1f} MB used); remove some documents first")
        meta = precheck(conn, notebook_id, data, name, source, limits)
        if meta["skipped"]:
            return {"name": meta["name"], "status": "skipped", "reason": meta["skipped"]}
        store.mark_processing(conn, notebook_id, meta)
        background.add_task(process_document, full, notebook_id, meta, data)
        return {"name": meta["name"], "status": "processing"}

    @app.post("/api/notebooks/{name}/documents", status_code=202)
    def upload_document(name: str, background: BackgroundTasks, file: UploadFile = File(...),
                        user_id=Depends(current_user), conn=Depends(get_conn)):
        full, notebook_id = owned_notebook(name, user_id, conn)
        data = read_limited(file, limits["max_file_mb"] * 1_000_000)
        filename = file.filename or "document"
        return accept_document(conn, user_id, full, notebook_id, data, filename, filename, background)

    @app.post("/api/notebooks/{name}/documents/url", status_code=202)
    def add_url(name: str, body: UrlBody, background: BackgroundTasks,
                user_id=Depends(current_user), conn=Depends(get_conn)):
        full, notebook_id = owned_notebook(name, user_id, conn)
        data, final_url, _ = fetch(body.url.strip(), max_bytes=limits["max_file_mb"] * 1_000_000,
                                   timeout=limits["url_timeout_s"], max_redirects=limits["max_redirects"])
        doc_name = name_from_url(final_url, sniff_type(data) or "html")
        return accept_document(conn, user_id, full, notebook_id, data, doc_name, body.url.strip(), background)

    @app.delete("/api/notebooks/{name}/documents/{document}", status_code=204)
    def delete_document(name: str, document: str, user_id=Depends(current_user), conn=Depends(get_conn)):
        full, notebook_id = owned_notebook(name, user_id, conn)
        store.remove_document(conn, notebook_id, document)
        (Path(limits["storage_dir"]) / full / document).unlink(missing_ok=True)

    # ----- asking
    def resolve_key(conn, user_id: int) -> tuple[str | None, str]:
        """(api_key, key_source). None = the server's key."""
        encrypted = accounts.get_api_key_enc(conn, user_id)
        if encrypted and (key := keybox.decrypt(encrypted)):
            return key, "user"
        if accounts.get_user(conn, user_id)["use_server_key"]:
            if accounts.server_spend_today(conn) >= settings["server_key_daily_usd"]:
                raise LimitError("the shared key has reached today's spending cap; add your own key in Settings")
            return None, "server"
        raise KeyMissing("add your Anthropic API key in Settings to ask questions")

    def question_json(row) -> dict:
        qid, question, response, key_source, usd, created = row
        return {"id": qid, "question": question, **response, "key_source": key_source,
                "usd": float(usd), "created_at": created}

    @app.post("/api/notebooks/{name}/ask")
    def ask_question(name: str, body: AskBody, user_id=Depends(current_user), conn=Depends(get_conn)):
        full, notebook_id = owned_notebook(name, user_id, conn)
        question = body.question.strip()
        if not question or len(question) > 2000:
            raise NotebookError("questions must be 1 to 2,000 characters")
        allowed = accounts.get_user(conn, user_id)["daily_question_limit"] or settings["questions_per_day"]
        asked = accounts.questions_today(conn, user_id)
        if asked >= allowed:
            raise LimitError(f"you've asked {asked} questions today, the daily limit; it resets at midnight UTC")
        api_key, key_source = resolve_key(conn, user_id)

        try:
            r = ask(conn, full, question, config, cache, notebook_id=notebook_id, api_key=api_key)
        except (anthropic.APIError, LLMError) as e:
            raise llm_http_error(e)

        model = config["answer"]["model"]
        usd = 0.0 if r["cached"] else cost_usd(r["usage"], model, config["pricing"])
        response = {
            "answer": r["answer"]["answer"], "found": r["answer"]["found"], "route": r["route"],
            "citations": [{"n": i, "chunk_key": c["chunk_key"], "doc": c["doc"], "label": cite_label(c),
                           "sections": c["sections"], "pages": c.get("pages") or [], "text": c["text"]}
                          for i, c in enumerate(r["cited_chunks"], 1)],
        }
        source = "cache" if r["cached"] else key_source
        qid, created = accounts.log_question(conn, user_id, notebook_id, question, response, model, source, usd)
        return question_json((qid, question, response, source, usd, created))

    @app.get("/api/notebooks/{name}/questions")
    def list_questions(name: str, user_id=Depends(current_user), conn=Depends(get_conn)):
        _, notebook_id = owned_notebook(name, user_id, conn)
        return [question_json(row) for row in accounts.history(conn, user_id, notebook_id)]

    # ----- admin (the admin section in Settings)
    @app.get("/api/admin/users")
    def admin_users(_=Depends(current_admin), conn=Depends(get_conn)):
        keys = ["username", "created_at", "has_api_key", "use_server_key", "daily_question_limit",
                "questions_today", "usd_total", "is_admin"]
        return [{**dict(zip(keys, row)), "usd_total": float(row[6])} for row in accounts.list_users(conn)]

    @app.post("/api/admin/invites", status_code=201)
    def admin_invite(body: InviteBody, _=Depends(current_admin), conn=Depends(get_conn)):
        return {"code": accounts.create_invite(conn, (body.note or "").strip() or None)}

    @app.patch("/api/admin/users/{username}")
    def admin_update_user(username: str, body: UserPatch, _=Depends(current_admin), conn=Depends(get_conn)):
        fields = {k: getattr(body, k) for k in body.model_fields_set}
        if fields.get("daily_question_limit") is not None and fields["daily_question_limit"] < 0:
            raise AccountError("the limit can't be negative")
        if fields.get("use_server_key") is None:
            fields.pop("use_server_key", None)
        if fields:
            accounts.update_user(conn, username, **fields)
        return admin_users(None, conn)

    # ----- the built UI, if present
    if WEB_DIST.is_dir():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")

    return app
