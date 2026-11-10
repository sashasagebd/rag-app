"""Accounts: every SQL statement about users, invites, sessions and the question log.

"Today" is the current UTC day (Postgres' date_trunc('day', now()) in a UTC database).
"""

import re
from datetime import datetime

import psycopg
from psycopg.types.json import Jsonb

from api.security import hash_password, new_invite_code, new_token, sha256, verify_password

USERNAME_RE = re.compile(r"^[a-z0-9_][a-z0-9_-]{2,31}$")
MIN_PASSWORD = 8


class AccountError(Exception):
    """A problem the user can fix; the message is safe to show."""


# ---------- invites and sign-up ----------

def create_invite(conn: psycopg.Connection, note: str | None = None) -> str:
    code = new_invite_code()
    conn.execute("INSERT INTO invites (code_hash, note) VALUES (%s, %s)", (sha256(code), note))
    return code


def signup(conn: psycopg.Connection, code: str, username: str, password: str) -> int:
    username = username.strip().lower()
    if not USERNAME_RE.match(username):
        raise AccountError("usernames are 3-32 characters: lowercase letters, digits, _ and -")
    if len(password) < MIN_PASSWORD:
        raise AccountError(f"the password must be at least {MIN_PASSWORD} characters")
    with conn.transaction():
        invite = conn.execute("SELECT used_by FROM invites WHERE code_hash = %s FOR UPDATE",
                              (sha256(code.strip().lower()),)).fetchone()
        if invite is None or invite[0] is not None:
            raise AccountError("that invite code is not valid (or has already been used)")
        if conn.execute("SELECT 1 FROM users WHERE username = %s", (username,)).fetchone():
            raise AccountError("that username is taken")
        user_id = conn.execute("INSERT INTO users (username, password_hash) VALUES (%s, %s) RETURNING id",
                               (username, hash_password(password))).fetchone()[0]
        conn.execute("UPDATE invites SET used_by = %s, used_at = now() WHERE code_hash = %s",
                     (user_id, sha256(code.strip().lower())))
    return user_id


def authenticate(conn: psycopg.Connection, username: str, password: str) -> int | None:
    row = conn.execute("SELECT id, password_hash FROM users WHERE username = %s",
                       (username.strip().lower(),)).fetchone()
    if row is None:
        verify_password(password, "scrypt$AAAAAAAAAAAAAAAAAAAAAA==$AAAA")  # same work either way: no timing hint
        return None
    return row[0] if verify_password(password, row[1]) else None


# ---------- sessions ----------

def create_session(conn: psycopg.Connection, user_id: int, days: int) -> str:
    token = new_token()
    conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (%s, %s, now() + %s * interval '1 day')",
                 (sha256(token), user_id, days))
    return token


def user_for_session(conn: psycopg.Connection, token: str | None) -> int | None:
    if not token:
        return None
    row = conn.execute("SELECT user_id FROM sessions WHERE token_hash = %s AND expires_at > now()",
                       (sha256(token),)).fetchone()
    return row[0] if row else None


def delete_session(conn: psycopg.Connection, token: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token_hash = %s OR expires_at < now()", (sha256(token),))


# ---------- the user's settings ----------

def get_user(conn: psycopg.Connection, user_id: int) -> dict:
    row = conn.execute("SELECT id, username, api_key_enc IS NOT NULL, api_key_hint, use_server_key, "
                       "daily_question_limit, created_at FROM users WHERE id = %s", (user_id,)).fetchone()
    keys = ["id", "username", "has_api_key", "api_key_hint", "use_server_key", "daily_question_limit", "created_at"]
    return dict(zip(keys, row))


def get_api_key_enc(conn: psycopg.Connection, user_id: int) -> str | None:
    return conn.execute("SELECT api_key_enc FROM users WHERE id = %s", (user_id,)).fetchone()[0]


def set_api_key(conn: psycopg.Connection, user_id: int, encrypted: str | None, hint: str | None) -> None:
    conn.execute("UPDATE users SET api_key_enc = %s, api_key_hint = %s WHERE id = %s", (encrypted, hint, user_id))


# ---------- usage and the question log ----------

def questions_today(conn: psycopg.Connection, user_id: int) -> int:
    return conn.execute("SELECT count(*) FROM questions WHERE user_id = %s AND created_at >= date_trunc('day', now())",
                        (user_id,)).fetchone()[0]


def server_spend_today(conn: psycopg.Connection) -> float:
    return float(conn.execute("SELECT coalesce(sum(usd), 0) FROM questions WHERE key_source = 'server' "
                              "AND created_at >= date_trunc('day', now())").fetchone()[0])


def log_question(conn: psycopg.Connection, user_id: int, collection_id: int, question: str, response: dict,
                 model: str, key_source: str, usd: float) -> tuple[int, datetime]:
    return conn.execute(
        "INSERT INTO questions (user_id, collection_id, question, response, model, key_source, usd) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id, created_at",
        (user_id, collection_id, question, Jsonb(response), model, key_source, usd)).fetchone()


def history(conn: psycopg.Connection, user_id: int, collection_id: int, limit: int = 50) -> list[tuple]:
    """(id, question, response, key_source, usd, created_at), oldest first."""
    rows = conn.execute("SELECT id, question, response, key_source, usd, created_at FROM questions "
                        "WHERE user_id = %s AND collection_id = %s ORDER BY id DESC LIMIT %s",
                        (user_id, collection_id, limit)).fetchall()
    return rows[::-1]


# ---------- admin ----------

def list_users(conn: psycopg.Connection) -> list[tuple]:
    """(username, created_at, has_key, use_server_key, daily_question_limit, questions_today, usd_total)."""
    return conn.execute(
        "SELECT u.username, u.created_at, u.api_key_enc IS NOT NULL, u.use_server_key, u.daily_question_limit, "
        "count(q.id) FILTER (WHERE q.created_at >= date_trunc('day', now())), coalesce(sum(q.usd), 0) "
        "FROM users u LEFT JOIN questions q ON q.user_id = u.id GROUP BY u.id ORDER BY u.username").fetchall()


def update_user(conn: psycopg.Connection, username: str, **fields) -> None:
    allowed = {"use_server_key", "daily_question_limit"}
    assert set(fields) <= allowed, fields
    sets = ", ".join(f"{k} = %({k})s" for k in fields)
    if conn.execute(f"UPDATE users SET {sets} WHERE username = %(u)s", {**fields, "u": username}).rowcount == 0:
        raise AccountError(f"no user named '{username}'")
