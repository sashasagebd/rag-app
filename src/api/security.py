"""Passwords, session tokens, invite codes and API-key encryption. Standard library + `cryptography`.

- Passwords: scrypt (memory-hard), random 16-byte salt, stored as "scrypt$<salt>$<hash>".
- Session tokens and invite codes: random; the database stores only their SHA-256, so a
  leaked database can't be used to log in or sign up.
- Users' Anthropic API keys: encrypted with Fernet (AES + HMAC) under a key derived from
  the APP_SECRET_KEY environment variable. Without that secret the stored keys are
  unreadable. If APP_SECRET_KEY is lost or changed, users simply re-enter their keys.
"""

import base64
import hashlib
import hmac
import os
import secrets

from cryptography.fernet import Fernet, InvalidToken

SCRYPT = {"n": 2**14, "r": 8, "p": 1, "dklen": 32}
INVITE_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o, 1/l/i look-alikes


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode()


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, **SCRYPT)
    return f"scrypt${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt, digest = stored.split("$")
    except ValueError:
        return False
    if scheme != "scrypt":
        return False
    candidate = hashlib.scrypt(password.encode(), salt=base64.urlsafe_b64decode(salt), **SCRYPT)
    return hmac.compare_digest(candidate, base64.urlsafe_b64decode(digest))


def new_token() -> str:
    return secrets.token_urlsafe(32)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def new_invite_code() -> str:
    """e.g. 'k7mq-2xpa-9rht': 12 random characters, about 59 bits."""
    chars = "".join(secrets.choice(INVITE_ALPHABET) for _ in range(12))
    return f"{chars[:4]}-{chars[4:8]}-{chars[8:]}"


def key_hint(api_key: str) -> str:
    return f"{api_key[:7]}…{api_key[-4:]}"


class KeyBox:
    """Encrypts and decrypts users' API keys with a key derived from APP_SECRET_KEY."""

    def __init__(self, secret: str | None):
        if not secret or len(secret) < 32:
            raise RuntimeError("APP_SECRET_KEY must be set to a random string of at least 32 characters "
                               "(e.g. python -c \"import secrets; print(secrets.token_urlsafe(48))\")")
        self._fernet = Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret.encode()).digest()))

    def encrypt(self, text: str) -> str:
        return self._fernet.encrypt(text.encode()).decode()

    def decrypt(self, token: str) -> str | None:
        """None if the token can't be decrypted (APP_SECRET_KEY changed): treat as 'no key'."""
        try:
            return self._fernet.decrypt(token.encode()).decode()
        except InvalidToken:
            return None
