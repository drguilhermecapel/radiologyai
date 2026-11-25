"""Usuários e sessões: criação, autenticação, revogação.

Não existe usuário nem senha padrão. O primeiro administrador é criado pela
CLI (``radiologyai users create --role admin``); depois, só um administrador
cria usuários pela API. Toda operação é auditada (sem a senha, obviamente).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from radiologyai.errors import RadiologyAIError
from radiologyai.persistence.security import (
    Role,
    check_password_policy,
    hash_password,
    hash_token,
    new_token,
    verify_password,
)

if TYPE_CHECKING:
    from datetime import datetime

    from radiologyai.persistence.db import Database
    from radiologyai.persistence.models import User

__all__ = [
    "AuthenticationError",
    "LoginThrottledError",
    "UserExistsError",
    "authenticate",
    "create_user",
    "issue_session",
    "revoke_session",
    "revoke_user_sessions",
]


class AuthenticationError(RadiologyAIError):
    """Credenciais inválidas (mensagem genérica: não revela se o usuário existe)."""


class LoginThrottledError(AuthenticationError):
    """Tentativas demais para o mesmo usuário em pouco tempo."""


class UserExistsError(RadiologyAIError):
    """Nome de usuário já cadastrado."""


@dataclass
class _Failures:
    count: int
    first: float


_THROTTLE_LOCK = threading.Lock()
_FAILURES: dict[str, _Failures] = {}
MAX_FAILURES = 5
WINDOW_SECONDS = 15 * 60


def _throttled(username: str) -> bool:
    with _THROTTLE_LOCK:
        entry = _FAILURES.get(username)
        if entry is None:
            return False
        if time.monotonic() - entry.first > WINDOW_SECONDS:
            _FAILURES.pop(username, None)
            return False
        return entry.count >= MAX_FAILURES


def _record_failure(username: str) -> None:
    with _THROTTLE_LOCK:
        entry = _FAILURES.get(username)
        now = time.monotonic()
        if entry is None or now - entry.first > WINDOW_SECONDS:
            _FAILURES[username] = _Failures(1, now)
        else:
            entry.count += 1


def reset_throttle() -> None:
    """Limpa o contador de falhas (testes e reinício administrativo)."""
    with _THROTTLE_LOCK:
        _FAILURES.clear()


def create_user(
    db: Database,
    *,
    username: str,
    full_name: str,
    role: str,
    password: str,
    professional_id: str | None = None,
    min_length: int = 12,
    actor_id: str | None = None,
    actor_username: str | None = None,
) -> User:
    """Cria usuário com senha argon2id; audita ``user.created``.

    Raises:
        UserExistsError: nome já usado.
        WeakPasswordError: senha fora da política.
        ValueError: papel desconhecido.
    """
    from sqlalchemy import select

    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import User

    Role(role)
    check_password_policy(password, min_length)
    with db.session() as session:
        if session.execute(select(User).where(User.username == username)).scalar_one_or_none():
            raise UserExistsError(f"usuário {username!r} já existe")
        user = User(
            username=username,
            full_name=full_name,
            role=role,
            password_hash=hash_password(password),
            professional_id=professional_id,
            active=True,
        )
        session.add(user)
    append_event(
        db,
        "user.created",
        actor_id=actor_id,
        actor_username=actor_username or "cli",
        subject_type="user",
        subject_id=user.id,
        payload={"username": username, "role": role},
    )
    return user


def authenticate(db: Database, username: str, password: str) -> User:
    """Confere credenciais; sempre executa o hash (tempo uniforme).

    Raises:
        LoginThrottledError: falhas demais recentes para o usuário.
        AuthenticationError: credenciais inválidas ou usuário inativo.
    """
    from sqlalchemy import select

    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import User, utcnow

    if _throttled(username):
        raise LoginThrottledError("muitas tentativas; aguarde 15 minutos")
    with db.session() as session:
        user = session.execute(select(User).where(User.username == username)).scalar_one_or_none()
        # Hash fictício só para igualar o tempo de resposta quando o usuário não existe.
        stored = user.password_hash if user is not None else _DUMMY_HASH.get()
        ok = verify_password(stored, password) and user is not None and user.active
        if ok and user is not None:
            user.last_login_at = utcnow()
    if not ok or user is None:
        _record_failure(username)
        append_event(
            db, "auth.login_failed", actor_username=username[:64], payload={"reason": "credenciais"}
        )
        raise AuthenticationError("usuário ou senha inválidos")
    return user


class _LazyHash:
    def __init__(self) -> None:
        self._value: str | None = None

    def get(self) -> str:
        if self._value is None:
            self._value = hash_password("radiologyai-hash-de-tempo-uniforme")
        return self._value


_DUMMY_HASH = _LazyHash()


def issue_session(db: Database, user: User, ttl_hours: float) -> tuple[str, datetime]:
    """Cria sessão; devolve o token (uma única vez) e a expiração. O banco guarda só o hash."""
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import AuthSession, utcnow

    token = new_token()
    expires = utcnow() + timedelta(hours=ttl_hours)
    with db.session() as session:
        row = AuthSession(user_id=user.id, token_sha256=hash_token(token), expires_at=expires)
        session.add(row)
    append_event(
        db,
        "auth.login",
        actor_id=user.id,
        actor_username=user.username,
        subject_type="session",
        subject_id=row.id,
        payload={"expires_at": expires.isoformat()},
    )
    return token, expires


def revoke_session(db: Database, session_id: str, *, actor_id: str, actor_username: str) -> None:
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import AuthSession, utcnow

    with db.session() as session:
        row = session.get(AuthSession, session_id)
        if row is not None and row.revoked_at is None:
            row.revoked_at = utcnow()
    append_event(
        db,
        "auth.logout",
        actor_id=actor_id,
        actor_username=actor_username,
        subject_type="session",
        subject_id=session_id,
    )


def revoke_user_sessions(db: Database, user_id: str) -> int:
    """Revoga todas as sessões ativas de um usuário (desativação, troca de senha/papel)."""
    from sqlalchemy import update

    from radiologyai.persistence.models import AuthSession, utcnow

    with db.session() as session:
        result = session.execute(
            update(AuthSession)
            .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
            .values(revoked_at=utcnow())
        )
        return int(result.rowcount or 0)  # type: ignore[attr-defined]
