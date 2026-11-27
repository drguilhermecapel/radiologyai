"""Usuários e sessões: criação, autenticação, revogação.

Não existe usuário nem senha padrão. O primeiro administrador é criado pela
CLI (``radiologyai users create --role admin``); depois, só um administrador
cria usuários pela API. Toda operação é auditada (sem a senha, obviamente).
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
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

    from sqlalchemy.orm import Session

    from radiologyai.persistence.db import Database
    from radiologyai.persistence.models import User

__all__ = [
    "AuthenticationError",
    "LoginThrottledError",
    "UserExistsError",
    "authenticate",
    "create_user",
    "issue_session",
    "reauthenticate",
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
MAX_TRACKED = 10_000  # teto de chaves rastreadas (nomes aleatórios não crescem a memória)
REAUTH_REVOKE_AFTER = MAX_FAILURES  # falhas de reautenticação que revogam as sessões
_PROCESS_KEY = secrets.token_bytes(32)


def _key(username: str, client: str) -> str:
    """Chave do limitador: (usuário, cliente). Um atacante noutro IP não tranca o usuário."""
    return f"{username}\x00{client}"


def _throttled(key: str) -> bool:
    with _THROTTLE_LOCK:
        entry = _FAILURES.get(key)
        if entry is None:
            return False
        if time.monotonic() - entry.first > WINDOW_SECONDS:
            _FAILURES.pop(key, None)
            return False
        return entry.count >= MAX_FAILURES


def _record_failure(key: str) -> int:
    """Conta a falha; devolve o total na janela. Mantém no máximo ``MAX_TRACKED`` chaves."""
    with _THROTTLE_LOCK:
        now = time.monotonic()
        entry = _FAILURES.get(key)
        if entry is None or now - entry.first > WINDOW_SECONDS:
            if key not in _FAILURES and len(_FAILURES) >= MAX_TRACKED:
                expired = [k for k, v in _FAILURES.items() if now - v.first > WINDOW_SECONDS]
                for k in expired:
                    _FAILURES.pop(k, None)
                if len(_FAILURES) >= MAX_TRACKED:
                    oldest = min(_FAILURES, key=lambda k: _FAILURES[k].first)
                    _FAILURES.pop(oldest, None)
            _FAILURES[key] = _Failures(1, now)
            return 1
        entry.count += 1
        return entry.count


def _clear_failures(key: str) -> None:
    with _THROTTLE_LOCK:
        _FAILURES.pop(key, None)


def username_digest(username: str) -> str:
    """Resumo com chave do processo para nome que NÃO é usuário cadastrado.

    O nome digitado pode ser a senha (digitada no campo errado): nunca entra
    cru na trilha imutável; o resumo só correlaciona tentativas.
    """
    return hmac.new(_PROCESS_KEY, username.encode("utf-8"), hashlib.sha256).hexdigest()[:16]


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
        session.flush()
        append_event(
            None,
            "user.created",
            actor_id=actor_id,
            actor_username=actor_username or "cli",
            subject_type="user",
            subject_id=user.id,
            payload={"username": username, "role": role},
            session=session,
        )
    return user


def authenticate(db: Database, username: str, password: str, *, client: str = "") -> User:
    """Confere credenciais; sempre executa o hash (tempo uniforme).

    O limitador é por (usuário, cliente): cinco falhas do MESMO cliente em 15
    minutos bloqueiam aquele cliente, não o usuário em todo lugar.

    Raises:
        LoginThrottledError: falhas demais recentes deste cliente para o usuário.
        AuthenticationError: credenciais inválidas ou usuário inativo.
    """
    from sqlalchemy import select

    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import User, utcnow

    key = _key(username, client)
    if _throttled(key):
        raise LoginThrottledError("muitas tentativas; aguarde 15 minutos")
    with db.session() as session:
        user = session.execute(select(User).where(User.username == username)).scalar_one_or_none()
        # Hash fictício só para igualar o tempo de resposta quando o usuário não existe.
        stored = user.password_hash if user is not None else _DUMMY_HASH.get()
        ok = verify_password(stored, password) and user is not None and user.active
        if ok and user is not None:
            user.last_login_at = utcnow()
        known = user is not None
    if not ok or user is None:
        _record_failure(key)
        append_event(
            db,
            "auth.login_failed",
            actor_username=username[:64] if known else None,
            payload={
                "reason": "credenciais",
                **({} if known else {"unknown_username_digest": username_digest(username)}),
            },
        )
        raise AuthenticationError("usuário ou senha inválidos")
    _clear_failures(key)
    return user


def reauthenticate(db: Database, user_id: str, password: str) -> bool:
    """Reautenticação (ex.: assinatura) com limite, auditoria e revogação.

    Cada falha é auditada (``auth.reauth_failed``). Na ``REAUTH_REVOKE_AFTER``-ésima
    falha em 15 minutos, todas as sessões do usuário são revogadas (token roubado
    deixa de valer) e novas tentativas recebem ``LoginThrottledError``.

    Raises:
        LoginThrottledError: falhas demais recentes.
    """
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import User

    key = _key(f"reauth:{user_id}", "")
    if _throttled(key):
        raise LoginThrottledError("muitas tentativas de reautenticação; aguarde 15 minutos")
    with db.session() as session:
        user = session.get(User, user_id)
        stored = user.password_hash if user is not None else _DUMMY_HASH.get()
        ok = verify_password(stored, password) and user is not None and user.active
        username = user.username if user is not None else None
    if ok:
        _clear_failures(key)
        return True
    failures = _record_failure(key)
    with db.session() as session:
        revoked = 0
        if failures >= REAUTH_REVOKE_AFTER:
            revoked = revoke_user_sessions(None, user_id, session=session)
        append_event(
            None,
            "auth.reauth_failed",
            actor_id=user_id,
            actor_username=username,
            subject_type="user",
            subject_id=user_id,
            payload={"failures_in_window": failures, "sessions_revoked": revoked},
            session=session,
        )
    return False


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
        session.flush()
        append_event(
            None,
            "auth.login",
            actor_id=user.id,
            actor_username=user.username,
            subject_type="session",
            subject_id=row.id,
            payload={"expires_at": expires.isoformat()},
            session=session,
        )
    return token, expires


def revoke_session(db: Database, session_id: str, *, actor_id: str, actor_username: str) -> None:
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import AuthSession, utcnow

    with db.session() as session:
        row = session.get(AuthSession, session_id)
        if row is not None and row.revoked_at is None:
            row.revoked_at = utcnow()
        session.flush()
        append_event(
            None,
            "auth.logout",
            actor_id=actor_id,
            actor_username=actor_username,
            subject_type="session",
            subject_id=session_id,
            session=session,
        )


def revoke_user_sessions(
    db: Database | None, user_id: str, *, session: Session | None = None
) -> int:
    """Revoga todas as sessões ativas de um usuário (desativação, troca de senha/papel).

    Com ``session``, revoga na transação dela (atômico com a mudança que a motivou).
    """
    from sqlalchemy import update

    from radiologyai.persistence.models import AuthSession, utcnow

    stmt = (
        update(AuthSession)
        .where(AuthSession.user_id == user_id, AuthSession.revoked_at.is_(None))
        .values(revoked_at=utcnow())
        .execution_options(synchronize_session=False)
    )
    if session is not None:
        return int(session.execute(stmt).rowcount or 0)  # type: ignore[attr-defined]
    assert db is not None
    with db.session() as own:
        return int(own.execute(stmt).rowcount or 0)  # type: ignore[attr-defined]
