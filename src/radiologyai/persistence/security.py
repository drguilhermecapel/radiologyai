"""Senhas (argon2id), tokens de sessão opacos e a matriz de papéis (RBAC).

* Senha só é guardada como hash argon2id; não existe credencial padrão.
* Token de sessão: 32 bytes aleatórios do ``secrets`` (CSPRNG do sistema),
  entregue uma única vez ao cliente; o banco guarda apenas o sha256.
* A matriz :data:`PERMISSIONS` é a fonte única do controle de acesso: cada rota
  da API declara uma permissão, e os testes conferem rota a rota.
"""

from __future__ import annotations

import hashlib
import secrets
from enum import StrEnum

from radiologyai.errors import RadiologyAIError

__all__ = [
    "PERMISSIONS",
    "Permission",
    "Role",
    "WeakPasswordError",
    "allowed",
    "check_password_policy",
    "hash_password",
    "hash_token",
    "new_token",
    "verify_password",
]


class WeakPasswordError(RadiologyAIError):
    """Senha fora da política mínima."""


class Role(StrEnum):
    ADMIN = "admin"
    PHYSICIAN = "physician"
    TECHNICIAN = "technician"
    VIEWER = "viewer"
    #: Equipe do piloto silencioso (G4): lê a saída de IA para comparação posterior,
    #: sem ato médico (não revisa, não cria nem assina laudo). REQ-195.
    RESEARCHER = "researcher"


class Permission(StrEnum):
    USERS_MANAGE = "users:manage"
    STUDIES_READ = "studies:read"
    STUDIES_UPLOAD = "studies:upload"
    STUDIES_SET_EXAM_TYPE = "studies:set_exam_type"
    ANALYSES_CREATE = "analyses:create"
    ANALYSES_READ = "analyses:read"
    RESEARCH_MODE = "research:view"
    REVIEWS_WRITE = "reviews:write"
    REPORTS_CREATE = "reports:create"
    REPORTS_READ = "reports:read"
    REPORTS_SIGN = "reports:sign"
    MODELS_READ = "models:read"
    CATALOG_READ = "catalog:read"
    AUDIT_READ = "audit:read"
    DICOMWEB_IMPORT = "dicomweb:import"
    DICOMWEB_EXPORT = "dicomweb:export"
    LEGACY_ANALYZE = "legacy:analyze"
    #: Vê a saída de IA quando o modo silencioso está ligado (REQ-195). ``research:view``
    #: não serve de portão: o médico assistente o tem.
    SHADOW_VIEW = "shadow:view"


_A, _P, _T, _V = Role.ADMIN, Role.PHYSICIAN, Role.TECHNICIAN, Role.VIEWER
_R = Role.RESEARCHER

#: Matriz RBAC. Assinar e revisar são atos médicos: o administrador NÃO assina.
PERMISSIONS: dict[Permission, frozenset[Role]] = {
    Permission.USERS_MANAGE: frozenset({_A}),
    Permission.STUDIES_READ: frozenset({_A, _P, _T, _V, _R}),
    Permission.STUDIES_UPLOAD: frozenset({_A, _P, _T}),
    Permission.STUDIES_SET_EXAM_TYPE: frozenset({_P, _T}),
    Permission.ANALYSES_CREATE: frozenset({_P, _T, _R}),
    Permission.ANALYSES_READ: frozenset({_A, _P, _T, _V, _R}),
    Permission.RESEARCH_MODE: frozenset({_A, _P, _R}),
    Permission.REVIEWS_WRITE: frozenset({_P}),
    Permission.REPORTS_CREATE: frozenset({_P}),
    Permission.REPORTS_READ: frozenset({_A, _P, _T, _V, _R}),
    Permission.REPORTS_SIGN: frozenset({_P}),
    Permission.MODELS_READ: frozenset({_A, _P, _T, _V, _R}),
    Permission.CATALOG_READ: frozenset({_A, _P, _T, _V, _R}),
    Permission.AUDIT_READ: frozenset({_A}),
    Permission.DICOMWEB_IMPORT: frozenset({_A, _P, _T}),
    Permission.DICOMWEB_EXPORT: frozenset({_P}),
    Permission.LEGACY_ANALYZE: frozenset({_P, _T}),
    Permission.SHADOW_VIEW: frozenset({_R}),
}


def allowed(role: str, permission: Permission) -> bool:
    """O papel tem a permissão? Papel desconhecido nunca tem."""
    try:
        return Role(role) in PERMISSIONS[permission]
    except ValueError:
        return False


def check_password_policy(password: str, min_length: int) -> None:
    """Política mínima: comprimento e não ser só um tipo de caractere.

    Raises:
        WeakPasswordError: senha curta ou trivial.
    """
    if len(password) < min_length:
        raise WeakPasswordError(f"a senha deve ter ao menos {min_length} caracteres")
    if len(set(password)) < 5:
        raise WeakPasswordError("a senha tem pouca variedade de caracteres")


def hash_password(password: str) -> str:
    from argon2 import PasswordHasher

    return PasswordHasher().hash(password)


def verify_password(stored_hash: str, password: str) -> bool:
    """Confere a senha em tempo constante (argon2); ``False`` em qualquer divergência."""
    from argon2 import PasswordHasher
    from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

    try:
        return bool(PasswordHasher().verify(stored_hash, password))
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


def new_token() -> str:
    """Token opaco de sessão (256 bits do CSPRNG do sistema operacional)."""
    return secrets.token_urlsafe(32)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
