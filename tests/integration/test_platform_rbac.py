"""Controle de acesso: cada rota declara uma permissão e a matriz é imposta rota a rota."""

from __future__ import annotations

import re

import pytest
from platform_support import HAS_PLATFORM, PASSWORD

pytestmark = pytest.mark.skipif(not HAS_PLATFORM, reason="requer os extras [api,platform]")

ROLES = ("admin", "physician", "technician", "viewer")
DUMMY = {
    "sha256": "0" * 64,
    "index": "0",
    "seg_index": "0",
}


def api_routes(app):
    """(método, caminho, permissão) de cada operação, lidos do OpenAPI gerado."""
    for path, ops in app.openapi()["paths"].items():
        for method, op in ops.items():
            yield method.upper(), path, op.get("x-permission")


def fill(path: str) -> str:
    return re.sub(
        r"\{(\w+)\}",
        lambda m: DUMMY.get(m.group(1), "00000000-0000-0000-0000-000000000000"),
        path,
    )


def call(api, method, path, headers):
    kwargs = {"headers": headers}
    if method in ("POST", "PUT", "PATCH") and path.rsplit("/", 1)[-1] != "studies":
        kwargs["json"] = {}
    return api.request(method, path, **kwargs)


@pytest.mark.requirement("REQ-151")
class TestRbacMatrix:
    def test_every_route_declares_a_permission(self, api):
        from radiologyai.persistence.security import Permission

        valid = {p.value for p in Permission} | {"public", "authenticated"}
        routes = list(api_routes(api.app))
        assert len(routes) > 30
        for method, path, declared in routes:
            assert declared in valid, f"{method} {path} sem x-permission válida"

    def test_only_health_login_and_measured_metrics_are_public(self, api):
        public = {path for _, path, declared in api_routes(api.app) if declared == "public"}
        assert public == {
            "/api/v1/healthz",
            "/api/v1/readyz",
            "/api/v1/health",
            "/api/v1/status",
            "/api/v1/metrics",
            "/api/v1/metrics/runs",
            "/api/v1/auth/login",
        }

    def test_matrix_enforced_route_by_route(self, api, headers):
        from radiologyai.persistence.security import PERMISSIONS, Permission, Role

        checked = 0
        for method, raw_path, declared in api_routes(api.app):
            path = fill(raw_path)
            if declared == "public" or raw_path.endswith("/auth/logout"):
                continue  # logout revogaria a sessão do papel; coberto em TestSessions
            anonymous = call(api, method, path, {})
            assert anonymous.status_code == 401, (
                f"{method} {path} sem token: {anonymous.status_code}"
            )
            for role in ROLES:
                response = call(api, method, path, headers[role])
                if declared == "authenticated":
                    allowed = True
                else:
                    allowed = Role(role) in PERMISSIONS[Permission(declared)]
                if allowed:
                    assert response.status_code not in (401, 403), (
                        f"{role} deveria acessar {method} {path}: {response.status_code} "
                        f"{response.text[:200]}"
                    )
                else:
                    assert response.status_code == 403, (
                        f"{role} NÃO deveria acessar {method} {path}: {response.status_code}"
                    )
                checked += 1
        assert checked > 100

    def test_signing_and_reviewing_are_physician_only(self):
        from radiologyai.persistence.security import PERMISSIONS, Permission, Role

        for p in (Permission.REPORTS_SIGN, Permission.REVIEWS_WRITE, Permission.REPORTS_CREATE):
            assert PERMISSIONS[p] == frozenset({Role.PHYSICIAN})
        assert Role.ADMIN not in PERMISSIONS[Permission.REPORTS_SIGN]
        assert Role.VIEWER not in PERMISSIONS[Permission.RESEARCH_MODE]
        assert Role.TECHNICIAN not in PERMISSIONS[Permission.RESEARCH_MODE]


@pytest.mark.requirement("REQ-150")
class TestSessions:
    def test_login_me_logout_revokes_token(self, api, make_user):
        make_user("viewer", username="vera")
        login = api.post("/api/v1/auth/login", json={"username": "vera", "password": PASSWORD})
        assert login.status_code == 200, login.text
        body = login.json()
        assert body["token_type"] == "bearer"
        assert body["user"]["role"] == "viewer"
        auth = {"Authorization": f"Bearer {body['token']}"}
        me = api.get("/api/v1/auth/me", headers=auth).json()
        assert "studies:read" in me["permissions"]
        assert "reports:sign" not in me["permissions"]
        assert api.post("/api/v1/auth/logout", headers=auth).status_code == 204
        assert api.get("/api/v1/auth/me", headers=auth).status_code == 401

    def test_wrong_password_is_generic_401(self, api, make_user):
        make_user("viewer", username="vera")
        wrong = api.post("/api/v1/auth/login", json={"username": "vera", "password": "x" * 14})
        ghost = api.post("/api/v1/auth/login", json={"username": "ninguem", "password": "x" * 14})
        assert wrong.status_code == ghost.status_code == 401
        assert wrong.json()["detail"] == ghost.json()["detail"]

    def test_expired_session_rejected(self, api, make_user, platform):
        from datetime import UTC, datetime, timedelta

        from radiologyai.persistence.models import AuthSession

        auth = make_user("viewer", username="vera")
        with platform.db.session() as session:
            for row in session.query(AuthSession).all():
                row.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        assert api.get("/api/v1/auth/me", headers=auth).status_code == 401

    def test_deactivation_and_role_change_revoke_sessions(self, api, headers, make_user):
        auth = make_user("technician", username="tito")
        users = api.get("/api/v1/users", headers=headers["admin"]).json()
        tito = next(u for u in users if u["username"] == "tito")
        changed = api.patch(
            f"/api/v1/users/{tito['id']}", headers=headers["admin"], json={"role": "viewer"}
        )
        assert changed.status_code == 200
        assert changed.json()["role"] == "viewer"
        assert api.get("/api/v1/auth/me", headers=auth).status_code == 401

    def test_admin_creates_user_with_password_policy(self, api, headers):
        weak = api.post(
            "/api/v1/users",
            headers=headers["admin"],
            json={"username": "novo", "full_name": "N", "role": "viewer", "password": "123"},
        )
        assert weak.status_code == 422
        ok = api.post(
            "/api/v1/users",
            headers=headers["admin"],
            json={"username": "novo", "full_name": "N", "role": "viewer", "password": PASSWORD},
        )
        assert ok.status_code == 201
        dup = api.post(
            "/api/v1/users",
            headers=headers["admin"],
            json={"username": "novo", "full_name": "N", "role": "viewer", "password": PASSWORD},
        )
        assert dup.status_code == 409

    def test_cors_closed_by_default(self, api):
        response = api.options(
            "/api/v1/healthz",
            headers={"Origin": "https://malicioso.example", "Access-Control-Request-Method": "GET"},
        )
        assert "access-control-allow-origin" not in {k.lower() for k in response.headers}
