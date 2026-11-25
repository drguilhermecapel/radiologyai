"""Persistência: migrações, senhas, sessões e trilha de auditoria encadeada no banco."""

from __future__ import annotations

import importlib.util
import json

import pytest

HAS = all(importlib.util.find_spec(m) for m in ("sqlalchemy", "alembic", "argon2"))
pytestmark = pytest.mark.skipif(not HAS, reason="requer o extra [platform]")


@pytest.fixture
def db(tmp_path):
    from radiologyai.persistence.db import Database

    database = Database(f"sqlite:///{tmp_path / 'p.db'}")
    database.migrate()
    yield database
    database.dispose()


@pytest.mark.requirement("REQ-166")
class TestMigrations:
    def test_migration_schema_equals_models(self, db):
        """O esquema criado pela migração Alembic é exatamente o dos modelos ORM."""
        from alembic.autogenerate import compare_metadata
        from alembic.runtime.migration import MigrationContext

        from radiologyai.persistence.models import Base

        with db.engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
        assert diff == []

    def test_migrate_is_idempotent_and_at_head(self, db):
        assert db.migrate() == db.head() == db.revision()

    def test_fresh_database_has_no_users(self, db):
        """Sem credencial padrão: o banco nasce sem nenhum usuário."""
        from sqlalchemy import func, select

        from radiologyai.persistence.models import User

        with db.session() as session:
            assert session.execute(select(func.count(User.id))).scalar_one() == 0


@pytest.mark.requirement("REQ-150")
class TestCredentials:
    def test_password_stored_only_as_argon2(self, db):
        from radiologyai.persistence.models import User
        from radiologyai.persistence.users import create_user

        user = create_user(
            db, username="ana", full_name="Ana", role="physician", password="uma-senha-longa-1"
        )
        with db.session() as session:
            stored = session.get(User, user.id).password_hash
        assert stored.startswith("$argon2id$")
        assert "uma-senha-longa-1" not in stored

    def test_weak_password_refused(self, db):
        from radiologyai.persistence.security import WeakPasswordError
        from radiologyai.persistence.users import create_user

        with pytest.raises(WeakPasswordError):
            create_user(db, username="b", full_name="B", role="viewer", password="curta")
        with pytest.raises(WeakPasswordError):
            create_user(db, username="c", full_name="C", role="viewer", password="a" * 20)

    def test_unknown_role_refused(self, db):
        from radiologyai.persistence.users import create_user

        with pytest.raises(ValueError, match="role|Role|superuser"):
            create_user(
                db, username="d", full_name="D", role="superuser", password="x" * 6 + "yz12345"
            )

    def test_token_stored_only_as_hash(self, db):
        from sqlalchemy import select

        from radiologyai.persistence.models import AuthSession
        from radiologyai.persistence.security import hash_token
        from radiologyai.persistence.users import create_user, issue_session

        user = create_user(
            db, username="e", full_name="E", role="viewer", password="senha-boa-12345"
        )
        token, _ = issue_session(db, user, 1.0)
        with db.session() as session:
            row = session.execute(select(AuthSession)).scalar_one()
        assert row.token_sha256 == hash_token(token)
        assert token not in row.token_sha256
        assert len(token) >= 40

    def test_authentication_is_generic_and_throttled(self, db):
        from radiologyai.persistence.users import (
            MAX_FAILURES,
            AuthenticationError,
            LoginThrottledError,
            authenticate,
            create_user,
            reset_throttle,
        )

        reset_throttle()
        create_user(db, username="f", full_name="F", role="viewer", password="senha-boa-12345")
        with pytest.raises(AuthenticationError, match="usuário ou senha inválidos"):
            authenticate(db, "nao-existe", "qualquer")
        for _ in range(MAX_FAILURES):
            with pytest.raises(AuthenticationError):
                authenticate(db, "f", "errada")
        with pytest.raises(LoginThrottledError):
            authenticate(db, "f", "senha-boa-12345")
        reset_throttle()
        assert authenticate(db, "f", "senha-boa-12345").username == "f"


@pytest.mark.requirement("REQ-155")
class TestAuditChain:
    def _fill(self, db, n=4):
        from radiologyai.persistence.audit import append_event

        return [
            append_event(db, "teste", actor_username="u", subject_id=str(i), payload={"i": i})
            for i in range(n)
        ]

    def test_clean_chain_verifies(self, db):
        from radiologyai.audit.trail import GENESIS_HASH
        from radiologyai.persistence.audit import verify_chain

        hashes = self._fill(db)
        with db.session() as session:
            status = verify_chain(session)
        assert status.ok
        assert status.events == 4
        assert status.last_hash == hashes[-1]
        from sqlalchemy import select

        from radiologyai.persistence.models import AuditEvent

        with db.session() as session:
            first = session.execute(select(AuditEvent).order_by(AuditEvent.seq)).scalars().first()
        assert first.prev_hash == GENESIS_HASH

    def test_modified_payload_detected_at_exact_event(self, db):
        from sqlalchemy import text

        from radiologyai.persistence.audit import verify_chain

        self._fill(db)
        with db.engine.begin() as conn:
            conn.execute(
                text("UPDATE audit_events SET payload_json = :p WHERE seq = 2"),
                {"p": json.dumps({"i": 999})},
            )
        with db.session() as session:
            status = verify_chain(session)
        assert not status.ok
        assert status.broken_at_seq == 2
        assert "adulterado" in status.detail

    def test_deleted_event_detected(self, db):
        from sqlalchemy import text

        from radiologyai.persistence.audit import verify_chain

        self._fill(db)
        with db.engine.begin() as conn:
            conn.execute(text("DELETE FROM audit_events WHERE seq = 3"))
        with db.session() as session:
            status = verify_chain(session)
        assert not status.ok
        assert status.broken_at_seq == 4

    def test_fork_from_same_link_impossible(self, db):
        """prev_hash é UNIQUE: duas continuações do mesmo elo não coexistem."""
        from sqlalchemy.exc import IntegrityError

        from radiologyai.persistence.models import AuditEvent

        self._fill(db, 2)
        with db.session() as session:
            first = session.get(AuditEvent, 1)
            prev = first.prev_hash
        with pytest.raises(IntegrityError), db.session() as session:
            session.add(
                AuditEvent(
                    seq=99,
                    timestamp_utc="x",
                    event="forjado",
                    payload_json="{}",
                    prev_hash=prev,
                    record_hash="f" * 64,
                )
            )

    def test_payload_with_patient_identifier_refused(self, db):
        from radiologyai.audit.trail import AuditError
        from radiologyai.persistence.audit import append_event

        with pytest.raises(AuditError, match="dado pessoal"):
            append_event(db, "x", payload={"nested": {"PatientName": "FULANO"}})
