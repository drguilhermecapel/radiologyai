"""esquema inicial da plataforma

Revision ID: 0001
Revises: (nenhuma)
Create Date: 2026-11-24 22:20:13.210713
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("media_type", sa.String(length=64), nullable=False),
        sa.Column("relpath", sa.String(length=128), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("sha256", name=op.f("pk_artifacts")),
    )
    op.create_table(
        "audit_events",
        sa.Column("seq", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("timestamp_utc", sa.String(length=40), nullable=False),
        sa.Column("event", sa.String(length=64), nullable=False),
        sa.Column("actor_id", sa.String(length=36), nullable=True),
        sa.Column("actor_username", sa.String(length=64), nullable=True),
        sa.Column("subject_type", sa.String(length=32), nullable=True),
        sa.Column("subject_id", sa.String(length=64), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=False),
        sa.Column("prev_hash", sa.String(length=64), nullable=False),
        sa.Column("record_hash", sa.String(length=64), nullable=False),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_audit_events")),
        sa.UniqueConstraint("prev_hash", name=op.f("uq_audit_events_prev_hash")),
        sa.UniqueConstraint("record_hash", name=op.f("uq_audit_events_record_hash")),
    )
    with op.batch_alter_table("audit_events", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_audit_events_event"), ["event"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_audit_events_subject_id"), ["subject_id"], unique=False
        )

    op.create_table(
        "users",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("full_name", sa.String(length=200), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("professional_id", sa.String(length=64), nullable=True),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_users")),
        sa.UniqueConstraint("username", name=op.f("uq_users_username")),
    )
    op.create_table(
        "auth_sessions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("user_id", sa.String(length=36), nullable=False),
        sa.Column("token_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], name=op.f("fk_auth_sessions_user_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_auth_sessions")),
        sa.UniqueConstraint("token_sha256", name=op.f("uq_auth_sessions_token_sha256")),
    )
    with op.batch_alter_table("auth_sessions", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_auth_sessions_user_id"), ["user_id"], unique=False)

    op.create_table(
        "studies",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_uid", sa.String(length=64), nullable=True),
        sa.Column("patient_pseudo_id", sa.String(length=64), nullable=True),
        sa.Column("modalities", sa.String(length=64), nullable=False),
        sa.Column("description", sa.String(length=256), nullable=True),
        sa.Column("exam_type", sa.String(length=32), nullable=False),
        sa.Column("exam_type_source", sa.String(length=16), nullable=False),
        sa.Column("exam_type_set_by", sa.String(length=36), nullable=True),
        sa.Column("exam_type_set_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("warnings", sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name=op.f("fk_studies_created_by_users")
        ),
        sa.ForeignKeyConstraint(
            ["exam_type_set_by"], ["users.id"], name=op.f("fk_studies_exam_type_set_by_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_studies")),
    )
    with op.batch_alter_table("studies", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_studies_created_at"), ["created_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_studies_exam_type"), ["exam_type"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_studies_patient_pseudo_id"), ["patient_pseudo_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_studies_study_uid"), ["study_uid"], unique=False)

    op.create_table(
        "analyses",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column("exam_type", sa.String(length=32), nullable=False),
        sa.Column("include_research", sa.Boolean(), nullable=False),
        sa.Column("model_ids", sa.JSON(), nullable=True),
        sa.Column("prompts", sa.JSON(), nullable=False),
        sa.Column("inputs", sa.JSON(), nullable=False),
        sa.Column("requested_by", sa.String(length=36), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("worker_id", sa.String(length=64), nullable=True),
        sa.Column("code_version", sa.String(length=32), nullable=False),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("result_sha256", sa.String(length=64), nullable=True),
        sa.Column("result_status", sa.String(length=16), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["requested_by"], ["users.id"], name=op.f("fk_analyses_requested_by_users")
        ),
        sa.ForeignKeyConstraint(
            ["study_id"], ["studies.id"], name=op.f("fk_analyses_study_id_studies")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_analyses")),
    )
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.create_index(
            "ix_analyses_state_requested", ["state", "requested_at"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_analyses_study_id"), ["study_id"], unique=False)

    op.create_table(
        "series",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("study_id", sa.String(length=36), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("series_uid", sa.String(length=64), nullable=True),
        sa.Column("modality", sa.String(length=8), nullable=False),
        sa.Column("source_format", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("description", sa.String(length=256), nullable=True),
        sa.Column("body_part", sa.String(length=64), nullable=True),
        sa.Column("view_position", sa.String(length=32), nullable=True),
        sa.Column("laterality", sa.String(length=1), nullable=True),
        sa.Column("rows", sa.Integer(), nullable=False),
        sa.Column("columns", sa.Integer(), nullable=False),
        sa.Column("n_frames", sa.Integer(), nullable=False),
        sa.Column("n_instances", sa.Integer(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("meta", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["study_id"], ["studies.id"], name=op.f("fk_series_study_id_studies")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_series")),
    )
    with op.batch_alter_table("series", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_series_study_id"), ["study_id"], unique=False)

    op.create_table(
        "findings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("model_id", sa.String(length=64), nullable=False),
        sa.Column("condition_id", sa.String(length=128), nullable=False),
        sa.Column("label", sa.String(length=128), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("decision", sa.String(length=16), nullable=False),
        sa.Column("evidence_level", sa.String(length=1), nullable=False),
        sa.Column("visibility", sa.String(length=8), nullable=False),
        sa.Column("raw_score", sa.Float(), nullable=True),
        sa.Column("op_normalized", sa.Float(), nullable=True),
        sa.Column("probability", sa.Float(), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("abstention_reason", sa.String(length=64), nullable=True),
        sa.Column("image_idx", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_id"], ["analyses.id"], name=op.f("fk_findings_analysis_id_analyses")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_findings")),
        sa.UniqueConstraint("analysis_id", "idx", name=op.f("uq_findings_analysis_id")),
    )
    with op.batch_alter_table("findings", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_findings_analysis_id"), ["analysis_id"], unique=False)
        batch_op.create_index(
            batch_op.f("ix_findings_condition_id"), ["condition_id"], unique=False
        )

    op.create_table(
        "instances",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("series_id", sa.String(length=36), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("sop_uid", sa.String(length=64), nullable=True),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("filename", sa.String(length=64), nullable=False),
        sa.ForeignKeyConstraint(
            ["artifact_sha256"],
            ["artifacts.sha256"],
            name=op.f("fk_instances_artifact_sha256_artifacts"),
        ),
        sa.ForeignKeyConstraint(
            ["series_id"], ["series.id"], name=op.f("fk_instances_series_id_series")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_instances")),
    )
    with op.batch_alter_table("instances", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_instances_series_id"), ["series_id"], unique=False)

    op.create_table(
        "reports",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("created_by", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("content_json", sa.Text(), nullable=False),
        sa.Column("report_sha256", sa.String(length=64), nullable=False),
        sa.Column("result_sha256", sa.String(length=64), nullable=False),
        sa.Column("signed_by", sa.String(length=36), nullable=True),
        sa.Column("signed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("signature_sha256", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(
            ["analysis_id"], ["analyses.id"], name=op.f("fk_reports_analysis_id_analyses")
        ),
        sa.ForeignKeyConstraint(
            ["created_by"], ["users.id"], name=op.f("fk_reports_created_by_users")
        ),
        sa.ForeignKeyConstraint(
            ["signed_by"], ["users.id"], name=op.f("fk_reports_signed_by_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reports")),
    )
    with op.batch_alter_table("reports", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_reports_analysis_id"), ["analysis_id"], unique=False)

    op.create_table(
        "reviews",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("analysis_id", sa.String(length=36), nullable=False),
        sa.Column("finding_idx", sa.Integer(), nullable=False),
        sa.Column("reviewer_id", sa.String(length=36), nullable=False),
        sa.Column("action", sa.String(length=8), nullable=False),
        sa.Column("amended_decision", sa.String(length=16), nullable=True),
        sa.Column("amended_text", sa.Text(), nullable=True),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column("result_sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["analysis_id"], ["analyses.id"], name=op.f("fk_reviews_analysis_id_analyses")
        ),
        sa.ForeignKeyConstraint(
            ["reviewer_id"], ["users.id"], name=op.f("fk_reviews_reviewer_id_users")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_reviews")),
    )
    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_reviews_analysis_id"), ["analysis_id"], unique=False)


def downgrade() -> None:
    with op.batch_alter_table("reviews", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_reviews_analysis_id"))

    op.drop_table("reviews")
    with op.batch_alter_table("reports", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_reports_analysis_id"))

    op.drop_table("reports")
    with op.batch_alter_table("instances", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_instances_series_id"))

    op.drop_table("instances")
    with op.batch_alter_table("findings", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_findings_condition_id"))
        batch_op.drop_index(batch_op.f("ix_findings_analysis_id"))

    op.drop_table("findings")
    with op.batch_alter_table("series", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_series_study_id"))

    op.drop_table("series")
    with op.batch_alter_table("analyses", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_analyses_study_id"))
        batch_op.drop_index("ix_analyses_state_requested")

    op.drop_table("analyses")
    with op.batch_alter_table("studies", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_studies_study_uid"))
        batch_op.drop_index(batch_op.f("ix_studies_patient_pseudo_id"))
        batch_op.drop_index(batch_op.f("ix_studies_exam_type"))
        batch_op.drop_index(batch_op.f("ix_studies_created_at"))

    op.drop_table("studies")
    with op.batch_alter_table("auth_sessions", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_auth_sessions_user_id"))

    op.drop_table("auth_sessions")
    op.drop_table("users")
    with op.batch_alter_table("audit_events", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_audit_events_subject_id"))
        batch_op.drop_index(batch_op.f("ix_audit_events_event"))

    op.drop_table("audit_events")
    op.drop_table("artifacts")
