"""Modo silencioso (REQ-195): portões e redação, sem banco nem modelo."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

HAS = all(importlib.util.find_spec(m) for m in ("fastapi", "sqlalchemy", "argon2"))
pytestmark = pytest.mark.skipif(not HAS, reason="requer os extras [api,platform]")


def _principal(role: str) -> Any:
    from radiologyai.api.deps import Principal

    return Principal("u1", f"{role}-x", "Pessoa", role, None, "s1")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    import radiologyai.persistence.audit as audit

    seen: list[dict[str, Any]] = []

    def fake(db: Any, event: str, **kw: Any) -> str:
        del db
        seen.append({"event": event, **kw})
        return "0" * 64

    monkeypatch.setattr(audit, "append_event", fake)
    return seen


def _platform(on: bool) -> Any:
    return SimpleNamespace(settings=SimpleNamespace(shadow_mode=on), db=None)


@pytest.mark.requirement("REQ-195")
@pytest.mark.parametrize("role", ["admin", "physician", "technician", "viewer", "researcher"])
def test_normal_mode_never_blocks(role: str, events: list[dict[str, Any]]) -> None:
    from radiologyai.api.shadow import ai_visible, block_ai, block_care_export

    platform = _platform(False)
    assert ai_visible(platform, _principal(role))
    block_ai(platform, _principal(role), action="x")
    block_care_export(platform, _principal(role), action="x")
    assert events == []


@pytest.mark.requirement("REQ-195")
@pytest.mark.parametrize("role", ["admin", "physician", "technician", "viewer"])
def test_shadow_blocks_clinical_roles_and_audits(role: str, events: list[dict[str, Any]]) -> None:
    from radiologyai.api.shadow import ShadowModeError, ai_visible, block_ai

    platform = _platform(True)
    assert not ai_visible(platform, _principal(role))
    with pytest.raises(ShadowModeError):
        block_ai(platform, _principal(role), action="report.pdf", subject_id="r1")
    (event,) = events
    assert event["event"] == "shadow.blocked"
    assert event["payload"] == {"action": "report.pdf", "role": role, "care_export": False}


@pytest.mark.requirement("REQ-195")
def test_shadow_lets_researcher_see_but_never_export_to_care(events: list[dict[str, Any]]) -> None:
    from radiologyai.api.shadow import ShadowModeError, ai_visible, block_ai, block_care_export

    platform = _platform(True)
    researcher = _principal("researcher")
    assert ai_visible(platform, researcher)
    block_ai(platform, researcher, action="report.pdf")
    assert events == []
    with pytest.raises(ShadowModeError):
        block_care_export(platform, researcher, action="report.export")
    assert events[-1]["payload"]["care_export"] is True


@pytest.mark.requirement("REQ-195")
def test_shadow_error_maps_to_403() -> None:
    from radiologyai.api.app import _error_status
    from radiologyai.api.shadow import ShadowModeError

    assert dict(_error_status())[ShadowModeError] == 403


@pytest.mark.requirement("REQ-195")
def test_redact_summary_uses_none_not_zero() -> None:
    from radiologyai.api.schemas import AnalysisSummary
    from radiologyai.api.shadow import redact_summary

    summary = AnalysisSummary(
        id="a",
        state="completed",
        result_status="completed",
        requested_at=datetime.now(UTC),
        finished_at=datetime.now(UTC),
        include_research=False,
        n_main_findings=3,
        n_research_findings=1,
    )
    out = redact_summary(summary)
    assert out.shadow is True
    assert out.n_main_findings is None  # vazio não é "zero achados" nem "exame normal"
    assert out.n_research_findings is None
    assert out.result_status is None
    assert summary.shadow is False
