"""Trilha de auditoria — a adulteração precisa ser detectável."""

from __future__ import annotations

import json

import pytest

from radiologyai.audit import GENESIS_HASH, AuditError, AuditTrail, PhysicianDecision

SHA = "d" * 64


def sample(**overrides):
    base = {
        "event": "inference",
        "input_sha256": SHA,
        "card_id": "xrv-densenet121-pc",
        "weights_sha256": "a" * 64,
        "code_version": "2.0.0.dev0",
        "modality": "XR",
        "user_id": "crm-sp-175873",
    }
    base.update(overrides)
    return base


@pytest.fixture
def trail(tmp_path):
    return AuditTrail(tmp_path / "audit.jsonl")


class TestAppendAndRead:
    def test_empty_trail_starts_at_genesis(self, trail):
        assert trail.last_hash() == GENESIS_HASH
        assert len(trail) == 0

    def test_append_returns_hash(self, trail):
        h = trail.record(**sample())
        assert len(h) == 64
        assert trail.last_hash() == h

    def test_records_are_chained(self, trail):
        first = trail.record(**sample())
        trail.record(**sample(event="review"))
        entries = trail.read()
        assert entries[0]["prev_hash"] == GENESIS_HASH
        assert entries[1]["prev_hash"] == first

    def test_read_preserves_order(self, trail):
        for i in range(5):
            trail.record(**sample(note=f"n{i}"))
        assert [e["note"] for e in trail.read()] == [f"n{i}" for i in range(5)]

    def test_no_patient_pixels_stored(self, trail):
        """A trilha guarda o hash da entrada, nunca a entrada."""
        trail.record(**sample())
        raw = trail.path.read_text(encoding="utf-8")
        assert SHA in raw
        assert "PixelData" not in raw


@pytest.mark.requirement("REQ-070")
class TestTamperDetection:
    def test_clean_chain_verifies(self, trail):
        for i in range(4):
            trail.record(**sample(note=str(i)))
        trail.verify()

    def test_modified_field_detected(self, trail):
        for i in range(3):
            trail.record(**sample(note=str(i)))

        lines = trail.path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[1])
        entry["physician_decision"] = PhysicianDecision.ACCEPTED
        lines[1] = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        trail.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(AuditError, match="adulterado"):
            trail.verify()

    def test_deleted_record_breaks_chain(self, trail):
        for i in range(4):
            trail.record(**sample(note=str(i)))

        lines = trail.path.read_text(encoding="utf-8").splitlines()
        del lines[1]
        trail.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(AuditError, match="elo quebrado"):
            trail.verify()

    def test_reordered_records_detected(self, trail):
        for i in range(3):
            trail.record(**sample(note=str(i)))

        lines = trail.path.read_text(encoding="utf-8").splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        trail.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(AuditError, match="elo quebrado"):
            trail.verify()

    def test_error_names_the_position(self, trail):
        for i in range(5):
            trail.record(**sample(note=str(i)))
        lines = trail.path.read_text(encoding="utf-8").splitlines()
        entry = json.loads(lines[3])
        entry["note"] = "alterado"
        lines[3] = json.dumps(entry, sort_keys=True, ensure_ascii=False)
        trail.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(AuditError, match="registro 3"):
            trail.verify()

    def test_corrupted_line_detected(self, trail):
        trail.record(**sample())
        trail.path.write_text("{isto nao e json}\n", encoding="utf-8")
        with pytest.raises(AuditError, match="corrompida"):
            trail.verify()

    def test_out_of_order_append_refused(self, trail):
        from radiologyai.audit import AuditRecord

        trail.record(**sample())
        stale = AuditRecord(
            timestamp_utc="2026-01-01T00:00:00Z", prev_hash=GENESIS_HASH, **sample()
        )
        with pytest.raises(AuditError, match="prev_hash não confere"):
            trail.append(stale)


@pytest.mark.requirement("REQ-071")
class TestDecisionReconstruction:
    def test_reconstructs_full_decision_path(self, trail):
        """Um radiologista precisa conseguir reconstruir o que aconteceu."""
        trail.record(**sample(event="inference", findings=[{"label": "Effusion", "score": 0.81}]))
        trail.record(
            **sample(
                event="review",
                physician_decision=PhysicianDecision.EDITED,
                note="Concordo com derrame; acrescento atelectasia.",
            )
        )

        history = trail.reconstruct(SHA)
        assert len(history) == 2
        assert history[0]["event"] == "inference"
        assert history[1]["physician_decision"] == PhysicianDecision.EDITED

    def test_every_record_identifies_model_and_weights(self, trail):
        trail.record(**sample())
        entry = trail.read()[0]
        for key in ("card_id", "weights_sha256", "code_version", "modality", "user_id"):
            assert entry[key], f"{key} ausente — decisão não reconstruível"

    def test_physician_decision_defaults_to_pending(self, trail):
        trail.record(**sample())
        assert trail.read()[0]["physician_decision"] == PhysicianDecision.PENDING

    def test_filters_by_input(self, trail):
        trail.record(**sample())
        trail.record(**sample(input_sha256="e" * 64))
        assert len(trail.reconstruct(SHA)) == 1

    def test_abstention_is_recorded(self, trail):
        trail.record(**sample(abstained=True, note="qualidade insuficiente"))
        assert trail.read()[0]["abstained"] is True


class TestHashDeterminism:
    def test_same_content_same_hash(self):
        from radiologyai.audit import AuditRecord

        kwargs = {"timestamp_utc": "2026-01-01T00:00:00Z", "prev_hash": GENESIS_HASH, **sample()}
        assert AuditRecord(**kwargs).compute_hash() == AuditRecord(**kwargs).compute_hash()

    def test_different_content_different_hash(self):
        from radiologyai.audit import AuditRecord

        base = {"timestamp_utc": "2026-01-01T00:00:00Z", "prev_hash": GENESIS_HASH}
        a = AuditRecord(**base, **sample(note="a"))
        b = AuditRecord(**base, **sample(note="b"))
        assert a.compute_hash() != b.compute_hash()
