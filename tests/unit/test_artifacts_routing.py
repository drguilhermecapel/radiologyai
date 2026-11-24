from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from medai.core.artifacts import ArtifactStore, sha256_bytes
from medai.core.types import ArtifactRef, ExamType, ImageKind, Modality
from medai.routing import RoutingTable, infer_exam_type, load_routing, role_of
from tests.support import synth


def test_store_is_content_addressed_and_idempotent(store: ArtifactStore) -> None:
    a = store.put_bytes(b"hello", "application/json")
    b = store.put_bytes(b"hello", "application/json")
    assert a == b
    assert a.sha256 == sha256_bytes(b"hello")
    assert store.read_bytes(a) == b"hello"
    assert store.read_bytes(a.sha256) == b"hello"


def test_store_detects_corruption_and_refuses_traversal(
    store: ArtifactStore, tmp_path: Path
) -> None:
    ref = store.put_bytes(b"hello", "application/json")
    store.path(ref).write_bytes(b"HELLO")
    with pytest.raises(OSError, match="corrupted"):
        store.read_bytes(ref)
    (tmp_path / "secret.txt").write_text("x")
    evil = ArtifactRef(sha256="0" * 64, media_type="text/plain", relpath="../secret.txt")
    with pytest.raises(ValueError, match="escapes"):
        store.path(evil)
    with pytest.raises(ValueError, match="sha256"):
        store.path("../../etc/passwd")


def test_labelmap_and_png_round_trip(store: ArtifactStore) -> None:
    labelmap = np.arange(24, dtype=np.int64).reshape(2, 3, 4) % 5
    ref = store.put_labelmap(labelmap)
    assert ref.shape == (2, 3, 4)
    assert np.array_equal(store.read_labelmap(ref), labelmap)
    with pytest.raises(ValueError, match="integer"):
        store.put_labelmap(np.zeros((2, 2), dtype=np.float32))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="uint8"):
        store.put_png_gray(np.zeros((2, 2), dtype=np.float32))  # type: ignore[arg-type]


def test_packaged_routing_covers_every_exam_type() -> None:
    table = load_routing()
    assert set(table.exam_types) == set(ExamType) - {ExamType.UNKNOWN}


def test_routing_table_selection() -> None:
    table = RoutingTable.model_validate(
        {"exam_types": {"cxr": {"main": ["a"], "research": ["z"], "interactive": ["i"]}}}
    )
    assert table.models_for(ExamType.CXR) == ["a"]
    assert table.models_for(ExamType.CXR, include_research=True) == ["a", "z"]
    assert table.interactive_for(ExamType.CXR) == ["i"]
    assert table.models_for(ExamType.MG) == []
    assert table.all_model_ids() == {"a", "z", "i"}


def test_exam_type_inference_is_conservative() -> None:
    assert infer_exam_type([]) is ExamType.UNKNOWN
    assert infer_exam_type([synth.projection()]) is ExamType.CXR  # body_part CHEST
    assert infer_exam_type([synth.projection(body_part="KNEE")]) is ExamType.MSK_XR
    assert infer_exam_type([synth.projection(body_part=None)]) is ExamType.UNKNOWN
    assert infer_exam_type([synth.projection(modality=Modality.MG)]) is ExamType.MG
    assert infer_exam_type([synth.projection(modality=Modality.OT)]) is ExamType.UNKNOWN
    assert infer_exam_type([synth.volume()]) is ExamType.CT_ABDOMEN
    assert synth.volume().kind is ImageKind.VOLUME


def test_roles() -> None:
    mg = synth.projection(modality=Modality.MG, laterality="L", view_position="mlo")
    assert role_of(mg) == "L-MLO"
    assert role_of(synth.projection(modality=Modality.MG)) is None
    assert role_of(synth.projection()) is None
