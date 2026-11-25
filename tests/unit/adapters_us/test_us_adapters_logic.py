"""Morfometria, sondas (determinismo), manifestos e fluxo dos adaptadores (segmentador injetado).

O segmentador injetado devolve uma máscara GEOMÉTRICA conhecida para testar o
encanamento (prompts → quadros → medidas → abstenções); nenhum valor clínico é
produzido por ele e ele nunca é distribuído.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from us_support import ellipse, half_ellipse, have, us_image

from radiologyai.analysis.adapters.us_common import (
    bbox_array,
    lesion_morphometry,
    pad_box,
)
from radiologyai.analysis.base import ModelInputs
from radiologyai.analysis.manifest import load_manifest
from radiologyai.analysis.types import AbstentionReason, ImageKind, Primitive
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import UnsupportedInputError

MANIFESTS = Path(__file__).resolve().parents[3] / "src/radiologyai/analysis/manifests"


# --------------------------------------------------------------------------- #
# Morfometria (REQ-147)
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-147")
@pytest.mark.parametrize("spacing", [(0.1, 0.1), (0.2, 0.1)])
def test_morphometry_of_ellipse(spacing: tuple[float, float]) -> None:
    shape = (int(40 / spacing[0]), int(60 / spacing[1]))
    m = ellipse(shape, (20.0, 30.0), (12.0, 7.0), spacing)
    morph = lesion_morphometry(m, spacing)
    assert morph.area_mm2 == pytest.approx(math.pi * 12 * 7, rel=0.02)
    assert morph.max_diameter_mm == pytest.approx(24.0, rel=0.02)
    assert morph.perpendicular_diameter_mm == pytest.approx(14.0, rel=0.03)


@pytest.mark.requirement("REQ-147")
def test_morphometry_without_spacing_returns_no_mm() -> None:
    m = ellipse((100, 100), (50, 50), (20, 10), (1.0, 1.0))
    morph = lesion_morphometry(m, None)
    assert morph.area_mm2 is None
    assert morph.max_diameter_mm is None
    assert morph.perpendicular_diameter_mm is None
    assert morph.area_px == int(m.sum())
    with pytest.raises(UnsupportedInputError):
        lesion_morphometry(np.zeros((5, 5), bool), (1.0, 1.0))


def test_box_helpers_clip_and_order() -> None:
    p = Primitive(type="bbox", xy=((50.0, 40.0), (10.0, 5.0)))
    assert bbox_array(p, (30, 30)).tolist() == [10.0, 5.0, 29.0, 29.0]
    assert pad_box(np.array([2.0, 2.0, 8.0, 8.0]), (10, 10), 5).tolist() == [0, 0, 9, 9]
    with pytest.raises(UnsupportedInputError):
        bbox_array(Primitive(type="bbox", xy=((1.0, 1.0), (2.0, 2.0))), (30, 30))


# --------------------------------------------------------------------------- #
# Sondas (REQ-148)
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-148")
@pytest.mark.skipif(not have("sklearn", "safetensors"), reason="extras eval/foundation ausentes")
def test_probe_fit_and_apply_are_deterministic(tmp_path: Path) -> None:
    from radiologyai.analysis.probes.artifact import (
        ProbeArtifact,
        load_probe,
        probe_logit,
        save_probe,
    )
    from radiologyai.analysis.probes.fit import fit_probe

    grid = np.linspace(-1.0, 1.0, 40)
    x = np.stack([np.sin(grid * (k + 1)) for k in range(6)], axis=1)
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(int)
    groups = np.arange(40) // 2
    f1 = fit_probe(x, y, groups, n_folds=4)
    f2 = fit_probe(x, y, groups, n_folds=4)
    assert f1.C == f2.C
    np.testing.assert_array_equal(f1.coef, f2.coef)
    art = ProbeArtifact(
        label="t",
        encoder_id="e",
        embedding_dim=6,
        mean=f1.mean,
        scale=f1.scale,
        coef=f1.coef,
        intercept=f1.intercept,
        training_data_sha256="a" * 64,
    )
    t, j = save_probe(art, tmp_path / "probe")
    again = load_probe(t.read_bytes(), j.read_bytes())
    np.testing.assert_array_equal(probe_logit(art, x), probe_logit(again, x))


@pytest.mark.requirement("REQ-148")
@pytest.mark.skipif(not have("PIL"), reason="Pillow ausente")
def test_us_canonical_is_deterministic_and_square() -> None:
    from radiologyai.analysis.adapters.us_probe import expand_box, us_canonical

    frame = (np.add.outer(np.arange(120), np.arange(90)) % 255).astype(np.uint8)
    box = np.array([10.0, 20.0, 50.0, 40.0])
    a = us_canonical(frame, box, 0.25)
    b = us_canonical(frame, box, 0.25)
    assert a.shape == (224, 224)
    assert a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert expand_box(box, frame.shape, 0.25) == (0, 15, 61, 46)
    assert us_canonical(frame, None, 0.0).shape == (224, 224)


@pytest.mark.requirement("REQ-148")
def test_us_probe_manifest_rules() -> None:
    from radiologyai.analysis.adapters.us_probe import USProbeAdapter
    from radiologyai.analysis.manifest import parse_manifest
    from radiologyai.errors import ManifestError

    base = {
        "id": "us-probe-x",
        "version": "1",
        "name": "x",
        "status": "planned",
        "planned_reason": "teste",
        "task": "classification",
        "capabilities": ["classify"],
        "summary": "x",
        "license": {"code": "MIT", "weights": "MIT", "commercial_use": "no"},
        "applicability": {
            "exam_types": ["us_breast"],
            "modalities": ["US"],
            "image_kinds": ["projection"],
        },
        "labels": [
            {
                "key": "l",
                "source_label": "l",
                "condition_id": "c",
                "score_kind": "logit",
                "code": {"scheme": "99MEDAI", "value": "c", "meaning": "c"},
            }
        ],
        "weights": [
            {"key": k, "url": "https://x.invalid/a", "filename": f"{k}.bin"}
            for k in ("encoder_weights", "probe_tensors", "probe_meta")
        ],
        "config": {"roi": "box"},
    }
    with pytest.raises(ManifestError, match="prompts"):
        USProbeAdapter(parse_manifest(base))
    ok = USProbeAdapter(parse_manifest({**base, "inputs": {"prompts": "required"}}))
    assert ok.roi == "box"
    with pytest.raises(ManifestError, match="roi"):
        USProbeAdapter(parse_manifest({**base, "config": {"roi": "other"}}))


# --------------------------------------------------------------------------- #
# Manifestos da frente US/eco
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-146")
def test_us_echo_manifests_load_and_declare_licences() -> None:
    m, _ = load_manifest(MANIFESTS / "medsam2-echo-lv.yaml")
    assert m.status == "available"
    assert m.license.commercial_use == "no"
    assert m.inputs.prompts == "required"
    assert m.effective_route == "interactive"
    assert m.visibility_default == "research"
    assert not m.labels  # nenhuma FE vira achado sem limiar com origem verificada
    s, _ = load_manifest(MANIFESTS / "medsam-vit-base.yaml")
    assert s.inputs.prompts == "required"
    for planned in ("panecho", "echoprime"):
        p, _ = load_manifest(MANIFESTS / f"{planned}.yaml")
        assert p.status == "planned"
        assert p.planned_reason


# --------------------------------------------------------------------------- #
# Fluxo do adaptador de eco com segmentador geométrico injetado (REQ-146)
# --------------------------------------------------------------------------- #
class _GeometricSegmenter:
    """Devolve meia elipse centrada na caixa: encanamento, não modelo."""

    def segment(self, frame: np.ndarray, box: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = frame.shape
        sp = (0.3, 0.3)
        height_mm = (box[3] - box[1]) * sp[0]
        width_mm = (box[2] - box[0]) * sp[1]
        mask = half_ellipse(height_mm / 1.2, width_mm / 2.6, sp)
        out = np.zeros((h, w), bool)
        y0, x0 = int(box[1]), int(box[0])
        hh, ww = min(mask.shape[0], h - y0), min(mask.shape[1], w - x0)
        out[y0 : y0 + hh, x0 : x0 + ww] = mask[:hh, :ww]
        return out, 0.9


def _echo_adapter() -> object:
    from radiologyai.analysis.adapters.echo_medsam2 import MedSAM2EchoLVAdapter

    m, _ = load_manifest(MANIFESTS / "medsam2-echo-lv.yaml")
    adapter = MedSAM2EchoLVAdapter(m)
    adapter.segmenter = _GeometricSegmenter()
    adapter.loaded = True
    return adapter


def _clip(view: str, spacing: tuple[float, float] | None = (0.3, 0.3)) -> object:
    frames = np.full((6, 400, 300), 80.0, dtype=np.float32)
    return us_image(
        frames, kind=ImageKind.CINE, spacing=spacing, view_position=view, patient_age_years=60.0
    )


def _prompts(views: tuple[str, ...]) -> tuple[Primitive, ...]:
    out = []
    for v in views:
        out.append(
            Primitive(type="bbox", xy=((40.0, 30.0), (240.0, 330.0)), frame=0, label=f"{v}:ed")
        )
        out.append(
            Primitive(type="bbox", xy=((60.0, 60.0), (220.0, 300.0)), frame=5, label=f"{v}:es")
        )
    return tuple(out)


@pytest.mark.requirement("REQ-146")
@pytest.mark.skipif(not have("scipy"), reason="scipy ausente")
def test_echo_adapter_requires_ed_es_boxes() -> None:
    adapter = _echo_adapter()
    images = (_clip("A4C"),)
    no_prompt = ModelInputs(images=images, exam_type=ExamType.ECHO, roles={"a4c": 0})
    qa = adapter.applicability(no_prompt)
    assert not qa.applicable
    assert AbstentionReason.MISSING_INPUT in qa.reasons
    only_ed = ModelInputs(
        images=images,
        exam_type=ExamType.ECHO,
        roles={"a4c": 0},
        prompts=_prompts(("a4c",))[:1],
    )
    assert not adapter.applicability(only_ed).applicable
    bad_frame = ModelInputs(
        images=images,
        exam_type=ExamType.ECHO,
        roles={"a4c": 0},
        prompts=(
            *_prompts(("a4c",))[:1],
            Primitive(type="bbox", xy=((1.0, 1.0), (90.0, 90.0)), frame=99, label="a4c:es"),
        ),
    )
    rep = adapter.applicability(bad_frame)
    assert not rep.applicable
    assert AbstentionReason.INPUT in rep.reasons


@pytest.mark.requirement("REQ-146")
@pytest.mark.skipif(not have("scipy"), reason="scipy ausente")
def test_echo_adapter_biplane_flow_and_labelmaps() -> None:
    adapter = _echo_adapter()
    inputs = ModelInputs(
        images=(_clip("A4C"), _clip("A2C")),
        exam_type=ExamType.ECHO,
        roles={"a4c": 0, "a2c": 1},
        prompts=_prompts(("a4c", "a2c")),
    )
    assert adapter.applicability(inputs).applicable
    out = adapter.predict(inputs)
    meas = {m.key: m for m in out.measurements}
    assert meas["lv_ef"].value is not None
    assert 0 < meas["lv_ef"].value < 100
    assert meas["lv_ef"].preconditions["views"] == "A4C+A2C"
    assert {lm.name for lm in out.labelmaps} == {"lv_cavity_a4c", "lv_cavity_a2c"}
    lm = out.labelmaps[0]
    assert lm.array.shape == (6, 400, 300)
    assert lm.array[0].any()
    assert lm.array[5].any()
    assert not lm.array[2].any()
    assert meas["lv_ef"].derived_from == ["lv_cavity_a4c", "lv_cavity_a2c"]


@pytest.mark.requirement("REQ-145")
@pytest.mark.skipif(not have("scipy"), reason="scipy ausente")
def test_echo_adapter_abstains_without_spacing() -> None:
    adapter = _echo_adapter()
    inputs = ModelInputs(
        images=(_clip("A4C", spacing=None),),
        exam_type=ExamType.ECHO,
        roles={"a4c": 0},
        prompts=_prompts(("a4c",)),
    )
    out = adapter.predict(inputs)
    meas = {m.key: m for m in out.measurements}
    assert meas["lv_ef"].value is None
    assert meas["lv_ef"].abstention is AbstentionReason.GEOMETRY
    assert out.labelmaps  # a máscara continua disponível para revisão


# --------------------------------------------------------------------------- #
# Fluxo do MedSAM com segmentador injetado (REQ-147)
# --------------------------------------------------------------------------- #
class _BoxFill:
    def embed(self, frame: np.ndarray) -> None:  # noqa: ARG002 - mesma assinatura do real
        return None

    def segment(
        self,
        frame: np.ndarray,
        box: np.ndarray,
        embedding: object = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, float]:
        out = np.zeros(frame.shape, bool)
        x0, y0, x1, y1 = (int(v) for v in box)
        out[y0 + 2 : y1 - 1, x0 + 2 : x1 - 1] = True
        return out, 0.8


@pytest.mark.requirement("REQ-147")
@pytest.mark.parametrize("spacing", [(0.1, 0.1), None])
def test_medsam_adapter_measures_only_with_spacing(spacing: tuple[float, float] | None) -> None:
    from radiologyai.analysis.adapters.us_medsam import MedSAMUSAdapter

    m, _ = load_manifest(MANIFESTS / "medsam-vit-base.yaml")
    adapter = MedSAMUSAdapter(m)
    adapter.segmenter = _BoxFill()
    img = us_image(np.full((100, 120), 90.0), spacing=spacing)
    prompts = (Primitive(type="bbox", xy=((10.0, 10.0), (60.0, 40.0)), label="lesion"),)
    inputs = ModelInputs(images=(img,), exam_type=ExamType.US_BREAST, prompts=prompts)
    assert adapter.applicability(inputs).applicable
    out = adapter.predict(inputs)
    meas = {x.key: x for x in out.measurements}
    assert set(meas) == {"lesion1_area", "lesion1_max_diameter", "lesion1_perp_diameter"}
    if spacing is None:
        assert all(
            x.value is None and x.abstention is AbstentionReason.GEOMETRY for x in meas.values()
        )
    else:
        assert meas["lesion1_area"].value == pytest.approx(47 * 27 * 0.01)
    assert out.labelmaps[0].array.sum() == 47 * 27
