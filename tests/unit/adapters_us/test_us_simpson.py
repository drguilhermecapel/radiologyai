"""Método dos discos (Simpson) em máscaras sintéticas com volume conhecido."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from us_support import half_ellipse

from radiologyai.analysis.adapters.echo_medsam2 import lv_measurements
from radiologyai.analysis.adapters.echo_simpson import (
    SimpsonError,
    biplane_volume,
    disc_geometry,
    ejection_fraction,
    single_plane_volume,
    trace_contour,
)
from radiologyai.analysis.types import AbstentionReason

pytestmark = pytest.mark.requirement("REQ-145")


def prolate_half(a: float, b1: float, b2: float) -> float:
    return 2.0 / 3.0 * math.pi * b1 * b2 * a / 1000.0  # mL


@pytest.mark.parametrize("spacing", [(0.3, 0.3), (0.5, 0.25), (0.2, 0.4)])
def test_single_plane_volume_matches_half_spheroid(spacing: tuple[float, float]) -> None:
    mask = half_ellipse(80.0, 25.0, spacing)
    geom = disc_geometry(mask, spacing)
    vol = single_plane_volume(geom).volume_ml
    assert vol == pytest.approx(prolate_half(80, 25, 25), rel=0.03)
    assert geom.long_axis_mm == pytest.approx(80.0, rel=0.03)
    assert geom.n_discs == 20


def test_biplane_volume_matches_elliptic_half_spheroid() -> None:
    sp = (0.3, 0.3)
    a4c = disc_geometry(half_ellipse(80.0, 25.0, sp), sp)
    a2c = disc_geometry(half_ellipse(80.0, 18.0, sp), sp)
    res = biplane_volume(a4c, a2c)
    assert res.volume_ml == pytest.approx(prolate_half(80, 25, 18), rel=0.03)
    assert "biplanar" in res.method


def test_spacing_changes_volume_cubically() -> None:
    mask = half_ellipse(80.0, 25.0, (0.3, 0.3))
    v1 = single_plane_volume(disc_geometry(mask, (0.3, 0.3))).volume_ml
    v2 = single_plane_volume(disc_geometry(mask, (0.6, 0.6))).volume_ml
    assert v2 / v1 == pytest.approx(8.0, rel=0.02)


def test_apex_down_orientation_is_supported() -> None:
    sp = (0.3, 0.3)
    up = single_plane_volume(disc_geometry(half_ellipse(70, 22, sp), sp)).volume_ml
    down = single_plane_volume(disc_geometry(half_ellipse(70, 22, sp, apex_up=False), sp)).volume_ml
    assert down == pytest.approx(up, rel=0.01)


def test_ejection_fraction_from_known_volumes() -> None:
    assert ejection_fraction(120.0, 60.0) == pytest.approx(50.0)
    with pytest.raises(SimpsonError):
        ejection_fraction(0.0, 10.0)
    with pytest.raises(SimpsonError):
        ejection_fraction(100.0, -1.0)


def test_empty_mask_and_bad_spacing_raise() -> None:
    with pytest.raises(SimpsonError, match="vazia"):
        disc_geometry(np.zeros((50, 50), bool), (0.3, 0.3))
    with pytest.raises(SimpsonError, match="espaçamento"):
        disc_geometry(half_ellipse(40, 10, (0.3, 0.3)), (0.0, 0.3))


def test_trace_contour_is_closed_boundary() -> None:
    m = np.zeros((10, 10), bool)
    m[2:6, 3:8] = True
    c = trace_contour(m)
    assert len(c) == 2 * (4 + 5) - 4  # perímetro de pixels de um retângulo 4x5
    assert {tuple(p) for p in c.astype(int)} <= {(r, k) for r in range(2, 6) for k in range(3, 8)}


def _traced(ed: float, es: float, spacing: tuple[float, float] | None) -> dict:
    sp = (0.3, 0.3)
    out = {}
    for view, b in (("a4c", 1.0), ("a2c", 0.8)):
        out[(view, "ed")] = (half_ellipse(80, 25 * b * ed, sp), spacing, 0)
        out[(view, "es")] = (half_ellipse(65, 25 * b * es, sp), spacing, 5)
    return out


def test_lv_measurements_biplane_ef_and_geometry() -> None:
    meas = {m.key: m for m in lv_measurements(_traced(1.0, 0.7, (0.3, 0.3)))}
    edv = prolate_half(80, 25, 20)
    esv = prolate_half(65, 25 * 0.7, 20 * 0.7)
    assert meas["lv_edv"].value == pytest.approx(edv, rel=0.04)
    assert meas["lv_esv"].value == pytest.approx(esv, rel=0.05)
    assert meas["lv_ef"].value == pytest.approx(100 * (edv - esv) / edv, abs=2.0)
    assert meas["lv_ef"].preconditions["views"] == "A4C+A2C"
    area = meas["lv_cavity_area_a4c_es"]
    assert area.geometry
    assert area.geometry[0].frame == 5
    assert any(p.label == "a4c:es:apex" for p in area.geometry)


def test_lv_measurements_single_plane_without_a2c() -> None:
    traced = {k: v for k, v in _traced(1.0, 0.7, (0.3, 0.3)).items() if k[0] == "a4c"}
    meas = {m.key: m for m in lv_measurements(traced)}
    assert meas["lv_ef"].value is not None
    assert meas["lv_ef"].preconditions["views"] == "A4C"


def test_lv_measurements_abstain_without_spacing() -> None:
    meas = {m.key: m for m in lv_measurements(_traced(1.0, 0.7, None))}
    for key in ("lv_edv", "lv_esv", "lv_ef"):
        assert meas[key].value is None
        assert meas[key].abstention is AbstentionReason.GEOMETRY


def test_lv_measurements_abstain_on_empty_mask() -> None:
    traced = _traced(1.0, 0.7, (0.3, 0.3))
    mask, sp, fr = traced[("a4c", "es")]
    traced[("a4c", "es")] = (np.zeros_like(mask), sp, fr)
    meas = {m.key: m for m in lv_measurements(traced)}
    assert meas["lv_ef"].value is None
    assert meas["lv_ef"].abstention is AbstentionReason.QUALITY


def test_shared_max_convention_uses_one_length_for_ed_and_es() -> None:
    traced = _traced(1.0, 0.7, (0.3, 0.3))
    per = {m.key: m for m in lv_measurements(traced)}
    shared = {m.key: m for m in lv_measurements(traced, length_convention="shared_max")}
    assert shared["lv_edv"].value == pytest.approx(per["lv_edv"].value, rel=1e-9)
    # ES (eixo 65 mm) passa a usar o eixo do ED (80 mm): VSF maior, FE menor
    assert shared["lv_esv"].value > per["lv_esv"].value
    assert shared["lv_ef"].value < per["lv_ef"].value
    assert "convenção CAMUS" in shared["lv_ef"].method


CAMUS_FULL = Path("/root/medai-cache/datasets/camus-full")


@pytest.mark.skipif(
    not (CAMUS_FULL / "patient0001").is_dir() or importlib.util.find_spec("nibabel") is None,
    reason="CAMUS local ausente",
)
def test_reference_masks_reproduce_camus_info_ef() -> None:
    """Sobre as máscaras do especialista, a convenção shared_max reproduz a FE do Info_*.cfg."""
    import nibabel as nib

    errors = []
    for k in range(1, 21):
        pid = f"patient{k:04d}"
        traced = {}
        for v, role in (("4CH", "a4c"), ("2CH", "a2c")):
            for inst in ("ED", "ES"):
                img = nib.load(str(CAMUS_FULL / pid / f"{pid}_{v}_{inst}_gt.nii.gz"))
                z = img.header.get_zooms()
                arr = np.asarray(img.dataobj).T == 1
                traced[(role, inst.lower())] = (arr, (float(z[1]), float(z[0])), 0)
        info = (CAMUS_FULL / pid / "Info_2CH.cfg").read_text().splitlines()
        ef_ref = float(next(x.split(": ")[1] for x in info if x.startswith("EF:")))
        m = {x.key: x for x in lv_measurements(traced, length_convention="shared_max")}
        errors.append(abs(m["lv_ef"].value - ef_ref))
    assert float(np.mean(errors)) < 2.0
    assert max(errors) < 5.0


def test_ef_abstains_when_esv_not_smaller_than_edv() -> None:
    traced = _traced(1.0, 1.3, (0.3, 0.3))  # ES maior que ED: contorno incoerente
    traced[("a4c", "es")] = (half_ellipse(90, 32, (0.3, 0.3)), (0.3, 0.3), 5)
    traced[("a2c", "es")] = (half_ellipse(90, 26, (0.3, 0.3)), (0.3, 0.3), 5)
    meas = {m.key: m for m in lv_measurements(traced)}
    assert meas["lv_esv"].value is not None
    assert meas["lv_ef"].value is None
    assert meas["lv_ef"].abstention is AbstentionReason.QUALITY
    assert "inconsistentes" in (meas["lv_ef"].note or "")
