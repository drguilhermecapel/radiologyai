"""Biomarcadores de TC em volumes sintéticos de geometria e HU conhecidas."""

from __future__ import annotations

import math

import numpy as np
import pytest
from ctmr_support import context, ct_meta, grid_mm

from radiologyai.analysis.biomarkers import ct
from radiologyai.analysis.types import AbstentionReason

SP = (2.0, 1.0, 1.0)  # (z, y, x) mm


def _block(shape, sl, value=1):
    arr = np.zeros(shape, dtype=np.uint8)
    arr[sl] = value
    return arr


# --------------------------------------------------------------------------- volumes
@pytest.mark.requirement("REQ-136")
class TestOrganVolume:
    def test_known_cube_volume(self):
        lm = _block((40, 50, 50), np.s_[10:30, 10:30, 10:30])
        ctx = context(np.zeros(lm.shape), lm, {"spleen": 1}, SP)
        m = ct.organ_volume(ctx, "spleen_volume_ml", ("spleen",))
        assert m.value == pytest.approx(20 * 20 * 20 * 2.0 / 1000.0)  # 16 mL
        assert m.unit == "mL"
        assert m.abstention is None

    def test_truncated_organ_abstains(self):
        lm = _block((40, 50, 50), np.s_[0:30, 10:30, 10:30])  # encosta no 1º corte
        ctx = context(np.zeros(lm.shape), lm, {"spleen": 1}, SP)
        m = ct.organ_volume(ctx, "spleen_volume_ml", ("spleen",))
        assert m.value is None
        assert m.abstention is AbstentionReason.ACQUISITION
        assert "cortada" in (m.note or "")

    def test_absent_organ_abstains_not_zero(self):
        lm = np.zeros((10, 10, 10), dtype=np.uint8)
        ctx = context(np.zeros(lm.shape), lm, {"spleen": 1}, SP)
        m = ct.organ_volume(ctx, "spleen_volume_ml", ("spleen",))
        assert m.value is None
        assert m.abstention is AbstentionReason.ACQUISITION


# --------------------------------------------------------------------------- L1
def _l1_volume(core_hu=150.0, shell_hu=700.0, z_range=(10, 30)):
    shape = (40, 80, 80)
    z, y, x = grid_mm(shape, SP)
    r = np.hypot(y - 40.0, x - 40.0)
    inside = (r <= 18.0) & (z >= z_range[0] * 2.0) & (z < z_range[1] * 2.0)
    lm = inside.astype(np.uint8)
    hu = np.full(shape, -50.0)
    hu[inside] = core_hu
    hu[inside & (r > 15.0)] = shell_hu  # cortical de 3 mm
    return hu, lm


@pytest.mark.requirement("REQ-133")
class TestL1Trabecular:
    def test_core_excludes_cortex(self):
        hu, lm = _l1_volume()
        m = ct.l1_trabecular_hu(context(hu, lm, {"vertebrae_L1": 1}, SP))
        assert m.value == pytest.approx(150.0, abs=0.5)
        assert m.unit == ct.HU

    @pytest.mark.parametrize(
        ("override", "reason"),
        [
            ({"kvp": 100.0}, AbstentionReason.ACQUISITION),
            ({"kvp": None}, AbstentionReason.ACQUISITION),
            ({"patient_age_years": None}, AbstentionReason.POPULATION),
            ({"patient_age_years": 12.0}, AbstentionReason.POPULATION),
        ],
    )
    def test_preconditions_abstain(self, override, reason):
        hu, lm = _l1_volume()
        m = ct.l1_trabecular_hu(context(hu, lm, {"vertebrae_L1": 1}, SP, ct_meta(**override)))
        assert m.value is None
        assert m.abstention is reason

    def test_contrast_is_allowed_for_l1(self):
        hu, lm = _l1_volume()
        meta = ct_meta(contrast_phase="portal_venous", contrast_agent="IOPAMIDOL")
        m = ct.l1_trabecular_hu(context(hu, lm, {"vertebrae_L1": 1}, SP, meta))
        assert m.value == pytest.approx(150.0, abs=0.5)

    def test_l1_cut_by_fov_abstains(self):
        hu, lm = _l1_volume(z_range=(0, 20))
        m = ct.l1_trabecular_hu(context(hu, lm, {"vertebrae_L1": 1}, SP))
        assert m.abstention is AbstentionReason.ACQUISITION


# --------------------------------------------------------------------------- fígado
def _liver_spleen(liver_hu=35.0, spleen_hu=50.0):
    shape = (40, 100, 100)
    lm = np.zeros(shape, dtype=np.uint8)
    lm[5:35, 10:70, 10:60] = 1
    lm[10:30, 20:50, 70:95] = 2
    hu = np.full(shape, -100.0)
    hu[lm == 1] = liver_hu
    hu[lm == 2] = spleen_hu
    return hu, lm


@pytest.mark.requirement("REQ-133")
class TestLiver:
    names = {"liver": 1, "spleen": 2}

    def test_unenhanced_values(self):
        hu, lm = _liver_spleen()
        ctx = context(hu, lm, self.names, SP)
        assert ct.liver_attenuation(ctx, "none").value == pytest.approx(35.0)
        assert ct.liver_spleen_difference(ctx).value == pytest.approx(-15.0)
        pv = ct.liver_attenuation(ctx, "portal_venous")
        assert pv.value is None
        assert pv.abstention is AbstentionReason.ACQUISITION

    @pytest.mark.parametrize(
        "override",
        [
            {"contrast_phase": None},  # fase desconhecida
            {"contrast_phase": "unknown"},
            {"contrast_phase": "none", "contrast_agent": "IOPAMIDOL"},  # conflito
            {"contrast_phase": "arterial", "contrast_agent": "IOPAMIDOL"},
            {"kvp": 140.0},
        ],
    )
    def test_unenhanced_requires_confirmed_phase_and_120kv(self, override):
        hu, lm = _liver_spleen()
        ctx = context(hu, lm, self.names, SP, ct_meta(**override))
        for m in (ct.liver_attenuation(ctx, "none"), ct.liver_spleen_difference(ctx)):
            assert m.value is None
            assert m.abstention is AbstentionReason.ACQUISITION

    def test_portal_venous_only_in_confirmed_phase(self):
        hu, lm = _liver_spleen(liver_hu=85.0, spleen_hu=110.0)
        meta = ct_meta(contrast_phase="portal_venous", contrast_agent="IOPAMIDOL")
        ctx = context(hu, lm, self.names, SP, meta)
        assert ct.liver_attenuation(ctx, "portal_venous").value == pytest.approx(85.0)
        assert ct.liver_attenuation(ctx, "none").abstention is AbstentionReason.ACQUISITION

    def test_erosion_removes_capsule(self):
        hu, lm = _liver_spleen()
        hu[(lm == 1) & (np.roll(lm, 2, axis=2) != 1)] = 200.0  # borda brilhante
        ctx = context(hu, lm, self.names, SP)
        assert ct.liver_attenuation(ctx, "none").value == pytest.approx(35.0)


# --------------------------------------------------------------------------- enfisema
def _lungs(frac_low=0.10):
    shape = (60, 60, 60)
    lm = np.zeros(shape, dtype=np.uint8)
    lm[5:55, 5:55, 5:30] = 1
    lm[5:55, 5:55, 30:55] = 3
    hu = np.full(shape, -850.0)
    idx = np.flatnonzero(lm.ravel() > 0)
    n_low = round(frac_low * idx.size)
    hu.ravel()[idx[:: max(1, idx.size // n_low)][:n_low]] = -980.0
    return hu, lm


@pytest.mark.requirement("REQ-133")
class TestEmphysema:
    names = {name: i + 1 for i, name in enumerate(ct.LUNG_LOBES)}

    def test_known_fraction(self):
        hu, lm = _lungs(0.10)
        m = ct.laa950_percent(context(hu, lm, self.names, (1.0, 1.0, 1.0)))
        assert m.value == pytest.approx(10.0, abs=0.05)
        assert m.unit == "%"

    @pytest.mark.parametrize(
        "override",
        [
            {"slice_thickness_mm": 5.0},
            {"convolution_kernel": "B70f"},
            {"convolution_kernel": "XYZ"},
            {"contrast_phase": None},
            {"series_description": "TORAX EXPIRACAO"},
        ],
    )
    def test_preconditions(self, override):
        hu, lm = _lungs()
        ctx = context(hu, lm, self.names, (1.0, 1.0, 1.0), ct_meta(**override))
        m = ct.laa950_percent(ctx)
        assert m.value is None
        assert m.abstention is AbstentionReason.ACQUISITION


# --------------------------------------------------------------------------- aorta
def _tube(shape, spacing, p0, p1, radius):
    """Cilindro sólido entre p0 e p1 (mm, ordem z, y, x)."""
    z, y, x = grid_mm(shape, spacing)
    pts = np.stack([z, y, x], axis=-1)
    a, b = np.asarray(p0, float), np.asarray(p1, float)
    d = b - a
    t = np.clip(((pts - a) @ d) / (d @ d), 0.0, 1.0)
    closest = a + t[..., None] * d
    return np.linalg.norm(pts - closest, axis=-1) <= radius


@pytest.mark.requirement("REQ-135")
class TestAorta:
    def test_oblique_abdominal_tube_measured_perpendicular(self):
        shape, sp = (100, 64, 200), (2.0, 1.0, 1.0)
        # eixo inclinado 30 graus: cresce 115 mm em x ao longo de 160 mm em z
        aorta = _tube(shape, sp, (20.0, 32.0, 40.0), (180.0, 32.0, 132.4), 12.0)
        lm = aorta.astype(np.uint8)
        lm[95:99, 50:60, 180:190] = 2  # L1 acima de tudo: toda a aorta é "abdominal"
        ctx = context(np.zeros(shape), lm, {"aorta": 1, "vertebrae_L1": 2}, sp)
        out = {m.key: m for m in ct.aortic_diameters(ctx)}
        abd = out["aorta_abdominal_max_diameter_mm"]
        assert abd.value == pytest.approx(24.0, abs=1.5)
        axial = 24.0 / math.cos(math.radians(30.0))
        assert abd.value < axial - 2.0  # não é o diâmetro axial (27,7 mm)
        assert abd.geometry
        assert abd.geometry[0].type == "line"
        assert out["aorta_ascending_max_diameter_mm"].value is None

    def test_thoracic_ascending_and_descending(self):
        shape, sp = (100, 160, 128), (2.0, 1.0, 1.0)
        asc = _tube(shape, sp, (40.0, 50.0, 64.0), (164.0, 50.0, 64.0), 20.0)
        desc = _tube(shape, sp, (10.0, 110.0, 64.0), (164.0, 110.0, 64.0), 12.0)
        arch = _tube(shape, sp, (164.0, 50.0, 64.0), (164.0, 110.0, 64.0), 12.0)
        lm = (asc | desc | arch).astype(np.uint8)
        lm[10:16, 140:150, 60:70] = 2  # T12
        ctx = context(np.zeros(shape), lm, {"aorta": 1, "vertebrae_T12": 2}, sp)
        out = {m.key: m for m in ct.aortic_diameters(ctx)}
        assert out["aorta_ascending_max_diameter_mm"].value == pytest.approx(40.0, abs=1.5)
        assert out["aorta_descending_thoracic_max_diameter_mm"].value == pytest.approx(
            24.0, abs=1.5
        )

    def test_caudal_cut_abstains_abdominal(self):
        shape, sp = (60, 64, 64), (2.0, 1.0, 1.0)
        aorta = _tube(shape, sp, (-10.0, 32.0, 32.0), (100.0, 32.0, 32.0), 12.0)
        lm = aorta.astype(np.uint8)
        lm[55:59, 50:60, 50:60] = 2
        ctx = context(np.zeros(shape), lm, {"aorta": 1, "vertebrae_L1": 2}, sp)
        abd = {m.key: m for m in ct.aortic_diameters(ctx)}["aorta_abdominal_max_diameter_mm"]
        assert abd.value is None
        assert abd.abstention is AbstentionReason.ACQUISITION

    def test_missing_landmark_abstains_abdominal(self):
        shape, sp = (60, 64, 64), (2.0, 1.0, 1.0)
        lm = _tube(shape, sp, (10.0, 32.0, 32.0), (100.0, 32.0, 32.0), 12.0).astype(np.uint8)
        ctx = context(np.zeros(shape), lm, {"aorta": 1}, sp)
        out = ct.aortic_diameters(ctx)
        assert all(m.value is None for m in out)

    def test_fragmented_mask_abstains(self):
        shape, sp = (60, 64, 64), (2.0, 1.0, 1.0)
        a = _tube(shape, sp, (10.0, 32.0, 32.0), (40.0, 32.0, 32.0), 10.0)
        b = _tube(shape, sp, (75.0, 32.0, 32.0), (100.0, 32.0, 32.0), 10.0)
        lm = (a | b).astype(np.uint8)
        ctx = context(np.zeros(shape), lm, {"aorta": 1}, sp)
        out = ct.aortic_diameters(ctx)
        assert all(m.abstention is AbstentionReason.QUALITY for m in out)


# --------------------------------------------------------------------------- Evans
def _head(horn_width_mm=40.0, skull_width_mm=140.0, touch=False):
    shape, sp = (30, 200, 200), (3.0, 1.0, 1.0)
    z, y, x = grid_mm(shape, sp)
    cx = 100.0 if not touch else 60.0
    brain = (((x - cx) / (skull_width_mm / 2)) ** 2 + ((y - 100.0) / 80.0) ** 2) <= 1.0
    brain &= (z >= 6.0) & (z <= 80.0)
    horns = (np.abs(x - cx) <= horn_width_mm / 2) & (np.abs(y - 70.0) <= 10.0)
    horns &= (z >= 30.0) & (z <= 50.0)
    horns &= np.abs(x - cx) >= 3.0  # dois cornos separados pelo septo
    return shape, sp, brain, horns


@pytest.mark.requirement("REQ-138")
class TestEvans:
    def test_known_ratio(self):
        shape, sp, brain, horns = _head()
        from ctmr_support import affine

        m = ct.evans_index(
            np.zeros(shape, np.float32),
            sp,
            affine(sp),
            ct_meta(),
            horns,
            brain,
            source="ventricle_parts",
        )
        assert m.value == pytest.approx(40.0 / 140.0, abs=0.01)
        assert len(m.geometry) == 2

    def test_requires_adult_and_full_skull(self):
        from ctmr_support import affine

        shape, sp, brain, horns = _head()
        m = ct.evans_index(
            np.zeros(shape, np.float32),
            sp,
            affine(sp),
            ct_meta(patient_age_years=None),
            horns,
            brain,
            source="v",
        )
        assert m.abstention is AbstentionReason.POPULATION
        shape, sp, brain, horns = _head(touch=True)
        m = ct.evans_index(
            np.zeros(shape, np.float32), sp, affine(sp), ct_meta(), horns, brain, source="v"
        )
        assert m.abstention is AbstentionReason.ACQUISITION
