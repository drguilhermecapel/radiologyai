"""Janelamento — presets e normalização."""

from __future__ import annotations

import numpy as np
import pytest

from radiologyai.io.windowing import WINDOW_PRESETS, apply_preset, apply_window


class TestApplyWindow:
    def test_normalizes_to_unit_range(self):
        arr = np.linspace(-1000, 1000, 100, dtype=np.float32)
        out = apply_window(arr, center=0.0, width=2000.0)
        assert out.min() == pytest.approx(0.0)
        assert out.max() == pytest.approx(1.0, abs=0.02)

    def test_output_is_float32(self):
        assert apply_window(np.zeros(10), 0.0, 100.0).dtype == np.float32

    def test_clips_outside_window(self):
        arr = np.array([-5000.0, 0.0, 5000.0], dtype=np.float32)
        out = apply_window(arr, center=0.0, width=100.0)
        assert out[0] == pytest.approx(0.0)
        assert out[2] == pytest.approx(1.0)

    def test_zero_width_raises(self):
        with pytest.raises(ValueError, match="width deve ser > 0"):
            apply_window(np.zeros(10), 0.0, 0.0)

    def test_negative_width_raises(self):
        with pytest.raises(ValueError, match="width deve ser > 0"):
            apply_window(np.zeros(10), 0.0, -100.0)


class TestPresets:
    def test_lung_window_matches_clinical_values(self):
        p = WINDOW_PRESETS["ct_lung"]
        assert (p.center, p.width) == (-600.0, 1500.0)

    def test_brain_window_matches_clinical_values(self):
        p = WINDOW_PRESETS["ct_brain"]
        assert (p.center, p.width) == (40.0, 80.0)

    def test_lung_window_separates_air_from_soft_tissue(self):
        """Ar (-1000 HU) e partes moles (+50 HU) devem cair em extremos distintos."""
        arr = np.array([-1000.0, 50.0], dtype=np.float32)
        out = apply_preset(arr, "ct_lung")
        assert out[1] - out[0] > 0.3

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="preset desconhecido"):
            apply_preset(np.zeros(4), "inexistente")

    @pytest.mark.parametrize("name", [k for k in WINDOW_PRESETS if k.startswith("ct_")])
    def test_all_ct_presets_are_usable(self, name):
        out = apply_preset(np.linspace(-1024, 3000, 50, dtype=np.float32), name)
        assert 0.0 <= out.min() <= out.max() <= 1.0
