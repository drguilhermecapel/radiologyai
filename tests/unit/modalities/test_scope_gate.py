"""Gate de escopo — o controle de risco para o perigo H-03.

Este é o teste que o sistema legado reprovaria: lá, uma incidência lateral ou
um exame pediátrico caíam em `_analyze_image_fallback()` e recebiam um escore
de patologia calculado por heurística de OpenCV.
"""

from __future__ import annotations

import pytest

from radiologyai.errors import OutOfScopeError
from radiologyai.io.metadata import extract_metadata
from radiologyai.modalities import available_modalities, resolve
from radiologyai.modalities.xr import XRPlugin


@pytest.mark.requirement("REQ-042")
class TestXRScopeGate:
    def test_frontal_adult_accepted(self, cr_dataset):
        report = XRPlugin().validate(extract_metadata(cr_dataset))
        assert report.accepted, report.reasons

    def test_lateral_view_rejected(self, lateral_dataset):
        report = XRPlugin().validate(extract_metadata(lateral_dataset))
        assert not report.accepted
        assert any("incidência" in r for r in report.reasons)

    def test_pediatric_rejected(self, pediatric_dataset):
        report = XRPlugin().validate(extract_metadata(pediatric_dataset))
        assert not report.accepted
        assert any("pediátrica" in r for r in report.reasons)

    def test_wrong_modality_rejected(self, ct_dataset):
        report = XRPlugin().validate(extract_metadata(ct_dataset))
        assert not report.accepted

    def test_rejection_raises_not_degrades(self, lateral_dataset):
        """Recusa é exceção, nunca uma predição degradada."""
        report = XRPlugin().validate(extract_metadata(lateral_dataset))
        with pytest.raises(OutOfScopeError, match="fora do uso pretendido"):
            report.raise_if_rejected()

    def test_accepted_report_does_not_raise(self, cr_dataset):
        XRPlugin().validate(extract_metadata(cr_dataset)).raise_if_rejected()

    def test_missing_age_is_not_a_rejection(self, cr_dataset):
        """Idade ausente é comum em DICOM anonimizado e não deve recusar."""
        del cr_dataset.PatientAge
        report = XRPlugin().validate(extract_metadata(cr_dataset))
        assert report.accepted, report.reasons

    def test_too_small_image_rejected(self, cr_dataset):
        cr_dataset.Rows = 64
        cr_dataset.Columns = 64
        report = XRPlugin().validate(extract_metadata(cr_dataset))
        assert not report.accepted
        assert any("mínimo" in r for r in report.reasons)


class TestRegistry:
    def test_all_four_modalities_registered(self):
        codes = {p.code for p in available_modalities().values()}
        assert codes == {"XR", "CT", "MR", "US"}

    def test_resolve_by_dicom_modality(self):
        assert resolve("CR").code == "XR"
        assert resolve("DX").code == "XR"
        assert resolve("CT").code == "CT"

    def test_resolve_is_case_insensitive(self):
        assert resolve("cr").code == "XR"

    def test_unknown_modality_raises(self):
        with pytest.raises(OutOfScopeError, match="não suportada"):
            resolve("XA")

    def test_only_xr_is_implemented(self):
        by_code = {p.code: p for p in available_modalities().values()}
        assert by_code["XR"].implemented is True
        for code in ("CT", "MR", "US"):
            assert by_code[code].implemented is False

    def test_3d_modalities_declare_dimensionality(self):
        """O caminho 3D existe no sistema de tipos desde a Fase 1."""
        by_code = {p.code: p for p in available_modalities().values()}
        assert by_code["CT"].dimensionality == "3d"
        assert by_code["MR"].dimensionality == "3d"
        assert by_code["US"].dimensionality == "cine"
        assert by_code["XR"].dimensionality == "2d"


class TestUnimplementedModalities:
    def test_unimplemented_never_accepts(self, ct_dataset):
        report = resolve("CT").validate(extract_metadata(ct_dataset))
        assert not report.accepted
        assert any("não implementada" in r for r in report.reasons)

    def test_unimplemented_preprocess_raises(self, ct_dataset):
        import numpy as np

        with pytest.raises(NotImplementedError, match="não implementado"):
            resolve("CT").preprocess(np.zeros((4, 4)), extract_metadata(ct_dataset))
