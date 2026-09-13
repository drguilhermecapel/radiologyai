"""Des-identificação — determinismo e cobertura de tags."""

from __future__ import annotations

import pytest

from radiologyai.io.deident import (
    TAGS_TO_PSEUDONYMIZE,
    TAGS_TO_PSEUDONYMIZE_UID,
    UID_ROOT,
    DeidentificationProfile,
    deidentify,
    has_burned_in_annotation,
)

KEY = b"chave-de-teste-nao-usar-em-producao"


@pytest.fixture
def profile() -> DeidentificationProfile:
    return DeidentificationProfile(key=KEY)


@pytest.mark.requirement("REQ-010")
class TestDeterminism:
    """O legado usava hash() builtin, salgado por processo (HONEST_STATUS §6)."""

    def test_same_input_same_pseudo_id(self, profile):
        assert profile.pseudo_id("PACIENTE-1") == profile.pseudo_id("PACIENTE-1")

    def test_different_input_different_pseudo_id(self, profile):
        assert profile.pseudo_id("PACIENTE-1") != profile.pseudo_id("PACIENTE-2")

    def test_stable_across_profile_instances(self):
        a = DeidentificationProfile(key=KEY)
        b = DeidentificationProfile(key=KEY)
        assert a.pseudo_id("X") == b.pseudo_id("X")

    def test_different_key_different_pseudo_id(self):
        a = DeidentificationProfile(key=KEY)
        b = DeidentificationProfile(key=b"outra-chave")
        assert a.pseudo_id("X") != b.pseudo_id("X")

    def test_pseudo_id_is_not_reversible_plaintext(self, profile):
        assert "PACIENTE-1" not in profile.pseudo_id("PACIENTE-1")


@pytest.mark.requirement("REQ-011")
class TestTagRemoval:
    def test_direct_identifiers_removed(self, cr_dataset, profile):
        out, _ = deidentify(cr_dataset, profile)
        for tag in ("PatientName", "PatientBirthDate", "AccessionNumber", "InstitutionName"):
            assert getattr(out, tag, "") == "", f"{tag} deveria ter sido removida"

    def test_identifiers_are_pseudonymized(self, cr_dataset, profile):
        out, _ = deidentify(cr_dataset, profile)
        for tag in TAGS_TO_PSEUDONYMIZE:
            original = getattr(cr_dataset, tag, None)
            if original:
                assert getattr(out, tag) != original
                assert str(getattr(out, tag)).startswith("ANON-")

    def test_uids_are_pseudonymized(self, cr_dataset, profile):
        out, _ = deidentify(cr_dataset, profile)
        for tag in TAGS_TO_PSEUDONYMIZE_UID:
            original = getattr(cr_dataset, tag, None)
            if original:
                assert getattr(out, tag) != original

    def test_pseudo_uids_remain_valid_dicom_uids(self, cr_dataset, profile):
        """VR 'UI' admite apenas dígitos e pontos, com no máximo 64 caracteres."""
        out, _ = deidentify(cr_dataset, profile)
        for tag in TAGS_TO_PSEUDONYMIZE_UID:
            value = str(getattr(out, tag, "") or "")
            if not value:
                continue
            assert value.startswith(f"{UID_ROOT}.")
            assert len(value) <= 64
            assert set(value) <= set("0123456789."), f"{tag} não é UID válido: {value}"

    def test_pseudo_uid_is_deterministic(self, profile):
        assert profile.pseudo_uid("1.2.3.4") == profile.pseudo_uid("1.2.3.4")
        assert profile.pseudo_uid("1.2.3.4") != profile.pseudo_uid("1.2.3.5")

    def test_patient_id_pseudo_returned(self, cr_dataset, profile):
        out, pseudo = deidentify(cr_dataset, profile)
        assert pseudo is not None
        assert out.PatientID == pseudo

    def test_dates_truncated_to_year(self, cr_dataset, profile):
        out, _ = deidentify(cr_dataset, profile)
        assert out.StudyDate == "20260101"
        assert out.StudyTime == ""

    def test_marks_identity_removed(self, cr_dataset, profile):
        out, _ = deidentify(cr_dataset, profile)
        assert out.PatientIdentityRemoved == "YES"
        assert "PS3.15" in out.DeidentificationMethod

    def test_original_dataset_untouched(self, cr_dataset, profile):
        original_name = str(cr_dataset.PatientName)
        deidentify(cr_dataset, profile)
        assert str(cr_dataset.PatientName) == original_name

    def test_clinically_useful_tags_preserved(self, cr_dataset, profile):
        """Sexo, idade e modalidade sobrevivem — são necessários para subgrupos."""
        out, _ = deidentify(cr_dataset, profile)
        assert out.PatientSex == "M"
        assert out.PatientAge == "045Y"
        assert out.Modality == "CR"


class TestBurnedInAnnotation:
    def test_absent_tag_returns_none_not_false(self, cr_dataset):
        """Ausência da tag NÃO prova ausência de texto gravado no pixel."""
        assert has_burned_in_annotation(cr_dataset) is None

    def test_yes_detected(self, cr_dataset):
        cr_dataset.BurnedInAnnotation = "YES"
        assert has_burned_in_annotation(cr_dataset) is True
