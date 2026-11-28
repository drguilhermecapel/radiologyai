"""Des-identificação PS3.15 Annex E aplicada dentro de sequências (perfil básico + opções)."""

from __future__ import annotations

import pytest
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from radiologyai.io.deident import UID_ROOT, DeidentificationProfile, deidentify

KEY = b"chave-de-teste-nao-usar-em-producao"
ORIGINAL_REF_UID = "1.2.3.4.5"
CR_CLASS = "1.2.840.10008.5.1.4.1.1.1"


def _item(**attrs: object) -> Dataset:
    item = Dataset()
    for key, value in attrs.items():
        setattr(item, key, value)
    return item


@pytest.fixture
def risky(cr_dataset: Dataset) -> Dataset:
    ds = cr_dataset
    ds.AcquisitionDateTime = "20240315101112"
    ds.InstanceCreationDate = "20240315"
    ds.InstanceCreationTime = "101112"
    ds.StudyDescription = "TORAX JOAO SILVA PA"
    ds.SeriesDescription = "PA 12345678"
    ds.PhysiciansOfRecord = "DRA^MARIA"
    ds.PatientBirthName = "SOUZA^JOAO"
    ds.RequestAttributesSequence = Sequence(
        [_item(AccessionNumber="ACC999", RequestedProcedureID="RP-77")]
    )
    ds.SourceImageSequence = Sequence(
        [_item(ReferencedSOPClassUID=CR_CLASS, ReferencedSOPInstanceUID=ORIGINAL_REF_UID)]
    )
    ds.ReferencedImageSequence = Sequence(
        [
            _item(
                ReferencedSOPClassUID=CR_CLASS,
                ReferencedSOPInstanceUID="1.2.3.4.6",
                # pessoa e data dentro de sequência: precisam sair também
                ContentCreatorName="OPERADOR^X",
                ContentDate="20240316",
            )
        ]
    )
    ds.add_new(0x00091010, "LO", "privado no topo")
    ds.SourceImageSequence[0].add_new(0x00111010, "LO", "privado aninhado")
    return ds


def _walk(ds: Dataset):
    for elem in ds:
        yield elem
        if elem.VR == "SQ":
            for item in elem.value:
                yield from _walk(item)


@pytest.mark.requirement("REQ-175")
def test_reviewed_scenario_values_do_not_survive(risky: Dataset) -> None:
    out, _ = deidentify(risky, DeidentificationProfile(key=KEY))
    text = repr(out)
    for leaked in ("JOAO", "SILVA", "MARIA", "SOUZA", "ACC999", "RP-77", "OPERADOR", "12345678"):
        assert leaked not in text, f"{leaked!r} sobreviveu à des-identificação"
    assert ORIGINAL_REF_UID not in text
    assert "RequestAttributesSequence" not in out
    assert "PhysiciansOfRecord" not in out
    assert "PatientBirthName" not in out
    # datas só com o ano; horas vazias (opção Modified Dates)
    assert out.AcquisitionDateTime == "20240101"
    assert out.InstanceCreationDate == "20240101"
    assert out.InstanceCreationTime == ""
    assert out.ReferencedImageSequence[0].ContentDate == "20240101"


@pytest.mark.requirement("REQ-175")
def test_nested_uids_remapped_but_classes_kept(risky: Dataset) -> None:
    profile = DeidentificationProfile(key=KEY)
    out, _ = deidentify(risky, profile)
    ref = out.SourceImageSequence[0]
    assert ref.ReferencedSOPClassUID == CR_CLASS
    assert ref.ReferencedSOPInstanceUID == profile.pseudo_uid(ORIGINAL_REF_UID)
    assert str(ref.ReferencedSOPInstanceUID).startswith(f"{UID_ROOT}.")
    assert out.SOPClassUID == CR_CLASS


@pytest.mark.requirement("REQ-175")
def test_no_person_name_date_or_private_tag_anywhere(risky: Dataset) -> None:
    out, _ = deidentify(risky, DeidentificationProfile(key=KEY))
    for elem in _walk(out):
        assert not elem.tag.is_private, f"tag privada {elem.tag} sobreviveu"
        if elem.VR == "PN":
            assert str(elem.value or "") == "", f"{elem.keyword} com nome"
        if elem.VR in ("DA", "DT") and elem.value:
            assert str(elem.value).endswith("0101"), f"{elem.keyword} com data completa"
        if elem.VR == "TM":
            assert str(elem.value or "") == ""


@pytest.mark.requirement("REQ-175")
def test_descriptions_keep_only_vocabulary_words(risky: Dataset) -> None:
    out, _ = deidentify(risky, DeidentificationProfile(key=KEY))
    assert out.StudyDescription == "TORAX PA"
    assert out.SeriesDescription == "PA"
    assert out.BodyPartExamined == "CHEST"  # não listado na tabela E.1-1: mantido


@pytest.mark.requirement("REQ-175")
def test_identity_removed_flag_comes_with_declared_profile(risky: Dataset) -> None:
    out, _ = deidentify(risky, DeidentificationProfile(key=KEY))
    assert out.PatientIdentityRemoved == "YES"
    codes = {
        (c.CodingSchemeDesignator, c.CodeValue) for c in out.DeidentificationMethodCodeSequence
    }
    assert ("DCM", "113100") in codes  # perfil básico
    assert out.LongitudinalTemporalInformationModified == "MODIFIED"
    # características do paciente mantidas pela opção declarada
    assert (out.PatientSex, out.PatientAge) == ("M", "045Y")


@pytest.mark.requirement("REQ-175")
def test_profile_codes_exist_in_the_dicom_dictionary() -> None:
    from pydicom.sr.codedict import codes

    from radiologyai.io.deident import METHOD_CODES

    known = {
        str(c.value): str(c.meaning)
        for c in (
            codes.DCM.BasicApplicationConfidentialityProfile,
            codes.DCM.RetainLongitudinalTemporalInformationModifiedDatesOption,
            codes.DCM.RetainPatientCharacteristicsOption,
        )
    }
    assert dict(METHOD_CODES) == known
