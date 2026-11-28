"""Des-identificação DICOM.

Substitui ``_anonymize_dicom`` do legado, que tinha dois defeitos graves
(HONEST_STATUS.md §6):

1. Cobria 7 tags. Não removia AccessionNumber, StudyDate/Time, InstitutionName,
   os UIDs de Study/Series/SOP, DeviceSerialNumber nem tags privadas.
2. Gerava pseudo-IDs com o ``hash()`` builtin do Python, que é **salgado por
   processo**: o mesmo paciente recebia identificador diferente a cada execução,
   impossibilitando manter um split estável por paciente.

Aqui os pseudo-IDs vêm de HMAC-SHA256 com chave persistente, portanto são
determinísticos entre execuções e entre máquinas, e não reversíveis sem a chave.

Escopo declarado: perfil *Basic Application Level Confidentiality* (DICOM
PS3.15 Annex E, tabela E.1-1) aplicado RECURSIVAMENTE (dentro de toda
sequência), com as opções *Retain Patient Characteristics* (sexo, idade,
peso, altura) e *Retain Longitudinal Temporal Information with Modified Dates*
(datas reduzidas ao ano, horas esvaziadas). Regras por VR cobrem o que a
tabela não lista nominalmente: todo ``PN`` é esvaziado, todo ``UI`` de
instância é remapeado (classes SOP e UIDs do padrão ficam), todo ``DA``/``DT``
fica só com o ano e todo ``TM`` é esvaziado. Descrições (estudo, série,
protocolo...) só mantêm palavras de um vocabulário fixo de anatomia,
incidência e técnica — nome que estivesse nelas some.

``PatientIdentityRemoved = YES`` só é gravado DEPOIS de tudo isso ter sido
aplicado. **Não cobre anotação gravada no pixel (burned-in annotation)** — ver
:func:`has_burned_in_annotation`.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any

# Tags removidas por completo (valor esvaziado).
# PS3.15 Annex E, tabela E.1-1, ações X/Z.
TAGS_TO_REMOVE: tuple[str, ...] = (
    "PatientName",
    "PatientBirthDate",
    "PatientBirthTime",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientMotherBirthName",
    "OtherPatientIDs",
    "OtherPatientNames",
    "OtherPatientIDsSequence",
    "ReferringPhysicianName",
    "ReferringPhysicianAddress",
    "ReferringPhysicianTelephoneNumbers",
    "PerformingPhysicianName",
    "NameOfPhysiciansReadingStudy",
    "OperatorsName",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "DeviceSerialNumber",
    "AccessionNumber",
    "StudyID",
    "RequestingPhysician",
    "RequestedProcedureID",
    "ScheduledPerformingPhysicianName",
    "IssuerOfPatientID",
    "MilitaryRank",
    "EthnicGroup",
    "Occupation",
    "AdditionalPatientHistory",
    "PatientComments",
    "StudyComments",
    "ImageComments",
)

# Identificadores substituídos por pseudo-ID determinístico (ação Z).
TAGS_TO_PSEUDONYMIZE: tuple[str, ...] = ("PatientID",)

# UIDs também são pseudonimizados, mas precisam permanecer UIDs DICOM válidos
# (VR "UI": apenas dígitos e pontos, no máximo 64 caracteres). Um pseudo-ID
# alfanumérico aqui produz um arquivo tecnicamente inválido.
TAGS_TO_PSEUDONYMIZE_UID: tuple[str, ...] = (
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SOPInstanceUID",
    "FrameOfReferenceUID",
)

# Raiz OID de uso local. 2.25 é o arco definido para UUIDs em forma decimal
# (ISO/IEC 9834-8), utilizável sem registro de OID próprio.
UID_ROOT = "2.25"

# Datas: mantidas apenas com o ano, para preservar utilidade epidemiológica
# sem permitir reidentificação por combinação data+local.
TAGS_TO_TRUNCATE_DATE: tuple[str, ...] = (
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
    "ContentDate",
)

TAGS_TO_CLEAR_TIME: tuple[str, ...] = (
    "StudyTime",
    "SeriesTime",
    "AcquisitionTime",
    "ContentTime",
)


@dataclass(frozen=True)
class DeidentificationProfile:
    """Configuração de des-identificação.

    Args:
        key: chave HMAC. Deve vir do ambiente em produção, nunca do código.
        remove_private_tags: remove todas as tags privadas (ímpares).
        keep_year_in_dates: mantém ``AAAA0101`` em vez de esvaziar a data.
    """

    key: bytes
    remove_private_tags: bool = True
    keep_year_in_dates: bool = True
    _prefix: str = field(default="ANON", repr=False)

    def _digest(self, value: str) -> bytes:
        return hmac.new(self.key, value.encode("utf-8"), hashlib.sha256).digest()

    def pseudo_id(self, value: str) -> str:
        """Pseudo-ID determinístico e estável entre execuções."""
        return f"{self._prefix}-{self._digest(value).hex()[:16]}"

    def pseudo_uid(self, value: str) -> str:
        """Pseudo-UID determinístico que é um UID DICOM válido.

        Deriva um inteiro de 122 bits do HMAC e o expressa sob a raiz ``2.25``,
        respeitando o VR "UI" (dígitos e pontos, <= 64 caracteres).
        """
        n = int.from_bytes(self._digest(value)[:16], "big") >> 6
        return f"{UID_ROOT}.{n}"


def has_burned_in_annotation(ds: Any) -> bool | None:
    """Lê DICOM (0028,0301) *Burned In Annotation*.

    Devolve ``None`` quando a tag está ausente — que é o caso comum e **não
    significa que não haja anotação gravada**. Esta função não inspeciona pixels.
    Remoção de texto gravado na imagem está fora do escopo desta versão e é um
    risco declarado no arquivo de risco (perigo H-09).
    """
    value = getattr(ds, "BurnedInAnnotation", None)
    if value is None:
        return None
    return str(value).strip().upper() == "YES"


# --------------------------------------------------------------------------- #
# Tabela de ações aplicada em qualquer profundidade (PS3.15 Annex E, E.1-1)
# --------------------------------------------------------------------------- #
#: Ação X (remover). Inclui sequências inteiras cujo conteúdo identifica.
REMOVE_ANYWHERE: frozenset[str] = frozenset(
    {
        "AcquisitionComments", "AcquisitionContextSequence", "AcquisitionProtocolDescription",
        "ActualHumanPerformersSequence", "AdditionalPatientHistory", "AdmissionID",
        "AdmittingDate", "AdmittingDiagnosesCodeSequence", "AdmittingDiagnosesDescription",
        "AdmittingTime", "Allergies", "AuthorObserverSequence", "BranchOfService", "CassetteID",
        "CommentsOnThePerformedProcedureStep", "ContentCreatorIdentificationCodeSequence",
        "ContentSequence", "ContributionDescription", "CountryOfResidence",
        "CurrentPatientLocation", "CustodialOrganizationSequence", "DataSetTrailingPadding",
        "DerivationDescription", "DetectorID", "DeviceSerialNumber", "DigitalSignaturesSequence",
        "DischargeDiagnosisDescription", "DistributionAddress", "DistributionName",
        "EthnicGroup", "FrameComments", "GantryID", "GeneratorID", "HumanPerformerName",
        "HumanPerformerOrganization", "IconImageSequence", "IdentifyingComments",
        "ImageComments", "ImagePresentationComments", "ImagingServiceRequestComments",
        "Impressions", "InstitutionAddress", "InstitutionCodeSequence", "InstitutionName",
        "InstitutionalDepartmentName", "InstitutionalDepartmentTypeCodeSequence",
        "InsurancePlanIdentification", "IntendedRecipientsOfResultsIdentificationSequence",
        "InterpretationApproverSequence", "InterpretationAuthor",
        "InterpretationDiagnosisDescription", "InterpretationIDIssuer",
        "InterpretationRecorder", "InterpretationText", "InterpretationTranscriber",
        "IssuerOfAdmissionID", "IssuerOfAdmissionIDSequence", "IssuerOfPatientID",
        "IssuerOfPatientIDQualifiersSequence", "IssuerOfServiceEpisodeID", "LastMenstrualDate",
        "MedicalAlerts", "MedicalRecordLocator", "MilitaryRank", "ModifiedAttributesSequence",
        "ModifiedImageDescription", "ModifyingDeviceID", "NameOfPhysiciansReadingStudy",
        "NamesOfIntendedRecipientsOfResults", "Occupation", "OperatorIdentificationSequence",
        "OperatorsName", "OrderCallbackPhoneNumber", "OrderEnteredBy", "OrderEntererLocation",
        "OriginalAttributesSequence", "OtherPatientIDs", "OtherPatientIDsSequence",
        "OtherPatientNames", "OverlayComments", "ParticipantSequence", "PatientAddress",
        "PatientBirthName", "PatientBirthTime", "PatientComments",
        "PatientInstitutionResidence", "PatientInsurancePlanCodeSequence",
        "PatientMotherBirthName", "PatientPrimaryLanguageCodeSequence",
        "PatientPrimaryLanguageModifierCodeSequence", "PatientReligiousPreference",
        "PatientState", "PatientTelephoneNumbers", "PatientTransportArrangements",
        "PerformedLocation", "PerformedProcedureStepDescription", "PerformedProcedureStepID",
        "PerformedStationAETitle", "PerformedStationGeographicLocationCodeSequence",
        "PerformedStationName", "PerformedStationNameCodeSequence",
        "PerformingPhysicianIdentificationSequence", "PerformingPhysicianName",
        "PersonAddress", "PersonTelephoneNumbers", "PhysicianApprovingInterpretation",
        "PhysiciansOfRecord", "PhysiciansOfRecordIdentificationSequence",
        "PhysiciansReadingStudyIdentificationSequence", "PlateID", "PreMedication",
        "PregnancyStatus", "ReasonForStudy", "ReasonForTheImagingServiceRequest",
        "ReasonForRequestedProcedureCodeSequence", "ReferencedDigitalSignatureSequence",
        "ReferencedPatientAliasSequence", "ReferencedPatientSequence",
        "ReferencedSOPInstanceMACSequence", "ReferringPhysicianAddress",
        "ReferringPhysicianIdentificationSequence", "ReferringPhysicianTelephoneNumbers",
        "RegionOfResidence", "RequestAttributesSequence", "RequestedContrastAgent",
        "RequestedProcedureComments", "RequestedProcedureID", "RequestedProcedureLocation",
        "RequestingPhysician", "RequestingPhysicianIdentificationSequence", "RequestingService",
        "ResponsibleOrganization", "ResponsiblePerson", "ResponsiblePersonRole",
        "ResultsComments", "ResultsDistributionListSequence", "ResultsIDIssuer",
        "ScheduledHumanPerformersSequence", "ScheduledPatientInstitutionResidence",
        "ScheduledPerformingPhysicianIdentificationSequence",
        "ScheduledPerformingPhysicianName", "ScheduledProcedureStepLocation",
        "ScheduledStationAETitle", "ScheduledStationGeographicLocationCodeSequence",
        "ScheduledStationName", "ScheduledStationNameCodeSequence", "ScheduledStudyLocation",
        "ScheduledStudyLocationAETitle", "ServiceEpisodeDescription", "ServiceEpisodeID",
        "SmokingStatus", "SpecialNeeds", "StationName", "StudyComments", "StudyIDIssuer",
        "TextComments", "TextString", "TimezoneOffsetFromUTC", "TopicAuthor", "TopicKeywords",
        "TopicSubject", "TopicTitle", "VerifyingOrganization", "VisitComments",
        "ReferencedPerformedProcedureStepSequence", "ReferencedStudySequence",
        "PerformedProcedureStepStartDate", "PerformedProcedureStepStartTime",
        "PerformedProcedureStepEndDate", "PerformedProcedureStepEndTime",
        "ScheduledProcedureStepStartDate", "ScheduledProcedureStepStartTime",
        "ScheduledProcedureStepEndDate", "ScheduledProcedureStepEndTime",
        "ScheduledProcedureStepDescription", "RequestedProcedureDescription",
        "InstanceCoercionDateTime", "ExpectedCompletionDateTime", "CurveDate", "CurveTime",
        "OverlayDate", "OverlayTime", "Arbitrary",
    }
)  # fmt: skip

#: Ação Z (valor vazio, atributo mantido): obrigatórios nos IODs de imagem.
EMPTY_ANYWHERE: frozenset[str] = frozenset(
    {
        "PatientName", "PatientBirthDate", "ReferringPhysicianName", "AccessionNumber",
        "StudyID", "ContentCreatorName", "FillerOrderNumberImagingServiceRequest",
        "PlacerOrderNumberImagingServiceRequest", "ReviewerName", "VerifyingObserverName",
        "PersonName", "ScheduledProcedureStepID", "ConsultingPhysicianName",
    }
)  # fmt: skip

#: Descrições: mantidas só com palavras do vocabulário fixo (limpeza por lista branca).
DESCRIPTORS: frozenset[str] = frozenset(
    {
        "StudyDescription", "SeriesDescription", "ProtocolName",
        "AcquisitionDeviceProcessingDescription", "ContrastBolusAgent",
    }
)  # fmt: skip

#: Características do paciente mantidas (opção Retain Patient Characteristics).
RETAINED_CHARACTERISTICS: frozenset[str] = frozenset(
    {"PatientSex", "PatientAge", "PatientSize", "PatientWeight", "PatientSexNeutered"}
)

_DESCRIPTOR_WORDS: frozenset[str] = frozenset(
    {
        "chest", "thorax", "torax", "tórax", "lung", "lungs", "pulmao", "pulmão", "pulmonar",
        "pecho", "head", "brain", "skull", "cranio", "crânio", "craneo", "neuro", "encefalo",
        "encéfalo", "abdomen", "abdome", "abdominal", "pelvis", "pelve", "pelvic", "liver",
        "figado", "fígado", "kidney", "kidneys", "renal", "rim", "rins", "urografia",
        "hand", "wrist", "finger", "elbow", "forearm", "humerus", "úmero", "umero", "shoulder",
        "clavicle", "clavicula", "clavícula", "knee", "ankle", "foot", "toe", "leg", "tibia",
        "tíbia", "fibula", "fíbula", "femur", "fêmur", "hip", "extremity", "mao", "mão",
        "punho", "dedo", "dedos", "cotovelo", "antebraco", "antebraço", "ombro", "joelho",
        "tornozelo", "pé", "pe", "perna", "quadril", "bacia", "breast", "mama", "mamas",
        "mamografia", "mammography", "mammo", "echo", "echocardiogram", "ecocardiograma",
        "ecocardiografia", "cardiac", "heart", "tte", "coração", "coracao", "cardiaco",
        "cardíaco", "spine", "coluna", "cervical", "lombar", "lumbar", "thoracic", "toracica",
        "torácica", "neck", "pescoço", "pescoco", "face", "sinus", "seios", "orbita", "órbita",
        "aorta", "angio", "angiotc", "mediastino", "axial", "coronal", "sagital", "sagittal",
        "frontal", "lateral", "perfil", "oblique", "obliqua", "oblíqua", "pa", "ap", "lat",
        "ll", "rl", "ct", "tc", "mr", "rm", "mri", "us", "usg", "cr", "dx", "mg", "xr", "rx",
        "cc", "mlo", "a4c", "a2c", "a3c", "plax", "psax", "apical", "parasternal",
        "subcostal", "contraste", "contrast", "sem", "com", "with", "without", "w", "wo",
        "de", "do", "da", "dos", "das", "e", "and", "of", "view", "incidencia", "incidência",
        "esquerda", "esquerdo", "direita", "direito", "left", "right", "bilateral", "l", "r",
        "scout", "topograma", "localizer", "routine", "rotina", "screening", "rastreamento",
        "diagnostic", "diagnostico", "diagnóstico", "original", "derived", "primary",
        "secondary", "other", "volume", "mpr", "mip", "bone",
        "soft", "tissue", "osso", "mediastinal", "arterial", "venosa", "venous",
        "portal", "fase", "phase", "delayed", "tardia", "helical", "helicoidal", "t1", "t2",
        "flair", "dwi", "adc", "swi", "gre", "pd", "stir", "post", "pos", "pós", "pre", "pré",
        "gd", "gado", "doppler", "cine", "raio", "x", "radiografia", "radiography",
        "tomografia", "tomography", "ressonancia", "ressonância", "ultrassom",
        "ultrassonografia", "ultrasound", "ecografia", "computed", "magnetic", "resonance",
        "digital", "computadorizada", "magnetica", "magnética", "torax/abdome",
    }
)  # fmt: skip

_STANDARD_UID_PREFIX = "1.2.840.10008."
_KEEP_UID_SUFFIXES = ("ClassUID", "ClassUIDInFile", "TransferSyntaxUID", "CodingSchemeUID")

#: Códigos DCM (CID 7050) do perfil e das opções efetivamente aplicadas.
METHOD_CODES: tuple[tuple[str, str], ...] = (
    ("113100", "Basic Application Confidentiality Profile"),
    ("113107", "Retain Longitudinal Temporal Information Modified Dates Option"),
    ("113108", "Retain Patient Characteristics Option"),
)


def clean_descriptor(text: str) -> str:
    """Mantém só palavras do vocabulário fixo (anatomia, incidência, técnica).

    Lista branca, não lista negra: nome de paciente ou médico, número de
    prontuário e datas digitados numa descrição desaparecem.
    """
    import re

    tokens = [t for t in re.split(r"[^0-9A-Za-zÀ-ÿ]+", text) if t]
    kept = [t for t in tokens if t.lower() in _DESCRIPTOR_WORDS]
    return " ".join(kept)


def _keep_uid(keyword: str, value: str) -> bool:
    return value.startswith(_STANDARD_UID_PREFIX) or keyword.endswith(_KEEP_UID_SUFFIXES)


def _clean_dataset(ds: Any, profile: DeidentificationProfile) -> None:
    """Aplica a tabela de ações e as regras por VR em ``ds`` e em todas as sequências."""
    for elem in list(ds):
        tag = elem.tag
        if tag.is_private:
            if profile.remove_private_tags:
                del ds[tag]
            continue
        keyword = elem.keyword or ""
        vr = elem.VR
        if keyword in REMOVE_ANYWHERE:
            del ds[tag]
            continue
        if keyword in EMPTY_ANYWHERE:
            if vr == "SQ":
                del ds[tag]
            else:
                elem.value = ""
            continue
        if vr == "SQ":
            for item in elem.value or []:
                _clean_dataset(item, profile)
            continue
        if keyword in RETAINED_CHARACTERISTICS:
            continue
        if keyword == "PatientID":
            # Em qualquer profundidade: pseudônimo HMAC determinístico (ação Z/U).
            if elem.value not in (None, ""):
                elem.value = profile.pseudo_id(str(elem.value))
            continue
        if keyword in DESCRIPTORS:
            values = elem.value if elem.VM > 1 else [elem.value]
            cleaned = [clean_descriptor(str(v)) for v in values if v not in (None, "")]
            kept = [c for c in cleaned if c]
            elem.value = kept if len(kept) > 1 else (kept[0] if kept else "")
            continue
        if vr == "PN":
            elem.value = ""
            continue
        if vr == "UI":
            values = elem.value if elem.VM > 1 else [elem.value]
            remapped = [
                str(v) if _keep_uid(keyword, str(v)) else profile.pseudo_uid(str(v))
                for v in values
                if v not in (None, "")
            ]
            elem.value = remapped if len(remapped) > 1 else (remapped[0] if remapped else "")
            continue
        if vr in ("DA", "DT"):
            text = str(elem.value or "")
            keep = profile.keep_year_in_dates and len(text) >= 4 and text[:4].isdigit()
            elem.value = f"{text[:4]}0101" if keep else ""
            continue
        if vr == "TM":
            elem.value = ""
            continue


def deidentify(ds: Any, profile: DeidentificationProfile) -> tuple[Any, str | None]:
    """Des-identifica uma CÓPIA de um ``pydicom.Dataset`` (o original não muda).

    Aplica o perfil básico do PS3.15 Annex E com as opções declaradas em
    ``METHOD_CODES``, em qualquer profundidade de sequência. Só depois disso
    grava ``PatientIdentityRemoved = YES`` e o método.

    Returns:
        ``(dataset_deidentificado, pseudo_patient_id)``. O pseudo-ID é ``None``
        quando o dataset não tinha ``PatientID``.
    """
    import copy

    from pydicom.dataset import Dataset

    out = copy.deepcopy(ds)
    _clean_dataset(out, profile)
    pseudo = getattr(out, "PatientID", None)
    pseudo_patient_id: str | None = str(pseudo) if pseudo not in (None, "") else None

    out.PatientIdentityRemoved = "YES"
    # LO: no máximo 64 caracteres; o detalhe das opções vai na sequência de códigos.
    out.DeidentificationMethod = "RadiologyAI PS3.15 Annex E basico recursivo; HMAC-SHA256"
    codes = []
    for value, meaning in METHOD_CODES:
        item = Dataset()
        item.CodeValue = value
        item.CodingSchemeDesignator = "DCM"
        item.CodeMeaning = meaning
        codes.append(item)
    out.DeidentificationMethodCodeSequence = codes
    out.LongitudinalTemporalInformationModified = "MODIFIED"

    return out, pseudo_patient_id
