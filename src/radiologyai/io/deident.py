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

Escopo declarado: implementa um subconjunto do perfil *Basic Application Level
Confidentiality* (DICOM PS3.15 Annex E). **Não cobre anotação gravada no pixel
(burned-in annotation)** — ver :func:`has_burned_in_annotation`.
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


def deidentify(ds: Any, profile: DeidentificationProfile) -> tuple[Any, str | None]:
    """Des-identifica um ``pydicom.Dataset`` in place numa cópia.

    Returns:
        ``(dataset_deidentificado, pseudo_patient_id)``. O pseudo-ID é ``None``
        quando o dataset não tinha ``PatientID``.
    """
    import copy

    out = copy.deepcopy(ds)
    pseudo_patient_id: str | None = None

    for tag in TAGS_TO_REMOVE:
        if tag in out:
            setattr(out, tag, "")

    for tag in TAGS_TO_PSEUDONYMIZE:
        original = getattr(out, tag, None)
        if original in (None, ""):
            continue
        pseudo = profile.pseudo_id(str(original))
        setattr(out, tag, pseudo)
        if tag == "PatientID":
            pseudo_patient_id = pseudo

    for tag in TAGS_TO_PSEUDONYMIZE_UID:
        original = getattr(out, tag, None)
        if original in (None, ""):
            continue
        setattr(out, tag, profile.pseudo_uid(str(original)))

    for tag in TAGS_TO_TRUNCATE_DATE:
        original = getattr(out, tag, None)
        if original in (None, ""):
            continue
        text = str(original)
        setattr(
            out,
            tag,
            f"{text[:4]}0101" if profile.keep_year_in_dates and len(text) >= 4 else "",
        )

    for tag in TAGS_TO_CLEAR_TIME:
        if tag in out:
            setattr(out, tag, "")

    if profile.remove_private_tags:
        out.remove_private_tags()

    out.PatientIdentityRemoved = "YES"
    out.DeidentificationMethod = "RadiologyAI PS3.15 Annex E (subconjunto); HMAC-SHA256"

    return out, pseudo_patient_id
