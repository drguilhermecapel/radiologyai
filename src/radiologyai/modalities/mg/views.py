"""Incidência e lateralidade de mamografia a partir do cabeçalho DICOM.

Ordem de precedência
--------------------
* Incidência: ``ViewPosition`` (0018,5101); sem ela, ``ViewCodeSequence``
  (0054,0220) com código SNOMED CT do CID 4014, ou o ``CodeMeaning`` oficial.
* Lateralidade: ``ImageLaterality`` (0020,0062); ``Laterality`` (0020,0060);
  por fim ``PatientOrientation`` (0020,0020). Na mamografia, o segundo valor
  de ``PatientOrientation`` dá a direção das colunas (de cima para baixo), que
  vai do lado LATERAL para o MEDIAL: na mama direita aponta para a esquerda do
  paciente (``L``, ``FL``), na esquerda para a direita (``R``, ``FR``)
  (PS3.3 C.8.19.6.10 / IHE Mammography Image). Valor ambíguo => ``None``.

Nada é adivinhado: quando nenhuma fonte é conclusiva, devolve ``None`` e o
portão do adaptador recusa a imagem.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

__all__ = [
    "MG_VIEW_CODES",
    "MG_VIEW_MEANINGS",
    "laterality_from_orientation",
    "normalize_view",
    "parse_laterality",
    "parse_view",
]

# CID 4014 (incidências de mamografia), SNOMED CT -> rótulo de ViewPosition.
MG_VIEW_CODES: dict[str, str] = {
    "399162004": "CC",
    "399368009": "MLO",
    "399260004": "ML",
    "399352003": "LM",
    "399099002": "XCCL",
    "399192003": "XCCM",
    "399188001": "FB",
    "399110001": "AT",
}
MG_VIEW_MEANINGS: dict[str, str] = {
    "cranio-caudal": "CC",
    "medio-lateral oblique": "MLO",
    "medio-lateral": "ML",
    "latero-medial": "LM",
    "exaggerated cranio-caudal": "XCCL",
    "cranio-caudal exaggerated laterally": "XCCL",
    "cranio-caudal exaggerated medially": "XCCM",
}

Laterality = Literal["L", "R"]


def normalize_view(value: Any) -> str | None:
    """``ViewPosition`` normalizado (maiúsculas, sem espaços); vazio => ``None``."""
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "")
    return text or None


def parse_view(view_position: Any, view_codes: Sequence[Any] | None = None) -> str | None:
    """Incidência pelo ``ViewPosition``; sem ele, pela ``ViewCodeSequence``.

    Itens da sequência podem ser ``Dataset`` do pydicom ou dicionários com
    ``CodeValue``, ``CodingSchemeDesignator`` e ``CodeMeaning``.
    """
    view = normalize_view(view_position)
    if view:
        return view
    for item in view_codes or ():
        get = item.get if isinstance(item, dict) else lambda k, it=item: getattr(it, k, None)
        scheme = str(get("CodingSchemeDesignator") or "").upper()
        code = str(get("CodeValue") or "")
        if scheme in ("SCT", "SRT") and code in MG_VIEW_CODES:
            return MG_VIEW_CODES[code]
        meaning = str(get("CodeMeaning") or "").strip().lower()
        if meaning in MG_VIEW_MEANINGS:
            return MG_VIEW_MEANINGS[meaning]
    return None


def laterality_from_orientation(orientation: Any) -> Laterality | None:
    """Lateralidade pela direção das colunas em ``PatientOrientation`` (ver módulo)."""
    if orientation is None or isinstance(orientation, (str, bytes)):
        values = str(orientation or "").replace("\\", " ").split()
    else:
        values = [str(v) for v in orientation]
    if len(values) != 2:
        return None
    column = values[1].strip().upper()
    has_left, has_right = "L" in column, "R" in column
    if has_left == has_right:
        return None
    return "R" if has_left else "L"


def parse_laterality(
    image_laterality: Any, laterality: Any = None, patient_orientation: Any = None
) -> tuple[Laterality | None, str | None]:
    """(lateralidade, tag de origem). ``B``/valores desconhecidos não servem por mama.

    REQ-194/REQ-141: ``ImageLaterality`` e ``Laterality`` que se contradizem (ex.: ``L`` e
    ``R``) não decidem o lado — devolve ``(None, "conflito")``, nunca uma tag escolhida em
    silêncio. ``PatientOrientation`` só define o lado quando nenhuma das duas tags existe; se
    existe tag e a orientação indica o lado oposto, também é conflito.
    """
    found: list[tuple[str, str]] = []
    for value, source in ((image_laterality, "ImageLaterality"), (laterality, "Laterality")):
        text = str(value or "").strip().upper()
        if text:
            found.append((text, source))
    if found:
        sides = {text for text, _ in found}
        if len(sides) > 1:
            return None, "conflito"
        text, source = found[0]
        if text in ("L", "R"):
            oriented = laterality_from_orientation(patient_orientation)
            if oriented is not None and oriented != text:
                return None, "conflito"  # tag e PatientOrientation se contradizem
            return ("L" if text == "L" else "R"), source
        return None, source  # B (ambas as mamas) ou código desconhecido: não é por mama
    lat = laterality_from_orientation(patient_orientation)
    return (lat, "PatientOrientation") if lat else (None, None)
