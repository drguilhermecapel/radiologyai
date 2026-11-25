"""DICOM SR (TID 1500, Comprehensive 3D SR) e DICOM SEG via highdicom.

* Códigos: somente ``pydicom.sr.codedict`` ou o esquema privado ``99MEDAI``;
  a única exceção é o idioma do conteúdo (``pt-BR``), que o TID 1500 exige no
  esquema RFC5646 (CID 5000)
  (os códigos dos achados já foram validados no manifesto por
  :func:`radiologyai.analysis.manifest.validate_code`; aqui são reconferidos).
* Decisões são codificadas em ``99MEDAI`` como *sugestão* — nunca com os
  códigos SNOMED "Present"/"Absent", que afirmariam a condição.
* Os UIDs do SR/SEG derivam deterministicamente do id do laudo.
* O SR só existe para estudo DICOM: as imagens de origem são a evidência.
"""

from __future__ import annotations

import importlib.util
import io
from typing import TYPE_CHECKING, Any

from radiologyai import __version__
from radiologyai.analysis.types import PRIVATE_SCHEME
from radiologyai.errors import BackendUnavailableError, RadiologyAIError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.reporting.content import ReportContent
    from radiologyai.reporting.pdf import SignOff

__all__ = ["ReportFormatError", "build_seg", "build_sr", "private"]

MANUFACTURER = "RadiologyAI (software de pesquisa)"


class ReportFormatError(RadiologyAIError):
    """O formato pedido não pode ser gerado para este estudo (ex.: SR sem DICOM de origem)."""


def _require() -> None:
    if importlib.util.find_spec("highdicom") is None:
        raise BackendUnavailableError(
            "highdicom não instalado; instale o extra: pip install 'radiologyai[platform]'"
        )


def private(value: str, meaning: str) -> Any:
    """Código do esquema privado 99MEDAI."""
    from highdicom.sr import CodedConcept

    return CodedConcept(value=value[:64], scheme_designator=PRIVATE_SCHEME, meaning=meaning[:64])


def _concept(scheme: str, value: str, meaning: str) -> Any:
    from highdicom.sr import CodedConcept

    from radiologyai.analysis.manifest import validate_code
    from radiologyai.catalog.schema import CodedConcept as Concept

    validate_code(Concept(scheme=scheme, value=value, meaning=meaning))
    return CodedConcept(value=value, scheme_designator=scheme, meaning=meaning[:64])


def _uid(*parts: str) -> str:
    from pydicom.uid import generate_uid

    return str(generate_uid(entropy_srcs=["radiologyai", *parts]))


def build_sr(
    content: ReportContent,
    sign: SignOff,
    evidence: list[Any],
    image_sop: dict[int, tuple[str, str]],
) -> bytes:
    """SR TID 1500 com um grupo de medidas/avaliações por achado do laudo (níveis A-C).

    Args:
        evidence: cabeçalhos DICOM (des-identificados) das imagens de origem.
        image_sop: índice da imagem no exame -> (SOPClassUID, SOPInstanceUID).

    Raises:
        ReportFormatError: estudo sem DICOM de origem.
    """
    _require()
    import highdicom as hd
    from highdicom.sr import (
        AlgorithmIdentification,
        Comprehensive3DSR,
        DeviceObserverIdentifyingAttributes,
        Measurement,
        MeasurementReport,
        MeasurementsAndQualitativeEvaluations,
        ObservationContext,
        ObserverContext,
        PersonObserverIdentifyingAttributes,
        QualitativeEvaluation,
        SourceImageForMeasurementGroup,
        TextEvaluation,
        TrackingIdentifier,
    )
    from pydicom.sr.codedict import codes

    if not evidence:
        raise ReportFormatError("DICOM SR exige estudo DICOM (imagens de origem como evidência)")

    device = ObserverContext(
        observer_type=codes.DCM.Device,
        observer_identifying_attributes=DeviceObserverIdentifyingAttributes(
            uid=_uid(content.analysis_id, "device"),
            name="RadiologyAI",
            manufacturer_name=MANUFACTURER,
            model_name=f"radiologyai {__version__}",
        ),
    )
    person = None
    if sign.signed_at is not None and sign.signer_name:
        person = ObserverContext(
            observer_type=codes.DCM.Person,
            observer_identifying_attributes=PersonObserverIdentifyingAttributes(
                name=sign.signer_name.replace(" ", "^", 1), login_name=sign.signer_username
            ),
        )
    context = ObservationContext(observer_person_context=person, observer_device_context=device)

    groups = []
    for f in content.findings:
        decision = private(f"decision.{f.decision}", f"suggestion: {f.decision}")
        evaluations = [
            QualitativeEvaluation(
                name=private("suggested-status", "Suggested finding status"), value=decision
            ),
            QualitativeEvaluation(
                name=private("evidence-level", "Evidence level"),
                value=private(f"level.{f.evidence_level}", f"Evidence level {f.evidence_level}"),
            ),
        ]
        if f.review is not None:
            evaluations.append(
                QualitativeEvaluation(
                    name=private("review", "Physician review"),
                    value=private(f"review.{f.review.action}", f"Physician {f.review.action}"),
                )
            )
        measurements = []
        if f.raw_score is not None:
            measurements.append(
                Measurement(
                    name=private("model-score", f"Model score ({f.score_kind})"),
                    value=float(f.raw_score),
                    unit=codes.UCUM.NoUnits,
                )
            )
        texts = []
        if f.operating_point:
            texts.append(
                TextEvaluation(
                    name=private("operating-point", "Operating point and source"),
                    value=f.operating_point[:1024],
                )
            )
        sources = []
        if f.image_idx in image_sop:
            sop_class, sop_uid = image_sop[f.image_idx]
            sources.append(SourceImageForMeasurementGroup(sop_class, sop_uid))
        groups.append(
            MeasurementsAndQualitativeEvaluations(
                tracking_identifier=TrackingIdentifier(
                    uid=_uid(content.analysis_id, "finding", str(f.idx)),
                    identifier=f"finding-{f.idx}",
                ),
                finding_type=_concept(f.code_scheme, f.code_value, f.code_meaning),
                algorithm_id=AlgorithmIdentification(name=f.model_id, version=f.model_version),
                measurements=measurements or None,
                qualitative_evaluations=evaluations,
                text_evaluations=texts or None,
                source_images=sources or None,
            )
        )
    from highdicom.sr import CodedConcept, LanguageOfContentItemAndDescendants

    # TID 1500 exige o idioma (TID 1204); o esquema obrigatório é RFC5646 (CID 5000).
    language = LanguageOfContentItemAndDescendants(
        language=CodedConcept(
            value="pt-BR", scheme_designator="RFC5646", meaning="Portuguese (Brazil)"
        )
    )
    if not groups:
        # TID 1500 exige ao menos um grupo; sem achado A-C, o grupo só carrega a
        # declaração textual (nunca uma afirmação de normalidade).
        statement = content.no_findings_statement or "Nenhum achado A-C."
        groups.append(
            MeasurementsAndQualitativeEvaluations(
                tracking_identifier=TrackingIdentifier(
                    uid=_uid(content.analysis_id, "statement"), identifier="sem-achados-A-C"
                ),
                text_evaluations=[
                    TextEvaluation(
                        name=private("report-statement", "Report statement"),
                        value=statement[:1024],
                    )
                ],
            )
        )
    report = MeasurementReport(
        observation_context=context,
        procedure_reported=codes.SCT.DiagnosticImaging,
        imaging_measurements=groups,
        language_of_content_item_and_descendants=language,
    )
    signed = sign.signed_at is not None and bool(sign.signer_name)
    sr = Comprehensive3DSR(
        evidence=evidence,
        content=report[0],
        series_instance_uid=_uid(sign.report_id, "sr-series"),
        series_number=9001,
        sop_instance_uid=_uid(sign.report_id, "sr", sign.signature_sha256 or "unsigned"),
        instance_number=1,
        manufacturer=MANUFACTURER,
        is_complete=False,
        is_final=False,
        is_verified=signed,
        verifying_observer_name=(sign.signer_name or "").replace(" ", "^", 1) if signed else None,
        verifying_organization="RadiologyAI - uso exclusivo em pesquisa" if signed else None,
    )
    sr.SeriesDescription = "RadiologyAI - relatorio preliminar de pesquisa (nao diagnostico)"
    del hd
    buf = io.BytesIO()
    sr.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def build_seg(
    *,
    report_id: str,
    seg_name: str,
    model_id: str,
    model_version: str,
    labelmap: npt.NDArray[np.integer[Any]],
    labels: dict[int, tuple[str, str]],
    source_images: list[Any],
) -> bytes:
    """DICOM SEG (LABELMAP) de um mapa de rótulos sobre as imagens de origem.

    Args:
        labels: índice -> (chave do rótulo, significado legível).
        source_images: datasets de origem na ordem do volume (ou a imagem 2D).

    Raises:
        ReportFormatError: geometria incompatível com as imagens de origem.
    """
    _require()
    import highdicom as hd
    import numpy as np
    from highdicom.seg import Segmentation, SegmentDescription

    if not source_images:
        raise ReportFormatError("DICOM SEG exige imagens DICOM de origem")
    arr = np.asarray(labelmap)
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    if arr.shape[0] != len(source_images):
        raise ReportFormatError(
            f"mapa com {arr.shape[0]} quadro(s) não corresponde a {len(source_images)} imagem(ns)"
        )
    present = sorted(int(v) for v in np.unique(arr) if v > 0 and int(v) in labels)
    if not present:
        raise ReportFormatError("mapa de rótulos vazio: nada a exportar em SEG")
    remap = np.zeros(int(arr.max()) + 1, dtype=np.uint16)
    for new, old in enumerate(present, start=1):
        remap[old] = new
    arr = remap[arr]
    from pydicom.sr.codedict import codes

    family = hd.AlgorithmIdentificationSequence(
        name=model_id[:64], family=codes.DCM.ArtificialIntelligence, version=model_version[:64]
    )
    descriptions = [
        SegmentDescription(
            segment_number=new,
            segment_label=labels[old][0][:64],
            segmented_property_category=private("model-segmentation", "Model segmentation"),
            segmented_property_type=private(labels[old][0], labels[old][1]),
            algorithm_type="AUTOMATIC",
            algorithm_identification=family,
        )
        for new, old in enumerate(present, start=1)
    ]
    dtype = np.uint8 if len(present) < 256 else np.uint16
    try:
        seg = Segmentation(
            source_images=source_images,
            pixel_array=arr.astype(dtype) if len(source_images) > 1 else arr[0].astype(dtype),
            segmentation_type="LABELMAP",
            segment_descriptions=descriptions,
            series_instance_uid=_uid(report_id, "seg-series", seg_name),
            series_number=9002,
            sop_instance_uid=_uid(report_id, "seg", seg_name),
            instance_number=1,
            manufacturer=MANUFACTURER,
            manufacturer_model_name=f"radiologyai {__version__}",
            software_versions=__version__,
            device_serial_number="pesquisa",
            content_description=f"{seg_name} - pesquisa, nao diagnostico"[:64],
        )
    except (ValueError, TypeError, AttributeError) as exc:
        raise ReportFormatError(f"SEG não pôde ser gerado: {exc}") from exc
    buf = io.BytesIO()
    seg.save_as(buf, enforce_file_format=True)
    return buf.getvalue()
