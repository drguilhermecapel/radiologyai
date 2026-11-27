"""FHIR R4 ``DiagnosticReport`` (JSON) do laudo preliminar de pesquisa.

* ``status`` é SEMPRE ``preliminary`` — software de pesquisa não emite laudo final.
* Achados vão como ``Observation`` contidas, também ``preliminary``, com o
  código do manifesto, o nível de evidência e a revisão do médico. O valor
  primário (``valueCodeableConcept``) é a decisão FINAL do médico (aceite ou
  emenda); a sugestão do modelo vai num componente separado. Achado rejeitado
  não tem valor (``dataAbsentReason``) e não conta na conclusão.
* Escore abaixo do limiar é ``decision.below_threshold`` ("não exclui a
  condição"), nunca ``negative``: o produto não exclui doença.
* Paciente só pelo pseudônimo; nenhum nome, data de nascimento ou documento.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from radiologyai.reporting.content import HEADER_PT, export_codes

if TYPE_CHECKING:
    from radiologyai.reporting.content import ReportContent
    from radiologyai.reporting.pdf import SignOff

__all__ = ["SYSTEMS", "build_diagnostic_report"]

SYSTEMS = {
    "SCT": "http://snomed.info/sct",
    "LN": "http://loinc.org",
    "DCM": "http://dicom.nema.org/resources/ontology/DCM",
    "UCUM": "http://unitsofmeasure.org",
    "99MEDAI": "urn:radiologyai:codes:99MEDAI",
}


def _coding(scheme: str, value: str, meaning: str) -> dict[str, str]:
    return {
        "system": SYSTEMS.get(scheme, f"urn:dicom:scheme:{scheme}"),
        "code": value,
        "display": meaning,
    }


def build_diagnostic_report(content: ReportContent, sign: SignOff, *, issued: str) -> bytes:
    """``DiagnosticReport`` FHIR R4 serializado (UTF-8, chaves ordenadas)."""
    from pydicom.sr.codedict import codes

    report_code = codes.LN.DiagnosticImagingReport
    observations: list[dict[str, Any]] = []
    for f in content.findings:
        components: list[dict[str, Any]] = [
            {
                "code": {"coding": [_coding("99MEDAI", "evidence-level", "Evidence level")]},
                "valueString": f.evidence_level,
            },
        ]
        if f.raw_score is not None:
            components.append(
                {
                    "code": {
                        "coding": [
                            _coding("99MEDAI", "model-score", f"Model score ({f.score_kind})")
                        ]
                    },
                    "valueQuantity": {"value": f.raw_score, "system": SYSTEMS["UCUM"], "code": "1"},
                }
            )
        if f.operating_point:
            components.append(
                {
                    "code": {"coding": [_coding("99MEDAI", "operating-point", "Operating point")]},
                    "valueString": f.operating_point,
                }
            )
        model_code, final_code = export_codes(f)
        components.append(
            {
                "code": {"coding": [_coding("99MEDAI", "model-suggestion", "Model suggestion")]},
                "valueCodeableConcept": {
                    "coding": [_coding("99MEDAI", *model_code)],
                    "text": f"sugestão do modelo: {f.decision_pt}",
                },
            }
        )
        if f.level_at_analysis is not None:
            components.append(
                {
                    "code": {
                        "coding": [
                            _coding("99MEDAI", "level-at-analysis", "Evidence level at analysis")
                        ]
                    },
                    "valueString": f.level_at_analysis,
                }
            )
        notes = [
            {
                "text": f"Modelo {f.model_id} {f.model_version}. Sugestão do modelo: "
                f"{f.decision_pt}. Decisão do médico: {f.physician_decision_pt}."
            }
        ]
        if f.review is not None:
            notes.append(
                {
                    "text": f"Revisão: {f.review.action} por {f.review.reviewer_name} em "
                    f"{f.review.reviewed_at}"
                    + (
                        f". Texto emendado: {f.review.amended_text}"
                        if f.review.amended_text
                        else ""
                    )
                }
            )
        observation: dict[str, Any] = {
            "resourceType": "Observation",
            "id": f"finding-{f.idx}",
            "status": "preliminary",
            "code": {
                "coding": [_coding(f.code_scheme, f.code_value, f.code_meaning)],
                "text": f.condition_name,
            },
            "subject": {"reference": "#patient"},
            "component": components,
            "note": notes,
            "method": {"text": "achado sugerido para confirmação médica (software de pesquisa)"},
        }
        if f.physician_decision == "rejected":
            assert final_code is not None
            observation["dataAbsentReason"] = {
                "coding": [_coding("99MEDAI", *final_code)],
                "text": f.physician_decision_pt,
            }
        elif final_code is not None:
            observation["valueCodeableConcept"] = {
                "coding": [_coding("99MEDAI", *final_code)],
                "text": f.physician_decision_pt,
            }
        else:
            observation["valueCodeableConcept"] = {
                "coding": [_coding("99MEDAI", *model_code)],
                "text": f"sugestão do modelo (não revisada): {f.decision_pt}",
            }
        observations.append(observation)
    patient: dict[str, Any] = {"resourceType": "Patient", "id": "patient"}
    if content.patient_pseudo_id:
        patient["identifier"] = [
            {"system": "urn:radiologyai:pseudonym", "value": content.patient_pseudo_id}
        ]
    rejected = sum(f.physician_decision == "rejected" for f in content.findings)
    kept = len(content.findings) - rejected
    conclusion = content.no_findings_statement or (
        f"{kept} achado(s) sugerido(s) com nível de evidência A-C; requerem confirmação médica."
        + (f" {rejected} sugestão(ões) rejeitada(s) pelo médico." if rejected else "")
    )
    if content.runs_statement:
        conclusion = f"{conclusion} {content.runs_statement}"
    resource: dict[str, Any] = {
        "resourceType": "DiagnosticReport",
        "id": sign.report_id,
        "meta": {"tag": [{"system": "urn:radiologyai:use", "code": "research-only"}]},
        "contained": [patient, *observations],
        "status": "preliminary",
        "category": [
            {
                "coding": [
                    {
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "RAD",
                        "display": "Radiology",
                    }
                ]
            }
        ],
        "code": {
            "coding": [_coding("LN", report_code.value, report_code.meaning)],
            "text": "Relatório preliminar de pesquisa",
        },
        "subject": {"reference": "#patient"},
        "issued": issued,
        "identifier": [
            {"system": "urn:radiologyai:report-sha256", "value": sign.report_sha256},
            {"system": "urn:radiologyai:result-sha256", "value": content.result_sha256},
        ],
        "result": [{"reference": f"#{o['id']}"} for o in observations],
        "conclusion": f"{HEADER_PT}. {conclusion}",
    }
    if sign.signed_at is not None and sign.signer_name:
        resource["resultsInterpreter"] = [{"display": sign.signer_name}]
        resource["extension"] = [
            {
                "url": "urn:radiologyai:signature-sha256",
                "valueString": sign.signature_sha256 or "",
            }
        ]
    return json.dumps(resource, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8")
