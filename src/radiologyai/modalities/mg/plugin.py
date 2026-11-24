"""Plugin de mamografia (MG): escopo de entrada antes de qualquer modelo (controle de H-03).

Recusa, em vez de degradar:

* modalidade diferente de MG;
* tomossíntese (SOP Class *Breast Tomosynthesis Image Storage*) e qualquer
  imagem multiframe — os modelos abertos usados aqui são de mamografia digital
  2D (FFDM) de quadro único;
* incidência informada que não seja CC ou MLO (ML, LM, XCCL, magnificação...);
  incidência ausente NÃO recusa aqui (pode vir da ``ViewCodeSequence``; o
  portão do adaptador decide);
* matriz menor que 256 px no menor lado;
* paciente com idade informada < 18 anos.

``preprocess`` não normaliza: a conversão para 8 bits depende da VOI do arquivo
e é feita pelo adaptador de cada modelo (ver :mod:`radiologyai.modalities.mg.voi`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from radiologyai.errors import UnsupportedInputError
from radiologyai.modalities.base import Dimensionality, ModalityPlugin, ValidationReport

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.io.metadata import StudyMetadata

__all__ = ["BREAST_TOMOSYNTHESIS_STORAGE", "MGPlugin"]

BREAST_TOMOSYNTHESIS_STORAGE = "1.2.840.10008.5.1.4.1.1.13.1.3"
MIN_MATRIX = 256
MIN_AGE_YEARS = 18.0
SUPPORTED_VIEWS = frozenset({"CC", "MLO"})


class MGPlugin(ModalityPlugin):
    """Mamografia digital 2D (FFDM), incidências CC e MLO."""

    code: ClassVar[str] = "MG"
    accepted_modalities: ClassVar[frozenset[str]] = frozenset({"MG"})
    dimensionality: ClassVar[Dimensionality] = "2d"
    implemented: ClassVar[bool] = True

    def validate(self, metadata: StudyMetadata) -> ValidationReport:
        """Gate de escopo de MG (controle de risco para H-03)."""
        reasons: list[str] = []
        if metadata.modality.upper() not in self.accepted_modalities:
            reasons.append(f"modalidade {metadata.modality!r} não é MG")
        if metadata.sop_class_uid == BREAST_TOMOSYNTHESIS_STORAGE:
            reasons.append("tomossíntese mamária não é suportada (só mamografia 2D)")
        if metadata.is_multiframe:
            reasons.append("imagem multiframe não é suportada em mamografia 2D")
        view = (metadata.view_position or "").strip().upper().replace(" ", "")
        if view and view not in SUPPORTED_VIEWS:
            reasons.append(f"incidência {view!r} fora do escopo (aceitas: CC, MLO)")
        for name, value in (("altura", metadata.rows), ("largura", metadata.columns)):
            if value is not None and value < MIN_MATRIX:
                reasons.append(f"{name} {value}px abaixo do mínimo {MIN_MATRIX}px")
        age = metadata.patient_age_years
        if age is not None and age < MIN_AGE_YEARS:
            reasons.append(f"paciente com {age:.1f} anos; população pediátrica fora do escopo")
        return ValidationReport(accepted=not reasons, modality=self.code, reasons=tuple(reasons))

    def preprocess(
        self,
        array: npt.NDArray[np.floating[Any]],
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> npt.NDArray[np.float32]:
        """Plano 2D finito em float32, sem janela nem normalização.

        Raises:
            UnsupportedInputError: não 2D ou com valores não finitos.
        """
        import numpy as np

        arr = np.ascontiguousarray(array, dtype=np.float32)
        if arr.ndim != 2:
            raise UnsupportedInputError(f"MG exige plano 2D (H, W); recebido {arr.shape}")
        if not np.isfinite(arr).all():
            raise UnsupportedInputError("MG: imagem com valores não finitos")
        return arr
