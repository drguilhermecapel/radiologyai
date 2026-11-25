"""Plugin de ultrassom (US) e ecocardiografia: imagens estáticas e clipes (cine).

Escopo de entrada antes de qualquer modelo (controle de H-03). Recusa, em vez
de degradar:

* modalidade diferente de US;
* matriz menor que :data:`MIN_MATRIX` px no menor lado;
* paciente com idade informada < 18 anos (os modelos abertos usados aqui foram
  desenvolvidos em adultos);
* clipe com mais de :data:`MAX_FRAMES` quadros (laço longo / volume 4D exportado).

Multiframe é aceito: é o cine de US/eco (``ImageKind.CINE``, ``(T, H, W)``).
Anotações gravadas na imagem (texto, réguas, marcas coloridas) não recusam o
exame aqui — são detectadas e relatadas por :mod:`radiologyai.modalities.us.qa`,
e cada adaptador decide (ver ``exam_qa``).

``preprocess`` não normaliza: só garante float32 finito com 2 (quadro) ou 3
(cine) dimensões espaciais/temporais; janela e redimensionamento são de cada
modelo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from radiologyai.errors import UnsupportedInputError
from radiologyai.modalities.base import Dimensionality, ModalityPlugin, ValidationReport

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.io.metadata import StudyMetadata

__all__ = ["MAX_FRAMES", "MIN_AGE_YEARS", "MIN_MATRIX", "USPlugin"]

MIN_MATRIX = 128
MIN_AGE_YEARS = 18.0
MAX_FRAMES = 2000


class USPlugin(ModalityPlugin):
    """Ultrassom 2D (modo B) e ecocardiografia transtorácica 2D: quadros e cine."""

    code: ClassVar[str] = "US"
    accepted_modalities: ClassVar[frozenset[str]] = frozenset({"US"})
    dimensionality: ClassVar[Dimensionality] = "cine"
    implemented: ClassVar[bool] = True

    def validate(self, metadata: StudyMetadata) -> ValidationReport:
        """Gate de escopo de US (controle de risco para H-03)."""
        reasons: list[str] = []
        if metadata.modality.upper() not in self.accepted_modalities:
            reasons.append(f"modalidade {metadata.modality!r} não é US")
        for name, value in (("altura", metadata.rows), ("largura", metadata.columns)):
            if value is not None and value < MIN_MATRIX:
                reasons.append(f"{name} {value}px abaixo do mínimo {MIN_MATRIX}px")
        if metadata.number_of_frames > MAX_FRAMES:
            reasons.append(
                f"clipe com {metadata.number_of_frames} quadros acima do máximo {MAX_FRAMES}"
            )
        age = metadata.patient_age_years
        if age is not None and age < MIN_AGE_YEARS:
            reasons.append(f"paciente com {age:.1f} anos; população pediátrica fora do escopo")
        return ValidationReport(accepted=not reasons, modality=self.code, reasons=tuple(reasons))

    def preprocess(
        self,
        array: npt.NDArray[np.floating[Any]],
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> npt.NDArray[np.float32]:
        """Quadro ``(H, W)`` ou cine ``(T, H, W)`` (opcionalmente com cor no fim), float32 finito.

        Raises:
            UnsupportedInputError: dimensões incompatíveis ou valores não finitos.
        """
        import numpy as np

        arr = np.ascontiguousarray(array, dtype=np.float32)
        spatial = arr.ndim - (1 if arr.ndim >= 3 and arr.shape[-1] == 3 else 0)
        if spatial not in (2, 3):
            raise UnsupportedInputError(
                f"US exige quadro (H, W) ou cine (T, H, W); recebido {arr.shape}"
            )
        if not np.isfinite(arr).all():
            raise UnsupportedInputError("US: imagem com valores não finitos")
        return arr
