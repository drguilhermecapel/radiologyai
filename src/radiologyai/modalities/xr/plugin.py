"""Plugin de radiografia (CR/DX).

Escopo declarado em REG-01 §2 (v1): tórax, incidência frontal (PA/AP), adultos.
Perfil, incidências especiais e população pediátrica estão FORA do escopo e são
recusados por :meth:`XRPlugin.validate`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from radiologyai.modalities.base import Dimensionality, ModalityPlugin, ValidationReport

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.io.metadata import StudyMetadata

ACCEPTED_VIEWS: frozenset[str] = frozenset({"PA", "AP"})
MIN_AGE_YEARS: float = 18.0
MIN_DIMENSION: int = 224


class XRPlugin(ModalityPlugin):
    """Radiografia computadorizada/digital."""

    code: ClassVar[str] = "XR"
    accepted_modalities: ClassVar[frozenset[str]] = frozenset({"CR", "DX"})
    dimensionality: ClassVar[Dimensionality] = "2d"
    implemented: ClassVar[bool] = True

    def validate(self, metadata: StudyMetadata) -> ValidationReport:
        """Gate de escopo — controle de risco para o perigo H-03.

        Recusa, em vez de degradar, quando a entrada está fora do uso pretendido.
        Idade ausente **não** é motivo de recusa (é comum em DICOM anonimizado);
        idade presente e abaixo de 18 anos é.
        """
        reasons: list[str] = []

        if metadata.modality.upper() not in self.accepted_modalities:
            reasons.append(f"modalidade {metadata.modality!r} não é CR nem DX")

        view = (metadata.view_position or "").upper()
        if view and view not in ACCEPTED_VIEWS:
            reasons.append(f"incidência {view!r} fora do escopo; aceitas: {sorted(ACCEPTED_VIEWS)}")

        age = metadata.patient_age_years
        if age is not None and age < MIN_AGE_YEARS:
            reasons.append(f"paciente com {age:.1f} anos; população pediátrica está fora do escopo")

        if metadata.rows is not None and metadata.rows < MIN_DIMENSION:
            reasons.append(f"altura {metadata.rows}px abaixo do mínimo {MIN_DIMENSION}px")
        if metadata.columns is not None and metadata.columns < MIN_DIMENSION:
            reasons.append(f"largura {metadata.columns}px abaixo do mínimo {MIN_DIMENSION}px")

        if metadata.is_multiframe:
            reasons.append("imagem multiframe não é esperada em radiografia")

        return ValidationReport(accepted=not reasons, modality=self.code, reasons=tuple(reasons))

    def preprocess(
        self,
        array: npt.NDArray[np.floating[Any]],
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> npt.NDArray[np.float32]:
        """Normaliza para [0, 1] em float32, preservando a faixa dinâmica.

        Não quantiza para uint8 em momento algum — esse era o defeito do legado.
        O redimensionamento para a resolução do modelo é responsabilidade do
        backend, que conhece a ``input_shape`` declarada no model card.
        """
        import numpy as np

        arr = np.asarray(array, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"radiografia deve ser 2D, recebido shape {arr.shape}")

        lo, hi = float(arr.min()), float(arr.max())
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / (hi - lo)).astype(np.float32)
