"""Plugin de RM: escopo de entrada volumétrica antes de qualquer modelo (controle de H-03).

Mesmas recusas estruturais da TC (:func:`radiologyai.modalities.ct.plugin.volumetric_scope`):
SOP Class diferente de *MR Image Storage*, multiframe, ``PixelSpacing`` ausente,
matriz pequena, idade informada < 18 anos.

Intensidade de RM não tem unidade física: ``preprocess`` devolve o volume em
float32 sem normalizar (cada modelo aplica a normalização com que foi
treinado) e recusa volume não 3D, não finito ou constante.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from radiologyai.errors import UnsupportedInputError
from radiologyai.modalities.base import Dimensionality, ModalityPlugin, ValidationReport
from radiologyai.modalities.ct.plugin import check_volume, volumetric_scope

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.io.metadata import StudyMetadata

MR_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.4"


class MRPlugin(ModalityPlugin):
    """Ressonância magnética: séries de cortes (intensidade sem unidade física)."""

    code: ClassVar[str] = "MR"
    accepted_modalities: ClassVar[frozenset[str]] = frozenset({"MR"})
    dimensionality: ClassVar[Dimensionality] = "3d"
    implemented: ClassVar[bool] = True

    def validate(self, metadata: StudyMetadata) -> ValidationReport:
        """Gate de escopo de RM (controle de risco para H-03)."""
        reasons = volumetric_scope(
            metadata, code=self.code, accepted=self.accepted_modalities, sop_class=MR_IMAGE_STORAGE
        )
        return ValidationReport(accepted=not reasons, modality=self.code, reasons=tuple(reasons))

    def preprocess(
        self,
        array: npt.NDArray[np.floating[Any]],
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> npt.NDArray[np.float32]:
        """Volume float32 ``(Z, Y, X)`` sem normalização.

        Raises:
            UnsupportedInputError: não 3D, não finito ou constante.
        """
        arr = check_volume(array, self.code)
        if float(arr.max()) <= float(arr.min()):
            raise UnsupportedInputError("RM: volume constante (sem sinal)")
        return arr
