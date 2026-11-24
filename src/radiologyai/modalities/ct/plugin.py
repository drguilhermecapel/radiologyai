"""Plugin de TC: escopo de entrada volumétrica antes de qualquer modelo (controle de H-03).

Coerente com o carregador de séries (:mod:`radiologyai.io.series`): só séries
de cortes com geometria (``ImagePositionPatient``/``ImageOrientationPatient``/
``PixelSpacing``), em HU após a Modality LUT. Recusa, em vez de degradar:

* SOP Class diferente de *CT Image Storage* (Enhanced CT multiframe, captura
  secundária, localizador salvo como outra classe);
* imagem multiframe dentro de série de cortes;
* ausência de ``PixelSpacing`` (nenhuma medida em mm seria possível);
* matriz menor que 64 x 64 (miniatura, não diagnóstico);
* paciente com idade informada < 18 anos (modelos e limiares derivados em adultos);
  idade ausente NÃO recusa aqui — os biomarcadores que exigem adulto se abstêm.

``preprocess`` não normaliza: devolve as HU em float32 ``(Z, Y, X)`` e recusa
volume não 3D, valores não finitos ou faixa incompatível com HU.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from radiologyai.errors import UnsupportedInputError
from radiologyai.modalities.base import Dimensionality, ModalityPlugin, ValidationReport

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from radiologyai.io.metadata import StudyMetadata

CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
MIN_MATRIX = 64
MIN_AGE_YEARS = 18.0
#: faixa física plausível de HU após a Modality LUT (ar ≈ -1000; metal satura ~3071+)
HU_PLAUSIBLE = (-2048.0, 32767.0)


def volumetric_scope(
    metadata: StudyMetadata,
    *,
    code: str,
    accepted: frozenset[str],
    sop_class: str,
) -> list[str]:
    """Motivos de recusa comuns a TC e RM (lista vazia = aceito)."""
    reasons: list[str] = []
    if metadata.modality.upper() not in accepted:
        reasons.append(f"modalidade {metadata.modality!r} não é {code}")
    if metadata.sop_class_uid and metadata.sop_class_uid != sop_class:
        reasons.append(
            f"SOP Class {metadata.sop_class_uid} não suportada; aceita {sop_class} "
            "(séries de cortes; Enhanced multiframe ainda não implementado)"
        )
    if metadata.is_multiframe:
        reasons.append("imagem multiframe não é suportada em série de cortes")
    if metadata.pixel_spacing is None:
        reasons.append("PixelSpacing ausente: nenhuma medida em mm seria possível")
    for name, value in (("altura", metadata.rows), ("largura", metadata.columns)):
        if value is not None and value < MIN_MATRIX:
            reasons.append(f"{name} {value}px abaixo do mínimo {MIN_MATRIX}px")
    age = metadata.patient_age_years
    if age is not None and age < MIN_AGE_YEARS:
        reasons.append(f"paciente com {age:.1f} anos; população pediátrica fora do escopo")
    return reasons


def check_volume(array: npt.NDArray[np.floating[Any]], code: str) -> npt.NDArray[np.float32]:
    """Volume 3D finito em float32 (sem cópia quando já é float32 contíguo)."""
    import numpy as np

    arr = np.ascontiguousarray(array, dtype=np.float32)
    if arr.ndim != 3:
        raise UnsupportedInputError(f"{code} exige volume 3D (Z, Y, X); recebido {arr.shape}")
    if min(arr.shape) < 1 or arr.shape[0] < 2:
        raise UnsupportedInputError(f"{code}: volume com menos de 2 cortes")
    if not np.isfinite(arr).all():
        raise UnsupportedInputError(f"{code}: volume com valores não finitos")
    return arr


class CTPlugin(ModalityPlugin):
    """Tomografia computadorizada: séries de cortes em HU."""

    code: ClassVar[str] = "CT"
    accepted_modalities: ClassVar[frozenset[str]] = frozenset({"CT"})
    dimensionality: ClassVar[Dimensionality] = "3d"
    implemented: ClassVar[bool] = True

    def validate(self, metadata: StudyMetadata) -> ValidationReport:
        """Gate de escopo de TC (controle de risco para H-03)."""
        reasons = volumetric_scope(
            metadata, code=self.code, accepted=self.accepted_modalities, sop_class=CT_IMAGE_STORAGE
        )
        return ValidationReport(accepted=not reasons, modality=self.code, reasons=tuple(reasons))

    def preprocess(
        self,
        array: npt.NDArray[np.floating[Any]],
        metadata: StudyMetadata,  # noqa: ARG002 - parte do contrato do plugin
    ) -> npt.NDArray[np.float32]:
        """HU em float32 ``(Z, Y, X)``, sem janela nem normalização (o modelo decide).

        Raises:
            UnsupportedInputError: não 3D, não finito, ou valores fora de faixa de HU
                (indício de pixels sem Modality LUT aplicada).
        """
        arr = check_volume(array, self.code)
        lo, hi = float(arr.min()), float(arr.max())
        if lo < HU_PLAUSIBLE[0] or hi > HU_PLAUSIBLE[1]:
            raise UnsupportedInputError(
                f"faixa [{lo:g}, {hi:g}] incompatível com HU; a Modality LUT foi aplicada?"
            )
        if lo > -500.0:
            raise UnsupportedInputError(
                f"mínimo {lo:g} HU: nenhum ar no volume; não parece TC em HU "
                "(pixels crus sem RescaleIntercept?)"
            )
        return arr
