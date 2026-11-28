"""Verificações de qualidade/aplicabilidade de imagens de US e eco (sobre ``ImageData``).

* :func:`overlay_mask` — anotações gravadas no pixel: pixels coloridos numa
  imagem de modo B (marcas, Doppler, barras de cor) e componentes fora do setor
  de imagem (texto, réguas, logotipos). Determinístico, sem modelo.
* :func:`exam_qa` — relatório de QA de uma imagem US: modalidade, tipo
  (quadro/cine), número de quadros e tempo entre quadros (cine), espaçamento
  físico e sua origem, anotações gravadas (cabeçalho DICOM e detecção) e cor.

Limitação declarada: marcas de medida (calipers) em tons de cinza DENTRO do
setor não são distinguíveis do tecido por esta heurística; datasets públicos
como o BUSI as contêm e isso é tratado na avaliação, não escondido aqui.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import TYPE_CHECKING

from radiologyai.analysis.types import (
    AbstentionReason,
    ImageData,
    ImageKind,
    Modality,
    QACheck,
    QAReport,
)
from radiologyai.errors import BackendUnavailableError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["OverlayReport", "exam_qa", "overlay_mask"]

COLOR_DELTA = 30.0  # max(R,G,B) - min(R,G,B) acima disto = pixel colorido
MAX_OVERLAY_FRACTION = 0.25  # acima disto o quadro não é modo B utilizável


@dataclass(frozen=True)
class OverlayReport:
    """Máscara de sobreposição detectada e as frações que a compõem."""

    mask: npt.NDArray[np.bool_]
    color_fraction: float
    outside_sector_fraction: float
    sector_bbox: tuple[int, int, int, int] | None  # x0, y0, x1, y1

    @property
    def fraction(self) -> float:
        return float(self.mask.mean())


def _require_scipy() -> None:
    if importlib.util.find_spec("scipy") is None:
        raise BackendUnavailableError(
            "scipy não instalado (componentes conexos); instale o extra: "
            "pip install 'radiologyai[eval]'"
        )


def overlay_mask(image: ImageData, frame: int = 0) -> OverlayReport:
    """Detecta anotações gravadas num quadro de US/eco.

    Raises:
        BackendUnavailableError: scipy ausente.
    """
    _require_scipy()
    import numpy as np
    from scipy import ndimage

    px = image.pixels[frame] if image.kind is ImageKind.CINE else image.pixels
    arr = np.asarray(px, dtype=np.float32)
    lo, hi = image.value_range
    norm = (arr - np.float32(lo)) * np.float32(255.0 / (hi - lo))
    if image.color:
        color = (norm.max(axis=-1) - norm.min(axis=-1)) > COLOR_DELTA
        gray = norm.mean(axis=-1)
    else:
        color = np.zeros(norm.shape, dtype=bool)
        gray = norm
    nonzero = gray > 5.0
    opened = ndimage.binary_opening(nonzero & ~color, structure=np.ones((5, 5), dtype=bool))
    labels, n = ndimage.label(opened)
    sector_bbox: tuple[int, int, int, int] | None = None
    if n == 0:
        outside = nonzero
    else:
        sizes = np.bincount(labels.ravel())[1:]
        sector = labels == (1 + int(np.argmax(sizes)))
        sector = ndimage.binary_fill_holes(ndimage.binary_dilation(sector, iterations=3))
        ys, xs = np.nonzero(sector)
        sector_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        outside = nonzero & ~sector
    mask = color | outside
    return OverlayReport(
        mask=mask,
        color_fraction=float(color.mean()),
        outside_sector_fraction=float(outside.mean()),
        sector_bbox=sector_bbox,
    )


def exam_qa(image: ImageData, *, require_spacing: bool = False) -> QAReport:
    """Relatório de QA de uma imagem de US/eco (não decide sobre modelos específicos).

    Com ``require_spacing``, falta de espaçamento físico vira abstenção por
    geometria (medidas em mm seriam inventadas).
    """
    checks: list[QACheck] = []
    reasons: list[AbstentionReason] = []
    ok_mod = image.modality is Modality.US
    checks.append(QACheck(name="modality", passed=ok_mod, value=image.modality.value))
    if not ok_mod:
        reasons.append(AbstentionReason.MODALITY)
    ok_kind = image.kind in (ImageKind.PROJECTION, ImageKind.CINE)
    checks.append(QACheck(name="image_kind", passed=ok_kind, value=image.kind.value))
    if not ok_kind:
        reasons.append(AbstentionReason.INPUT)
    if image.kind is ImageKind.CINE:
        n = int(image.pixels.shape[0])
        checks.append(QACheck(name="n_frames", passed=n >= 2, value=float(n), threshold=2.0))
        if n < 2:
            reasons.append(AbstentionReason.INPUT)
        ft = image.meta.frame_time_ms
        checks.append(
            QACheck(
                name="frame_time_ms",
                passed=True,
                value=ft,
                detail=None if ft is not None else "tempo entre quadros ausente",
            )
        )
    has_spacing = image.spacing_mm is not None
    checks.append(
        QACheck(
            name="pixel_spacing",
            passed=has_spacing or not require_spacing,
            value=image.meta.pixel_spacing_source,
            detail=None if has_spacing else "sem espaçamento físico: medidas só em pixels",
        )
    )
    if require_spacing and not has_spacing:
        reasons.append(AbstentionReason.GEOMETRY)
    burned = image.meta.burned_in_annotation
    checks.append(
        QACheck(
            name="burned_in_annotation_header",
            passed=True,
            value=None if burned is None else ("YES" if burned else "NO"),
        )
    )
    if importlib.util.find_spec("scipy") is not None:
        report = overlay_mask(image, 0)
        ok_overlay = report.fraction <= MAX_OVERLAY_FRACTION
        checks.append(
            QACheck(
                name="overlay_fraction",
                passed=ok_overlay,
                value=round(report.fraction, 6),
                threshold=MAX_OVERLAY_FRACTION,
                detail=(
                    f"colorido {report.color_fraction:.4f}; "
                    f"fora do setor {report.outside_sector_fraction:.4f}"
                ),
            )
        )
        if not ok_overlay:
            reasons.append(AbstentionReason.QUALITY)
    if reasons:
        return QAReport.reject(*dict.fromkeys(reasons), checks=checks)
    return QAReport.ok(checks)
