"""Janelamento (window center / window width) por modalidade.

Presets migrados de ``legacy/src/medai_dicom_processor.py`` e
``legacy/src/medai_modality_normalizer.py``, que estavam corretos. O que estava
errado no legado era o *consumidor*: ``dicom_to_array()`` quantizava em uint8
logo depois, descartando a faixa de Hounsfield que acabara de calcular
(HONEST_STATUS.md §10). Aqui o janelamento devolve float32 em [0, 1] e a
quantização, quando necessária, é decisão explícita da camada de exibição.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt


class WindowPreset(NamedTuple):
    """Janela radiológica. ``center``/``width`` em Unidades Hounsfield para TC."""

    center: float
    width: float
    description: str


WINDOW_PRESETS: dict[str, WindowPreset] = {
    # Tomografia computadorizada — em HU
    "ct_lung": WindowPreset(-600.0, 1500.0, "Parênquima pulmonar"),
    "ct_bone": WindowPreset(300.0, 1500.0, "Estruturas ósseas"),
    "ct_brain": WindowPreset(40.0, 80.0, "Tecido cerebral"),
    "ct_soft_tissue": WindowPreset(50.0, 350.0, "Partes moles"),
    "ct_mediastinum": WindowPreset(50.0, 400.0, "Mediastino"),
    "ct_liver": WindowPreset(60.0, 160.0, "Fígado"),
    "ct_angio": WindowPreset(300.0, 600.0, "Angiografia por TC"),
    # Radiografia — a janela vem do cabeçalho quando presente
    "xr_chest": WindowPreset(0.0, 0.0, "Tórax: usar VOI LUT do cabeçalho"),
}


def apply_window(
    array: npt.NDArray[np.floating[Any]],
    center: float,
    width: float,
) -> npt.NDArray[np.float32]:
    """Aplica janela e normaliza para [0, 1] em float32.

    Args:
        array: pixels já convertidos para a escala física (HU para TC).
        center: window center.
        width: window width. Deve ser > 0.

    Raises:
        ValueError: se ``width`` não for positivo.
    """
    import numpy as np

    if not width > 0:
        raise ValueError(f"window width deve ser > 0, recebido {width!r}")

    low = center - width / 2.0
    windowed = (array.astype(np.float32) - low) / np.float32(width)
    clipped: npt.NDArray[np.float32] = np.clip(windowed, 0.0, 1.0, dtype=np.float32)
    return clipped


def apply_preset(array: npt.NDArray[np.floating[Any]], preset: str) -> npt.NDArray[np.float32]:
    """Aplica um preset nomeado de :data:`WINDOW_PRESETS`."""
    if preset not in WINDOW_PRESETS:
        raise KeyError(f"preset desconhecido {preset!r}; disponíveis: {sorted(WINDOW_PRESETS)}")
    p = WINDOW_PRESETS[preset]
    return apply_window(array, p.center, p.width)
