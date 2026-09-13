"""Leitura de DICOM para array de pixels na escala física correta.

Corrige três defeitos do legado (HONEST_STATUS.md §10, ROADMAP §1.10):

1. ``RescaleSlope``/``RescaleIntercept`` eram aplicados, mas o resultado era
   imediatamente quantizado com ``(arr * 255).astype(np.uint8)``, descartando a
   faixa de Hounsfield. Aqui nada é quantizado: devolve-se float32 em HU.
2. Não havia ``apply_voi_lut`` no caminho principal, então a janela gravada no
   cabeçalho pelo equipamento era ignorada.
3. ``MONOCHROME1`` (branco = valor baixo) nunca era invertido, o que produz
   imagem em negativo para uma fração real dos equipamentos.

Também remove o cache do legado, que fazia MD5 do arquivo inteiro em memória a
cada leitura e retinha todos os datasets já lidos, indefinidamente.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from radiologyai.errors import InvalidDICOMError, MissingTagError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

PixelScale = Literal["stored", "modality", "voi"]


def read_dicom(path: str | Path, *, stop_before_pixels: bool = False) -> Any:
    """Lê um arquivo DICOM e devolve o ``pydicom.Dataset``.

    Args:
        path: caminho do arquivo.
        stop_before_pixels: lê apenas o cabeçalho. Use para triagem rápida de
            metadados sem pagar o custo de decodificar os pixels.

    Raises:
        InvalidDICOMError: arquivo inexistente, ilegível ou não-DICOM.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    p = Path(path)
    if not p.is_file():
        raise InvalidDICOMError(f"arquivo não encontrado: {p}")

    try:
        return pydicom.dcmread(str(p), stop_before_pixels=stop_before_pixels)
    except InvalidDicomError as exc:
        raise InvalidDICOMError(f"não é um DICOM válido: {p}") from exc
    except Exception as exc:  # noqa: BLE001 - reembalado com contexto
        raise InvalidDICOMError(f"falha ao ler {p}: {exc}") from exc


def _apply_modality_lut(ds: Any, raw: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    """Aplica RescaleSlope/RescaleIntercept — converte para a escala física (HU em TC)."""
    import numpy as np

    slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
    return (raw.astype(np.float32) * np.float32(slope)) + np.float32(intercept)


def _invert_if_monochrome1(ds: Any, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Inverte a imagem quando PhotometricInterpretation é MONOCHROME1.

    Em MONOCHROME1 o valor mínimo é branco. Sem esta inversão a imagem é exibida
    e processada em negativo — e um modelo treinado em MONOCHROME2 falha de modo
    silencioso e catastrófico (perigo H-05).
    """
    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
    if photometric != "MONOCHROME1":
        return arr
    inverted: npt.NDArray[np.float32] = arr.max() - arr + arr.min()
    return inverted


def to_pixel_array(
    ds: Any,
    *,
    scale: PixelScale = "modality",
    frame: int | None = None,
) -> npt.NDArray[np.float32]:
    """Converte um dataset em array float32, na escala pedida.

    Args:
        ds: ``pydicom.Dataset`` com pixels.
        scale: ``"stored"`` devolve os valores brutos; ``"modality"`` aplica
            RescaleSlope/Intercept (HU para TC); ``"voi"`` aplica também a VOI
            LUT / janela gravada no cabeçalho e normaliza para [0, 1].
        frame: índice do frame em séries multiframe/cine. ``None`` devolve todos.

    Nunca quantiza para uint8. A saída é float32 e preserva a faixa dinâmica.

    Raises:
        MissingTagError: o dataset não contém dados de pixel.
        ValueError: ``frame`` fora do intervalo.
    """
    import numpy as np

    if not hasattr(ds, "pixel_array"):
        raise MissingTagError("dataset não contém PixelData")

    try:
        raw = ds.pixel_array
    except Exception as exc:  # noqa: BLE001
        raise InvalidDICOMError(f"falha ao decodificar PixelData: {exc}") from exc

    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    if frame is not None:
        if n_frames <= 1:
            if frame != 0:
                raise ValueError(f"frame {frame} pedido, mas a imagem é single-frame")
        else:
            if not 0 <= frame < n_frames:
                raise ValueError(f"frame {frame} fora do intervalo [0, {n_frames})")
            raw = raw[frame]

    if scale == "stored":
        return np.asarray(raw, dtype=np.float32)

    arr = _apply_modality_lut(ds, np.asarray(raw))
    arr = _invert_if_monochrome1(ds, arr)

    if scale == "modality":
        return arr

    # scale == "voi"
    return _apply_voi(ds, arr)


def _apply_voi(ds: Any, arr: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Aplica a janela do cabeçalho e normaliza para [0, 1].

    Usa ``WindowCenter``/``WindowWidth`` quando presentes — inclusive quando são
    multivalorados, caso em que o primeiro par é usado. Sem janela no cabeçalho,
    faz normalização min-max, que é o comportamento honesto: não inventamos uma
    janela clínica que o equipamento não declarou.
    """
    import numpy as np

    from radiologyai.io.windowing import apply_window

    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)

    if center is not None and width is not None:
        # MultiValue quando o equipamento grava mais de uma janela sugerida.
        if isinstance(center, (list, tuple)) or type(center).__name__ == "MultiValue":
            center = center[0]
        if isinstance(width, (list, tuple)) or type(width).__name__ == "MultiValue":
            width = width[0]
        try:
            c, w = float(center), float(width)
        except (TypeError, ValueError):
            c = w = 0.0
        if w > 0:
            return apply_window(arr, c, w)

    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)
