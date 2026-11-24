"""Transformação VOI (janela / VOI LUT) de mamografia, fiel ao ``pydicom``.

Por que existe
--------------
Os modelos ``ianpan/mammoscreen`` e ``ianpan/mammo-crop`` foram treinados com
imagens 8 bits obtidas por ``pydicom.pixels.apply_voi_lut`` sobre os pixels do
arquivo, seguida de inversão MONOCHROME1 e min-max (cartão do modelo: "it is
important to apply the lookup table to the pixel values"). O carregador comum
(:func:`radiologyai.analysis.imaging.load_dicom`) guarda só ``(centro,
largura)`` em ``ImageData.window`` e NÃO a função VOI (``LINEAR``,
``LINEAR_EXACT`` ou ``SIGMOID``) nem a ``VOILUTSequence`` — e a CMMD, por
exemplo, declara ``VOILUTFunction = SIGMOID``. Aplicar uma janela linear no
lugar de uma sigmoide mudaria a entrada do modelo em silêncio.

Este módulo extrai do cabeçalho tudo o que ``apply_voi_lut`` usa
(:func:`voi_from_dataset`, serializável em JSON, guardado em
``ImageData.extra[VOI_EXTRA_KEY]``) e reaplica a transformação com o próprio
``pydicom`` (:func:`apply_voi`) — nada é reimplementado à mão.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from radiologyai.errors import UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["VOI_EXTRA_KEY", "apply_voi", "voi_from_dataset"]

#: chave em ``ImageData.extra`` com os parâmetros VOI do arquivo de origem
VOI_EXTRA_KEY = "dicom_voi"


def _floats(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        return [float(value)]
    return [float(v) for v in value]


def voi_from_dataset(ds: Any) -> dict[str, Any]:
    """Parâmetros VOI do cabeçalho, no formato aceito por :func:`apply_voi`.

    Campos: ``photometric``, ``bits_stored``, ``pixel_representation``,
    ``rescale_slope``/``rescale_intercept`` (ou ``None``),
    ``modality_lut_bits`` (bits do descritor da ModalityLUTSequence, ou
    ``None``), ``window_center``/``window_width`` (listas), ``function``
    (``VOILUTFunction`` como gravado, ou ``None``) e ``lut`` (primeiro item da
    ``VOILUTSequence``: ``descriptor`` e ``data``; ou ``None``).
    """
    lut: dict[str, Any] | None = None
    seq = getattr(ds, "VOILUTSequence", None)
    if seq:
        item = seq[0]
        descriptor = [int(v) for v in item.LUTDescriptor]
        raw = item.LUTData
        data = [int(v) for v in raw] if hasattr(raw, "__len__") else [int(raw)]
        lut = {"descriptor": descriptor, "data": data}
    mod_bits: int | None = None
    mseq = getattr(ds, "ModalityLUTSequence", None)
    if mseq:
        mod_bits = int(mseq[0].LUTDescriptor[2])
    slope = getattr(ds, "RescaleSlope", None)
    intercept = getattr(ds, "RescaleIntercept", None)
    function = getattr(ds, "VOILUTFunction", None)
    return {
        "photometric": str(getattr(ds, "PhotometricInterpretation", "") or "").upper(),
        "bits_stored": int(getattr(ds, "BitsStored", 0) or 0),
        "pixel_representation": int(getattr(ds, "PixelRepresentation", 0) or 0),
        "rescale_slope": None if slope is None else float(slope),
        "rescale_intercept": None if intercept is None else float(intercept),
        "modality_lut_bits": mod_bits,
        "window_center": _floats(getattr(ds, "WindowCenter", None)),
        "window_width": _floats(getattr(ds, "WindowWidth", None)),
        "function": None if function in (None, "") else str(function).upper(),
        "lut": lut,
    }


def _dataset(voi: dict[str, Any]) -> Any:
    from pydicom.dataset import Dataset

    ds = Dataset()
    ds.PhotometricInterpretation = voi["photometric"]
    ds.BitsStored = int(voi["bits_stored"])
    ds.BitsAllocated = 16 if int(voi["bits_stored"]) > 8 else 8
    ds.PixelRepresentation = int(voi["pixel_representation"])
    if voi.get("rescale_slope") is not None and voi.get("rescale_intercept") is not None:
        ds.RescaleSlope = voi["rescale_slope"]
        ds.RescaleIntercept = voi["rescale_intercept"]
    if voi.get("modality_lut_bits") is not None:
        item = Dataset()
        item.LUTDescriptor = [0, 0, int(voi["modality_lut_bits"])]
        ds.ModalityLUTSequence = [item]
    centers, widths = list(voi.get("window_center") or []), list(voi.get("window_width") or [])
    if centers and widths:
        ds.WindowCenter = centers if len(centers) > 1 else centers[0]
        ds.WindowWidth = widths if len(widths) > 1 else widths[0]
    if voi.get("function"):
        ds.VOILUTFunction = voi["function"]
    lut = voi.get("lut")
    if lut:
        item = Dataset()
        item.LUTDescriptor = list(lut["descriptor"])
        item.LUTData = list(lut["data"])
        ds.VOILUTSequence = [item]
    return ds


def apply_voi(
    values: npt.NDArray[Any], voi: dict[str, Any], *, index: int = 0
) -> npt.NDArray[np.float64]:
    """Aplica a VOI do arquivo (LUT preferida à janela, como ``apply_voi_lut``).

    Args:
        values: saída da Modality LUT (mesmos valores que o ``pixel_array`` quando
            a reescala é identidade), na polaridade GRAVADA (sem inversão).
        voi: dicionário de :func:`voi_from_dataset`.
        index: qual das janelas/LUTs alternativas usar (o original usa a primeira).

    Raises:
        UnsupportedInputError: fotometria não monocromática, função VOI
            desconhecida, ou LUT com valores fora do descritor.
    """
    import numpy as np
    from pydicom.pixels import apply_voi_lut

    if voi.get("photometric") not in ("MONOCHROME1", "MONOCHROME2"):
        raise UnsupportedInputError(
            f"VOI só se aplica a MONOCHROME1/2; recebido {voi.get('photometric')!r}"
        )
    function = voi.get("function")
    if function not in (None, "LINEAR", "LINEAR_EXACT", "SIGMOID"):
        raise UnsupportedInputError(f"VOILUTFunction desconhecida: {function!r}")
    arr = np.asarray(values)
    if voi.get("lut"):
        # A LUT é indexada por valor inteiro: só aceita entrada integral.
        if not np.array_equal(arr, np.round(arr)):
            raise UnsupportedInputError("VOI LUT exige valores inteiros de entrada")
        arr = arr.astype(np.int64)
    elif np.array_equal(arr, np.round(arr)):
        arr = arr.astype(np.int64)
    try:
        out = apply_voi_lut(arr, _dataset(voi), index=index, prefer_lut=True)
    except (ValueError, IndexError) as exc:
        raise UnsupportedInputError(f"VOI do arquivo não aplicável: {exc}") from exc
    return np.asarray(out, dtype=np.float64)
