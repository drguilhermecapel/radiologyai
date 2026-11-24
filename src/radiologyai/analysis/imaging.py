"""Construção de :class:`~radiologyai.analysis.types.ImageData` a partir de arquivos.

Fontes suportadas:

* :func:`load_dicom` — um arquivo DICOM (projeção; multiframe de US vira cine).
* :func:`load_series` — diretório de cortes DICOM → volume (via
  :mod:`radiologyai.io.series`), ordenado por ``ImagePositionPatient``.
* :func:`load_nifti` — volume NIfTI (``nibabel``, extra ``imaging``); RAS → LPS.
* :func:`load_raster` — PNG/JPEG (modalidade ``OT`` salvo indicação explícita).
* :func:`load_cine` — sequência eco/US: DICOM multiframe ou NIfTI 2D+t.

Convenções (ver ``ImageData``): float32, MONOCHROME2, Modality LUT aplicada
(HU em TC), janela VOI guardada à parte e NUNCA aplicada aos pixels,
``value_range`` derivado do formato (BitsStored), não do conteúdo.

Nenhum identificador direto de paciente entra em ``ImageMeta``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from radiologyai.analysis.types import ImageData, ImageKind, ImageMeta, Modality
from radiologyai.errors import (
    BackendUnavailableError,
    InvalidDICOMError,
    MissingTagError,
    UnsupportedInputError,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = [
    "image_meta_from_dataset",
    "load_cine",
    "load_dicom",
    "load_nifti",
    "load_path",
    "load_raster",
    "load_series",
]

_MODALITIES = {m.value for m in Modality}
_US_REGION_CM = 3  # PhysicalUnitsXDirection == 3 => centímetros
# CID 4014 (incidências de mamografia), SNOMED CT -> rótulo de ViewPosition.
_MG_VIEW_CODES = {
    "399162004": "CC",
    "399368009": "MLO",
    "399260004": "ML",
    "399352003": "LM",
    "399099002": "XCCL",
    "399192003": "XCCM",
    "399188001": "FB",
    "399110001": "AT",
}


def _view_position(ds: Any, fallback: str | None) -> str | None:
    """``ViewPosition``; sem ele, mamografia usa ``ViewCodeSequence`` (CID 4014, SCT)."""
    if fallback:
        return fallback
    for item in getattr(ds, "ViewCodeSequence", None) or ():
        scheme = str(getattr(item, "CodingSchemeDesignator", "")).upper()
        if scheme in ("SCT", "SRT") and str(getattr(item, "CodeValue", "")) in _MG_VIEW_CODES:
            return _MG_VIEW_CODES[str(item.CodeValue)]
    return None


def _require(module: str, extra: str) -> None:
    if importlib.util.find_spec(module) is None:
        raise BackendUnavailableError(
            f"{module} não instalado; instale o extra: pip install 'radiologyai[{extra}]'"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _opt_str(ds: Any, name: str) -> str | None:
    value = getattr(ds, name, None)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _opt_float(ds: Any, name: str) -> float | None:
    value = getattr(ds, name, None)
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _modality(ds: Any, path: Path) -> Modality:
    code = str(getattr(ds, "Modality", "") or "").upper()
    if code not in _MODALITIES:
        raise UnsupportedInputError(f"{path}: modalidade DICOM não suportada {code!r}")
    return Modality(code)


def image_meta_from_dataset(
    ds: Any,
    *,
    content_sha256: str,
    source_format: Literal["dicom", "dicom_series"] = "dicom",
    sop_uids: tuple[str, ...] | None = None,
    pixel_spacing_source: Literal[
        "PixelSpacing", "ImagerPixelSpacing", "UltrasoundRegions", "nifti", "none"
    ] = "none",
    **overrides: Any,
) -> ImageMeta:
    """Metadados de aquisição a partir de um ``pydicom.Dataset`` (sem identificadores)."""
    from radiologyai.io.metadata import extract_metadata

    basic = extract_metadata(ds)
    sex = basic.patient_sex if basic.patient_sex in ("F", "M", "O") else None
    laterality = (_opt_str(ds, "ImageLaterality") or _opt_str(ds, "Laterality") or "").upper()
    burned = _opt_str(ds, "BurnedInAnnotation")
    contrast = getattr(ds, "ContrastBolusAgent", None)
    frame_time = _opt_float(ds, "FrameTime")
    if frame_time is None:
        cine_rate = _opt_float(ds, "CineRate") or _opt_float(ds, "RecommendedDisplayFrameRate")
        frame_time = 1000.0 / cine_rate if cine_rate else None
    data: dict[str, Any] = {
        "source_format": source_format,
        "content_sha256": content_sha256,
        "study_uid": _opt_str(ds, "StudyInstanceUID"),
        "series_uid": _opt_str(ds, "SeriesInstanceUID"),
        "sop_uids": sop_uids
        if sop_uids is not None
        else tuple(u for u in (_opt_str(ds, "SOPInstanceUID"),) if u),
        "sop_class_uid": basic.sop_class_uid,
        "patient_sex": sex,
        "patient_age_years": basic.patient_age_years,
        "body_part": basic.body_part,
        "view_position": _view_position(ds, basic.view_position),
        "laterality": laterality if laterality in ("L", "R", "B") else None,
        "study_description": _opt_str(ds, "StudyDescription"),
        "series_description": _opt_str(ds, "SeriesDescription"),
        "presentation_intent": _opt_str(ds, "PresentationIntentType"),
        "photometric_interpretation": basic.photometric_interpretation,
        "bits_stored": int(ds.BitsStored) if getattr(ds, "BitsStored", None) else None,
        "rows": basic.rows,
        "columns": basic.columns,
        "n_frames": basic.number_of_frames,
        "frame_time_ms": frame_time,
        "pixel_spacing_source": pixel_spacing_source,
        "slice_thickness_mm": basic.slice_thickness,
        "kvp": _opt_float(ds, "KVP"),
        "convolution_kernel": _opt_str(ds, "ConvolutionKernel"),
        "contrast_agent": None if contrast is None else str(contrast).strip(),
        "manufacturer": basic.manufacturer,
        "manufacturer_model": _opt_str(ds, "ManufacturerModelName"),
        "burned_in_annotation": None if burned is None else burned.upper() == "YES",
    }
    data.update(overrides)
    return ImageMeta.model_validate(data)


def _projection_spacing(
    ds: Any,
) -> tuple[
    tuple[float, float] | None,
    Literal["PixelSpacing", "ImagerPixelSpacing", "UltrasoundRegions", "none"],
]:
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        value = getattr(ds, tag, None)
        if value is not None:
            try:
                row, col = float(value[0]), float(value[1])
            except (TypeError, ValueError, IndexError):
                continue
            if row > 0 and col > 0:
                return (row, col), (
                    "PixelSpacing" if tag == "PixelSpacing" else "ImagerPixelSpacing"
                )
    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if regions:
        region = regions[0]
        units_x = int(getattr(region, "PhysicalUnitsXDirection", 0) or 0)
        units_y = int(getattr(region, "PhysicalUnitsYDirection", 0) or 0)
        dx = _opt_float(region, "PhysicalDeltaX")
        dy = _opt_float(region, "PhysicalDeltaY")
        if units_x == units_y == _US_REGION_CM and dx and dy and dx > 0 and dy > 0:
            return (abs(dy) * 10.0, abs(dx) * 10.0), "UltrasoundRegions"
    return None, "none"


def _decode(ds: Any, path: Path) -> tuple[npt.NDArray[np.float32], bool, tuple[float, float]]:
    """Pixels após Modality LUT, polaridade MONOCHROME2; devolve (pixels, cor, faixa)."""
    import numpy as np

    from radiologyai.io.series import _modality_lut, representable_range

    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "").upper()
    samples = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    if "PixelData" not in ds:
        raise MissingTagError(f"{path}: dataset não contém PixelData")
    try:
        stored = np.asarray(ds.pixel_array)
    except Exception as exc:  # noqa: BLE001 - reembalado com contexto
        raise InvalidDICOMError(f"{path}: falha ao decodificar PixelData: {exc}") from exc
    if samples == 3:
        # pydicom 3 já converte YBR_* para RGB em pixel_array.
        return np.asarray(stored, dtype=np.float32), True, (0.0, 255.0)
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise UnsupportedInputError(f"{path}: PhotometricInterpretation {photometric!r}")
    values = _modality_lut(ds, stored)
    lo, hi = representable_range(ds)
    if photometric == "MONOCHROME1":
        # Inversão pela faixa representável (independe do conteúdo da imagem).
        values = (np.float32(lo + hi) - values).astype(np.float32)
    return np.asarray(values, dtype=np.float32), False, (lo, hi)


def load_dicom(path: str | Path, **meta_overrides: Any) -> ImageData:
    """Um arquivo DICOM → ``ImageData``.

    Imagem de quadro único vira ``PROJECTION``. Multiframe de US vira ``CINE``.
    Multiframe de outras modalidades (TC/RM enhanced, tomossíntese) é recusado:
    a geometria por quadro não é inferida.

    Raises:
        InvalidDICOMError: arquivo ilegível.
        UnsupportedInputError: modalidade/fotometria/multiframe não suportados.
    """
    from radiologyai.io.reader import read_dicom
    from radiologyai.io.series import first_window

    p = Path(path)
    ds = read_dicom(p)
    modality = _modality(ds, p)
    n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    pixels, color, value_range = _decode(ds, p)
    spacing, source = _projection_spacing(ds)
    meta = image_meta_from_dataset(
        ds,
        content_sha256=_sha256_file(p),
        pixel_spacing_source=source,
        **meta_overrides,
    )
    if n_frames > 1:
        if modality is not Modality.US:
            raise UnsupportedInputError(
                f"{p}: multiframe {modality.value} não suportado (geometria por quadro); "
                "use uma série de cortes ou NIfTI"
            )
        kind = ImageKind.CINE
    else:
        kind = ImageKind.PROJECTION
    return ImageData(
        pixels=pixels,
        kind=kind,
        modality=modality,
        meta=meta,
        spacing_mm=spacing,
        window=None if color else first_window(ds),
        value_range=value_range,
        color=color,
    )


def load_cine(
    path: str | Path,
    *,
    modality: Modality = Modality.US,
    frame_time_ms: float | None = None,
    **meta_overrides: Any,
) -> ImageData:
    """Sequência eco/US → ``ImageData`` ``CINE`` ``(T, H, W)``.

    Aceita DICOM multiframe (US) ou NIfTI 2D+t (eixos ``(x, y, t)`` ou
    ``(x, y, 1, t)``, como no CAMUS). Para NIfTI, ``frame_time_ms`` vem do
    cabeçalho (``pixdim[4]`` com unidade de tempo) ou do argumento.

    Raises:
        UnsupportedInputError: arquivo sem eixo temporal.
    """
    p = Path(path)
    if p.name.endswith((".nii", ".nii.gz")):
        return _load_nifti(
            p, modality=modality, cine=True, frame_time_ms=frame_time_ms, **meta_overrides
        )
    image = load_dicom(p, **meta_overrides)
    if image.kind is not ImageKind.CINE:
        raise UnsupportedInputError(f"{p}: DICOM de quadro único não é cine")
    return image


def load_series(
    directory: str | Path, *, series_uid: str | None = None, **meta_overrides: Any
) -> ImageData:
    """Diretório de cortes DICOM → ``ImageData`` ``VOLUME`` ``(Z, Y, X)`` com afim LPS."""
    from radiologyai.io.series import load_series_volume

    volume = load_series_volume(Path(directory), series_uid=series_uid)
    ref = volume.datasets[0]
    modality = _modality(ref, volume.files[0])
    meta = image_meta_from_dataset(
        ref,
        content_sha256=volume.content_sha256,
        source_format="dicom_series",
        sop_uids=tuple(str(getattr(d, "SOPInstanceUID", "")) for d in volume.datasets),
        pixel_spacing_source="PixelSpacing",
        **{"n_frames": len(volume.datasets), **meta_overrides},
    )
    return ImageData(
        pixels=volume.pixels,
        kind=ImageKind.VOLUME,
        modality=modality,
        meta=meta,
        spacing_mm=volume.spacing_zyx,
        affine_lps=volume.affine_lps,
        window=volume.window,
        value_range=volume.value_range,
    )


def load_nifti(
    path: str | Path,
    *,
    modality: Modality,
    value_range: tuple[float, float] | None = None,
    **meta_overrides: Any,
) -> ImageData:
    """Volume NIfTI → ``ImageData`` ``VOLUME`` ``(Z, Y, X)``.

    NIfTI não registra modalidade: ``modality`` é obrigatório (nunca adivinhado).
    A afim RAS do NIfTI é convertida para LPS. ``value_range`` padrão: faixa
    do tipo de dado armazenado (TC: ``(-1024, 3071)`` costuma ser informado).

    Raises:
        BackendUnavailableError: nibabel ausente (extra ``imaging``).
        UnsupportedInputError: não é 3D.
    """
    return _load_nifti(
        Path(path), modality=modality, cine=False, value_range=value_range, **meta_overrides
    )


def _load_nifti(
    p: Path,
    *,
    modality: Modality,
    cine: bool,
    frame_time_ms: float | None = None,
    value_range: tuple[float, float] | None = None,
    **meta_overrides: Any,
) -> ImageData:
    _require("nibabel", "imaging")
    import nibabel as nib
    import numpy as np

    img: Any = nib.load(str(p))
    data = np.asarray(img.dataobj)
    zooms = [float(z) for z in img.header.get_zooms()]
    if cine:
        if data.ndim == 4 and data.shape[2] == 1:
            data = data[:, :, 0, :]
            zooms = [zooms[0], zooms[1], zooms[3] if len(zooms) > 3 else 0.0]
        if data.ndim != 3:
            raise UnsupportedInputError(f"{p}: cine NIfTI deve ser (x, y, t); shape {data.shape}")
        pixels = np.ascontiguousarray(np.transpose(data, (2, 1, 0)), dtype=np.float32)
        spacing: tuple[float, ...] | None = (zooms[1], zooms[0])
        if frame_time_ms is None and len(zooms) > 2 and zooms[2] > 0:
            unit = img.header.get_xyzt_units()[1]
            scale = {"sec": 1000.0, "msec": 1.0, "usec": 1e-3}.get(str(unit))
            frame_time_ms = zooms[2] * scale if scale else None
        kind, affine = ImageKind.CINE, None
    else:
        if data.ndim != 3:
            raise UnsupportedInputError(f"{p}: volume NIfTI deve ser 3D; shape {data.shape}")
        pixels = np.ascontiguousarray(np.transpose(data, (2, 1, 0)), dtype=np.float32)
        spacing = (zooms[2], zooms[1], zooms[0])
        ras_to_lps = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine = ras_to_lps @ np.asarray(img.affine, dtype=np.float64)
        kind = ImageKind.VOLUME
    if spacing is not None and not all(s > 0 for s in spacing):
        spacing = None
    if value_range is None:
        dtype = np.dtype(img.get_data_dtype())
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype.type)
            slope, inter = img.dataobj.slope, img.dataobj.inter
            slope = 1.0 if slope is None or not np.isfinite(slope) else float(slope)
            inter = 0.0 if inter is None or not np.isfinite(inter) else float(inter)
            ends = sorted((info.min * slope + inter, info.max * slope + inter))
            value_range = (float(ends[0]), float(ends[1]))
        else:
            finite = pixels[np.isfinite(pixels)]
            if finite.size == 0:
                raise UnsupportedInputError(f"{p}: volume sem valores finitos")
            # Float sem faixa declarada: o intervalo observado é o único dado honesto;
            # informe ``value_range`` explicitamente quando conhecido.
            lo, hi = float(finite.min()), float(finite.max())
            value_range = (lo, hi if hi > lo else lo + 1.0)
    meta = ImageMeta.model_validate(
        {
            "source_format": "nifti",
            "content_sha256": _sha256_file(p),
            "pixel_spacing_source": "nifti" if spacing else "none",
            "n_frames": int(pixels.shape[0]) if cine else 1,
            "frame_time_ms": frame_time_ms,
            **meta_overrides,
        }
    )
    return ImageData(
        pixels=pixels,
        kind=kind,
        modality=modality,
        meta=meta,
        spacing_mm=spacing,
        affine_lps=affine,
        value_range=value_range,
    )


def load_raster(
    path: str | Path,
    *,
    modality: Modality = Modality.OT,
    spacing_mm: tuple[float, float] | None = None,
    keep_color: bool = False,
    **meta_overrides: Any,
) -> ImageData:
    """PNG/JPEG → ``ImageData`` ``PROJECTION``.

    Raster não traz metadados de aquisição: modalidade padrão ``OT`` (os portões
    de aplicabilidade a recusam salvo indicação explícita do chamador — por
    exemplo, o avaliador que sabe que o dataset é de radiografias). Imagens
    coloridas viram tons de cinza (luminância ITU-R 601) salvo ``keep_color``.
    Fotometria assumida: MONOCHROME2 (mais claro = maior).

    Raises:
        BackendUnavailableError: Pillow ausente (extra ``imaging``).
    """
    _require("PIL", "imaging")
    import numpy as np
    from PIL import Image

    p = Path(path)
    with Image.open(p) as img:
        mode = img.mode
        if mode in ("I;16", "I;16B", "I;16L", "I"):
            arr = np.asarray(img, dtype=np.float32)
            bits = 16 if mode.startswith("I;16") else 32
            value_range = (0.0, float(2**bits - 1))
            color = False
        elif mode in ("L", "P", "LA", "1"):
            arr = np.asarray(img.convert("L"), dtype=np.float32)
            value_range, color = (0.0, 255.0), False
        else:
            rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
            if keep_color:
                arr, color = rgb, True
            else:
                arr = np.asarray(img.convert("L"), dtype=np.float32)
                color = False
            value_range = (0.0, 255.0)
    meta = ImageMeta.model_validate(
        {
            "source_format": "raster",
            "content_sha256": _sha256_file(p),
            "rows": int(arr.shape[0]),
            "columns": int(arr.shape[1]),
            **meta_overrides,
        }
    )
    return ImageData(
        pixels=np.ascontiguousarray(arr, dtype=np.float32),
        kind=ImageKind.PROJECTION,
        modality=modality,
        meta=meta,
        spacing_mm=spacing_mm,
        value_range=value_range,
        color=color,
    )


def load_path(path: str | Path, **kwargs: Any) -> ImageData:
    """Despacha pelo tipo do caminho: diretório → série; .nii(.gz) → NIfTI; raster; DICOM.

    NIfTI exige ``modality=`` (não é adivinhada).
    """
    p = Path(path)
    if p.is_dir():
        return load_series(p, **kwargs)
    name = p.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        if "modality" not in kwargs:
            raise UnsupportedInputError("NIfTI não registra modalidade; informe modality=")
        return load_nifti(p, **kwargs)
    if name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        return load_raster(p, **kwargs)
    return load_dicom(p, **kwargs)
