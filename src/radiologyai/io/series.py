"""Série DICOM (diretório de cortes) → volume 3D com geometria.

Ordena os cortes pela projeção de ``ImagePositionPatient`` na normal do plano
(produto vetorial das direções de ``ImageOrientationPatient``) — nunca por
``InstanceNumber`` nem pelo nome do arquivo, que não têm garantia geométrica.

Recusa, em vez de adivinhar:

* diretório com mais de uma série sem escolha explícita de ``series_uid``;
* cortes com orientação, tamanho ou espaçamento de pixel diferentes;
* espaçamento entre cortes irregular (sem reamostragem silenciosa);
* posições duplicadas (cortes repetidos ou multiecos misturados).

Os valores de pixel passam pela Modality LUT corte a corte (``RescaleSlope``
e ``RescaleIntercept`` podem variar entre cortes); em TC o resultado é HU.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from radiologyai.errors import InvalidDICOMError, MissingTagError, UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["SeriesVolume", "list_series", "load_series_volume"]

_SPACING_RTOL = 0.01  # 1 % de tolerância no espaçamento entre cortes
_ORIENT_ATOL = 1e-4


@dataclass(frozen=True, slots=True)
class SeriesVolume:
    """Volume ``(Z, Y, X)`` float32 após Modality LUT, com geometria LPS."""

    pixels: npt.NDArray[np.float32]
    spacing_zyx: tuple[float, float, float]
    affine_lps: npt.NDArray[np.float64]  # índice (x, y, z, 1) -> mm LPS
    value_range: tuple[float, float]
    window: tuple[float, float] | None
    datasets: tuple[Any, ...]  # cabeçalhos (sem pixels) na ordem geométrica
    files: tuple[Path, ...]  # arquivos na ordem geométrica
    content_sha256: str  # sha256 sobre os sha256 dos arquivos, na ordem geométrica


def _float_list(value: Any, n: int, tag: str, path: Path) -> list[float]:
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError) as exc:
        raise MissingTagError(f"{path}: {tag} inválido") from exc
    if len(out) != n:
        raise MissingTagError(f"{path}: {tag} deve ter {n} valores")
    return out


def list_series(directory: Path) -> dict[str, list[Path]]:
    """Agrupa os arquivos DICOM de um diretório (recursivo) por ``SeriesInstanceUID``.

    Arquivos que não são DICOM são ignorados; DICOM sem ``SeriesInstanceUID`` levanta.
    """
    import pydicom
    from pydicom.errors import InvalidDicomError

    directory = Path(directory)
    if not directory.is_dir():
        raise InvalidDICOMError(f"diretório não encontrado: {directory}")
    groups: dict[str, list[Path]] = {}
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
        except InvalidDicomError:
            continue
        uid = getattr(ds, "SeriesInstanceUID", None)
        if not uid:
            raise MissingTagError(f"{path}: SeriesInstanceUID ausente")
        groups.setdefault(str(uid), []).append(path)
    return groups


def _modality_lut(ds: Any, raw: npt.NDArray[Any]) -> npt.NDArray[np.float32]:
    import numpy as np
    from pydicom.pixels.processing import apply_modality_lut

    out: npt.NDArray[np.float32] = np.asarray(apply_modality_lut(raw, ds), dtype=np.float32)
    return out


def _stored_range(ds: Any) -> tuple[float, float]:
    bits = int(getattr(ds, "BitsStored", 16) or 16)
    signed = int(getattr(ds, "PixelRepresentation", 0) or 0) == 1
    if signed:
        return float(-(2 ** (bits - 1))), float(2 ** (bits - 1) - 1)
    return 0.0, float(2**bits - 1)


def representable_range(ds: Any) -> tuple[float, float]:
    """Faixa representável após a Modality LUT, a partir de BitsStored (não do conteúdo)."""
    import numpy as np

    lo, hi = _stored_range(ds)
    dtype = np.int32 if lo < 0 else np.uint32
    ends = _modality_lut(ds, np.array([lo, hi], dtype=dtype))
    a, b = float(ends.min()), float(ends.max())
    return (a, b) if a < b else (a, a + 1.0)


def first_window(ds: Any) -> tuple[float, float] | None:
    """Primeira janela VOI (centro, largura) do cabeçalho, ou ``None``."""
    center = getattr(ds, "WindowCenter", None)
    width = getattr(ds, "WindowWidth", None)
    if center is None or width is None:
        return None
    if not isinstance(center, (str, bytes)) and hasattr(center, "__len__"):
        center = center[0]
    if not isinstance(width, (str, bytes)) and hasattr(width, "__len__"):
        width = width[0]
    try:
        c, w = float(center), float(width)
    except (TypeError, ValueError):
        return None
    return (c, w) if w > 0 else None


def load_series_volume(directory: Path, *, series_uid: str | None = None) -> SeriesVolume:
    """Monta o volume de uma série a partir de um diretório.

    Args:
        directory: diretório com os arquivos DICOM (busca recursiva).
        series_uid: série escolhida quando o diretório contém mais de uma.

    Raises:
        UnsupportedInputError: várias séries sem escolha, geometria inconsistente,
            espaçamento irregular, posições duplicadas, série de um só corte.
        MissingTagError: tags geométricas ausentes.
        InvalidDICOMError: diretório inexistente ou arquivos ilegíveis.
    """
    import numpy as np
    import pydicom

    groups = list_series(directory)
    if not groups:
        raise InvalidDICOMError(f"nenhum arquivo DICOM em {directory}")
    if series_uid is None:
        if len(groups) > 1:
            raise UnsupportedInputError(
                f"{directory} contém {len(groups)} séries; escolha uma (series_uid): "
                + ", ".join(sorted(groups))
            )
        series_uid = next(iter(groups))
    if series_uid not in groups:
        raise UnsupportedInputError(f"série {series_uid!r} não encontrada em {directory}")
    files = groups[series_uid]
    if len(files) < 2:
        raise UnsupportedInputError("série com um único corte não forma volume")

    headers = [pydicom.dcmread(str(p), stop_before_pixels=True) for p in files]
    ref = headers[0]
    for tag in ("ImageOrientationPatient", "ImagePositionPatient", "PixelSpacing"):
        missing = [str(p) for p, h in zip(files, headers, strict=True) if tag not in h]
        if missing:
            raise MissingTagError(f"{tag} ausente em {len(missing)} corte(s), ex.: {missing[0]}")
    iop = np.array(_float_list(ref.ImageOrientationPatient, 6, "IOP", files[0]), dtype=np.float64)
    row_dir, col_dir = iop[:3], iop[3:]
    normal = np.cross(row_dir, col_dir)
    if not np.isclose(np.linalg.norm(normal), 1.0, atol=1e-3):
        raise UnsupportedInputError("ImageOrientationPatient não é ortonormal")
    pixel_spacing = _float_list(ref.PixelSpacing, 2, "PixelSpacing", files[0])
    shape = (int(ref.Rows), int(ref.Columns))

    positions: list[tuple[float, int]] = []
    for i, (path, header) in enumerate(zip(files, headers, strict=True)):
        h_iop = np.array(_float_list(header.ImageOrientationPatient, 6, "IOP", path))
        if not np.allclose(h_iop, iop, atol=_ORIENT_ATOL):
            raise UnsupportedInputError(f"{path}: orientação difere do restante da série")
        if (int(header.Rows), int(header.Columns)) != shape:
            raise UnsupportedInputError(f"{path}: dimensão do corte difere do restante da série")
        h_ps = _float_list(header.PixelSpacing, 2, "PixelSpacing", path)
        if not np.allclose(h_ps, pixel_spacing, rtol=1e-4):
            raise UnsupportedInputError(f"{path}: PixelSpacing difere do restante da série")
        if int(getattr(header, "NumberOfFrames", 1) or 1) > 1:
            raise UnsupportedInputError(f"{path}: multiframe dentro de série de cortes")
        ipp = np.array(_float_list(header.ImagePositionPatient, 3, "IPP", path))
        positions.append((float(ipp @ normal), i))

    positions.sort()
    offsets = np.array([p for p, _ in positions])
    gaps = np.diff(offsets)
    if np.any(gaps <= 1e-6):
        raise UnsupportedInputError("posições de corte duplicadas na série")
    slice_spacing = float(np.median(gaps))
    if np.any(np.abs(gaps - slice_spacing) > _SPACING_RTOL * slice_spacing):
        raise UnsupportedInputError(
            "espaçamento entre cortes irregular "
            f"(min {gaps.min():.4f} mm, max {gaps.max():.4f} mm); reamostragem não é implícita"
        )

    order = [i for _, i in positions]
    ordered_files = tuple(files[i] for i in order)
    slices: list[npt.NDArray[np.float32]] = []
    datasets: list[Any] = []
    digests: list[str] = []
    lo_all, hi_all = np.inf, -np.inf
    for path in ordered_files:
        raw_bytes = path.read_bytes()
        digests.append(hashlib.sha256(raw_bytes).hexdigest())
        ds = pydicom.dcmread(str(path))
        photometric = str(getattr(ds, "PhotometricInterpretation", "")).upper()
        if photometric not in ("MONOCHROME1", "MONOCHROME2"):
            raise UnsupportedInputError(f"{path}: volume exige tons de cinza; é {photometric}")
        try:
            raw = ds.pixel_array
        except Exception as exc:  # noqa: BLE001 - reembalado com contexto
            raise InvalidDICOMError(f"{path}: falha ao decodificar PixelData: {exc}") from exc
        values = _modality_lut(ds, np.asarray(raw))
        lo, hi = representable_range(ds)
        if photometric == "MONOCHROME1":
            values = (np.float32(lo + hi) - values).astype(np.float32)
        lo_all, hi_all = min(lo_all, lo), max(hi_all, hi)
        slices.append(values)
        del ds.PixelData
        datasets.append(ds)

    volume = np.ascontiguousarray(np.stack(slices, axis=0), dtype=np.float32)
    first_ipp = np.array(_float_list(headers[order[0]].ImagePositionPatient, 3, "IPP", files[0]))
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 0] = row_dir * pixel_spacing[1]  # índice x (coluna) anda ao longo da linha
    affine[:3, 1] = col_dir * pixel_spacing[0]  # índice y (linha) anda ao longo da coluna
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = first_ipp
    content = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
    return SeriesVolume(
        pixels=volume,
        spacing_zyx=(slice_spacing, pixel_spacing[0], pixel_spacing[1]),
        affine_lps=affine,
        value_range=(float(lo_all), float(hi_all)),
        window=first_window(headers[order[0]]),
        datasets=tuple(datasets),
        files=ordered_files,
        content_sha256=content,
    )
