"""Ingestão de exames: DICOM (arquivos ou zip), NIfTI e PNG/JPEG.

Ordem obrigatória, todas as etapas antes de qualquer gravação no banco:

1. Expansão de zip com recusa de caminho absoluto, ``..``, links, arquivos
   cifrados e zip aninhado; contagem de membros e de bytes DESCOMPACTADOS
   lidos de fato (o cabeçalho do zip não é confiável).
2. Classificação pelo conteúdo (assinatura de PNG/JPEG, preâmbulo ``DICM``,
   sufixo ``.nii``/``.nii.gz``). Arquivo não reconhecido é recusado, nunca
   ignorado em silêncio.
3. Des-identificação de todo DICOM com :func:`radiologyai.io.deident.deidentify`
   e a chave ``MEDAI_DEIDENT_KEY`` (sem chave, ingestão de DICOM é recusada).
   PNG/JPEG são recodificados como PNG sem metadados (EXIF/tEXt); NIfTI tem os
   campos de texto livre do cabeçalho apagados.
4. Decodificação real de cada imagem (``radiologyai.analysis.imaging``): o que
   não decodifica não entra.
5. Tipo de exame pelos metadados (:func:`infer_exam_type`) ou declarado por um
   humano; ``unknown`` exige escolha humana antes da análise.

Metadados informados por humano (incidência, idade, região...) só PREENCHEM
campos ausentes do DICOM — nunca sobrescrevem um valor gravado — e ficam
registrados como tal na série e na auditoria.

Nenhum identificador direto de paciente é gravado, nem em log.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from radiologyai.errors import ConfigurationError, IngestionError

if TYPE_CHECKING:
    from radiologyai.analysis.types import ImageData
    from radiologyai.catalog.schema import ExamType
    from radiologyai.persistence.blobs import BlobStore
    from radiologyai.persistence.config import PlatformSettings
    from radiologyai.persistence.db import Database

__all__ = [
    "OVERRIDABLE_FIELDS",
    "DeidentKeyMissingError",
    "HumanMeta",
    "IngestRejectedError",
    "IngestResult",
    "UploadTooLargeError",
    "apply_overrides",
    "expand_uploads",
    "ingest",
]

log = logging.getLogger("radiologyai.interop.ingest")

OVERRIDABLE_FIELDS = (
    "body_part",
    "view_position",
    "laterality",
    "patient_age_years",
    "patient_sex",
    "sequence_role",
)
_VOLUMETRIC = {"CT", "MR"}
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_ZIP_MAGIC = b"PK\x03\x04"
_NON_IMAGE_SOP_PREFIXES = (
    "1.2.840.10008.1.3.10",  # DICOMDIR
    "1.2.840.10008.5.1.4.1.1.88.",  # SR
    "1.2.840.10008.5.1.4.1.1.11.",  # estados de apresentação
)


class IngestRejectedError(IngestionError):
    """Envio recusado (formato, mistura de estudos, zip inseguro, nada decodificável)."""


class UploadTooLargeError(IngestRejectedError):
    """Envio excede o limite de bytes, de membros ou de bytes descompactados."""


class DeidentKeyMissingError(ConfigurationError):
    """``MEDAI_DEIDENT_KEY`` ausente: DICOM não é ingerido sem des-identificação."""


class HumanMeta(BaseModel):
    """Metadados declarados por um humano no envio (preenchem lacunas; nunca sobrescrevem)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    modality: Literal["CR", "DX", "CT", "MR", "MG", "US", "OT"] | None = None
    body_part: str | None = Field(default=None, max_length=64)
    view_position: str | None = Field(default=None, max_length=32)
    laterality: Literal["L", "R", "B"] | None = None
    patient_age_years: float | None = Field(default=None, ge=0, le=130)
    patient_sex: Literal["F", "M", "O"] | None = None
    sequence_role: str | None = Field(default=None, max_length=32)

    def overrides(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in OVERRIDABLE_FIELDS if getattr(self, k) is not None}


@dataclass(frozen=True, slots=True)
class IngestResult:
    study_id: str
    n_images: int
    exam_type: str
    exam_type_source: str
    warnings: tuple[str, ...]


@dataclass(slots=True)
class _Item:
    name: str
    path: Path


@dataclass(slots=True)
class _StoredFile:
    data: bytes
    media_type: str
    filename: str
    sop_uid: str | None = None


@dataclass(slots=True)
class _PreparedImage:
    fmt: Literal["dicom", "dicom_series", "nifti", "raster"]
    files: list[_StoredFile]
    modality: str | None = None
    series_uid: str | None = None
    image: ImageData | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 1. Expansão (zip seguro)
# --------------------------------------------------------------------------- #
def _check_member_name(name: str) -> None:
    norm = name.replace("\\", "/")
    if norm.startswith("/") or re.match(r"^[A-Za-z]:", norm):
        raise IngestRejectedError(f"zip com caminho absoluto recusado: {name!r}")
    if ".." in PurePosixPath(norm).parts:
        raise IngestRejectedError(f"zip com '..' (zip-slip) recusado: {name!r}")


def _is_junk(name: str) -> bool:
    norm = name.replace("\\", "/")
    base = norm.rsplit("/", 1)[-1]
    return norm.startswith("__MACOSX/") or base.startswith("._") or base in (".DS_Store",)


def expand_uploads(
    items: Iterable[tuple[str, Path]], workdir: Path, settings: PlatformSettings
) -> list[tuple[str, Path]]:
    """Expande zips de forma segura; devolve ``(nome, caminho)`` de cada arquivo.

    Membros são gravados em ``workdir`` com nomes gerados (o nome do membro
    nunca vira caminho). O total descompactado é contado nos bytes lidos.

    Raises:
        IngestRejectedError: membro com caminho perigoso, link, cifrado ou zip aninhado.
        UploadTooLargeError: membros ou bytes descompactados acima do limite.
    """
    out: list[tuple[str, Path]] = []
    total = 0
    members = 0
    for name, path in items:
        with path.open("rb") as fh:
            head = fh.read(4)
        if head != _ZIP_MAGIC:
            out.append((name, path))
            continue
        try:
            zf = zipfile.ZipFile(path)
        except zipfile.BadZipFile as exc:
            raise IngestRejectedError(f"{name}: zip inválido: {exc}") from exc
        with zf:
            for info in zf.infolist():
                _check_member_name(info.filename)
                if info.is_dir() or _is_junk(info.filename):
                    continue
                mode = (info.external_attr >> 16) & 0o170000
                if mode == stat.S_IFLNK:
                    raise IngestRejectedError(f"zip contém link simbólico: {info.filename!r}")
                if info.flag_bits & 0x1:
                    raise IngestRejectedError(f"zip contém membro cifrado: {info.filename!r}")
                members += 1
                if members > settings.max_zip_members:
                    raise UploadTooLargeError(
                        f"zip excede {settings.max_zip_members} arquivos por envio"
                    )
                target = workdir / f"zip-{members:06d}"
                with zf.open(info) as src, target.open("wb") as dst:
                    while chunk := src.read(1 << 20):
                        total += len(chunk)
                        if total > settings.max_uncompressed_bytes:
                            raise UploadTooLargeError(
                                "conteúdo descompactado excede "
                                f"{settings.max_uncompressed_bytes} bytes"
                            )
                        dst.write(chunk)
                with target.open("rb") as fh:
                    if fh.read(4) == _ZIP_MAGIC:
                        raise IngestRejectedError(f"zip aninhado recusado: {info.filename!r}")
                out.append((info.filename, target))
    return out


# --------------------------------------------------------------------------- #
# 2-3. Classificação e des-identificação
# --------------------------------------------------------------------------- #
def _classify(name: str, path: Path) -> Literal["dicom", "nifti", "raster"]:
    lower = name.lower()
    if lower.endswith((".nii", ".nii.gz")):
        return "nifti"
    with path.open("rb") as fh:
        head = fh.read(132)
    if head.startswith(_PNG_MAGIC) or head.startswith(_JPEG_MAGIC):
        return "raster"
    if head[128:132] == b"DICM":
        return "dicom"
    raise IngestRejectedError(
        f"{name}: formato não reconhecido (aceitos: DICOM Part 10, zip, NIfTI, PNG, JPEG)"
    )


def _deident_dicom(path: Path, name: str, key: bytes) -> tuple[Any, bytes] | None:
    """Lê, des-identifica e serializa; ``None`` para objeto DICOM sem imagem."""
    import pydicom

    from radiologyai.io.deident import DeidentificationProfile, deidentify

    try:
        ds = pydicom.dcmread(str(path))
    except Exception as exc:  # noqa: BLE001 - reembalado com contexto
        raise IngestRejectedError(f"{name}: DICOM ilegível: {exc}") from exc
    sop_class = str(getattr(ds, "SOPClassUID", "") or "")
    if "PixelData" not in ds or sop_class.startswith(_NON_IMAGE_SOP_PREFIXES):
        return None
    out, _pseudo = deidentify(ds, DeidentificationProfile(key=key))
    if getattr(out, "file_meta", None) is not None and "SOPInstanceUID" in out:
        out.file_meta.MediaStorageSOPInstanceUID = out.SOPInstanceUID
    buf = io.BytesIO()
    out.save_as(buf, enforce_file_format=True)
    return out, buf.getvalue()


def _clean_raster(path: Path, name: str) -> bytes:
    """PNG sem metadados a partir de PNG/JPEG (EXIF, tEXt e afins descartados)."""
    import numpy as np
    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(path) as img:
            img.load()
            mode = img.mode
            if mode in ("I;16", "I;16B", "I;16L"):
                arr = np.asarray(img, dtype=np.uint16)
            elif mode == "I":
                arr32 = np.asarray(img)
                if arr32.min() < 0 or arr32.max() > 65535:
                    raise IngestRejectedError(f"{name}: PNG de 32 bits fora da faixa de 16 bits")
                arr = arr32.astype(np.uint16)
            elif mode in ("L", "LA", "1"):
                arr = np.asarray(img.convert("L"), dtype=np.uint8)
            else:
                arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise IngestRejectedError(f"{name}: imagem ilegível: {exc}") from exc
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG")
    return buf.getvalue()


def _clean_nifti(path: Path, name: str, workdir: Path) -> bytes:
    """NIfTI com campos de texto livre do cabeçalho apagados, gravado como .nii.gz."""
    import importlib.util

    if importlib.util.find_spec("nibabel") is None:
        from radiologyai.errors import BackendUnavailableError

        raise BackendUnavailableError(
            "nibabel não instalado; instale o extra: pip install 'radiologyai[imaging]'"
        )
    import nibabel as nib

    suffix = ".nii.gz" if name.lower().endswith(".gz") else ".nii"
    src = workdir / f"src-{path.name}{suffix}"
    shutil.copyfile(path, src)
    try:
        img: Any = nib.load(str(src))
        header = img.header
        for key in ("descrip", "aux_file", "intent_name", "db_name"):
            if key in header:
                header[key] = b""
        header.extensions.clear()
        target = workdir / f"clean-{path.name}.nii.gz"
        nib.save(img, str(target))
    except Exception as exc:  # noqa: BLE001 - reembalado com contexto
        raise IngestRejectedError(f"{name}: NIfTI ilegível: {exc}") from exc
    return target.read_bytes()


# --------------------------------------------------------------------------- #
# 4. Decodificação
# --------------------------------------------------------------------------- #
def apply_overrides(image: ImageData, overrides: dict[str, Any]) -> ImageData:
    """Preenche campos AUSENTES de ``image.meta`` com os declarados por humano."""
    import dataclasses

    from radiologyai.analysis.types import ImageMeta

    if not overrides:
        return image
    current = image.meta.model_dump()
    fill = {
        k: v
        for k, v in overrides.items()
        if k in OVERRIDABLE_FIELDS
        and current.get(k)
        in (
            None,
            "",
        )
    }
    if not fill:
        return image
    meta = ImageMeta.model_validate({**current, **fill})
    return dataclasses.replace(image, meta=meta)


def decode_files(
    fmt: str,
    files: list[tuple[str, Path]],
    *,
    modality: str | None,
    overrides: dict[str, Any],
    workdir: Path,
) -> ImageData:
    """Decodifica arquivos já des-identificados num ``ImageData`` (mesmo caminho do worker)."""
    from radiologyai.analysis import imaging
    from radiologyai.analysis.types import Modality

    if fmt == "dicom":
        image = imaging.load_dicom(files[0][1])
    elif fmt == "dicom_series":
        series_dir = Path(tempfile.mkdtemp(dir=workdir, prefix="series-"))
        for name, path in files:
            (series_dir / name).symlink_to(path.resolve())
        image = imaging.load_series(series_dir)
    elif fmt == "nifti":
        if modality is None or modality in ("OT",):
            raise IngestRejectedError("NIfTI não registra modalidade; informe modality (CT ou MR)")
        name, path = files[0]
        link = workdir / f"nifti-{path.name}.nii.gz"
        if not link.exists():
            link.symlink_to(path.resolve())
        image = imaging.load_nifti(link, modality=Modality(modality))
    elif fmt == "raster":
        image = imaging.load_raster(
            files[0][1], modality=Modality(modality or "OT"), keep_color=False
        )
    else:  # pragma: no cover - formato vem do próprio código
        raise IngestRejectedError(f"formato desconhecido {fmt!r}")
    return apply_overrides(image, overrides)


# --------------------------------------------------------------------------- #
# Orquestração
# --------------------------------------------------------------------------- #
def _prepare(
    items: list[tuple[str, Path]],
    *,
    workdir: Path,
    key: Callable[[], bytes],
    human: HumanMeta,
    warnings: list[str],
) -> tuple[list[_PreparedImage], str | None, str | None]:
    kinds = {name: _classify(name, path) for name, path in items}
    if not items:
        raise IngestRejectedError("nenhum arquivo enviado")
    if "dicom" in kinds.values() and set(kinds.values()) != {"dicom"}:
        raise IngestRejectedError("não misture DICOM com NIfTI/PNG/JPEG no mesmo envio")
    overrides = human.overrides()
    prepared: list[_PreparedImage] = []
    study_uid: str | None = None
    patient_pseudo: str | None = None
    if set(kinds.values()) == {"dicom"}:
        secret = key()
        by_series: dict[str, list[tuple[Any, bytes]]] = {}
        studies: set[str] = set()
        skipped = 0
        for name, path in items:
            done = _deident_dicom(path, name, secret)
            if done is None:
                skipped += 1
                continue
            ds, data = done
            studies.add(str(getattr(ds, "StudyInstanceUID", "") or ""))
            by_series.setdefault(str(getattr(ds, "SeriesInstanceUID", "") or ""), []).append(
                (ds, data)
            )
            patient_pseudo = patient_pseudo or (str(ds.PatientID) if ds.get("PatientID") else None)
        if skipped:
            warnings.append(f"{skipped} objeto(s) DICOM sem imagem ignorado(s) (DICOMDIR/SR/PR)")
        if len(studies) > 1:
            raise IngestRejectedError(
                f"o envio contém {len(studies)} estudos; envie um estudo por vez"
            )
        study_uid = next(iter(studies), None) or None
        for series_uid, members in by_series.items():
            modality = str(getattr(members[0][0], "Modality", "") or "")
            frames = [int(getattr(ds, "NumberOfFrames", 1) or 1) for ds, _ in members]
            if modality in _VOLUMETRIC and len(members) > 1 and max(frames) == 1:
                prepared.append(
                    _PreparedImage(
                        fmt="dicom_series",
                        files=[
                            _StoredFile(
                                data, "application/dicom", f"{i:06d}.dcm", ds.SOPInstanceUID
                            )
                            for i, (ds, data) in enumerate(members)
                        ],
                        modality=modality,
                        series_uid=series_uid or None,
                        overrides=overrides,
                    )
                )
                continue
            for ds, data in members:
                prepared.append(
                    _PreparedImage(
                        fmt="dicom",
                        files=[
                            _StoredFile(data, "application/dicom", "000000.dcm", ds.SOPInstanceUID)
                        ],
                        modality=modality,
                        series_uid=series_uid or None,
                        overrides=overrides,
                    )
                )
    else:
        for name, path in items:
            if kinds[name] == "raster":
                data = _clean_raster(path, name)
                prepared.append(
                    _PreparedImage(
                        fmt="raster",
                        files=[_StoredFile(data, "image/png", "000000.png")],
                        modality=human.modality or "OT",
                        overrides=overrides,
                    )
                )
            else:
                data = _clean_nifti(path, name, workdir)
                prepared.append(
                    _PreparedImage(
                        fmt="nifti",
                        files=[_StoredFile(data, "application/gzip", "000000.nii.gz")],
                        modality=human.modality,
                        overrides=overrides,
                    )
                )
    if not prepared:
        raise IngestRejectedError("nenhuma imagem decodificável no envio")
    return prepared, study_uid, patient_pseudo


def ingest(
    items: list[tuple[str, Path]],
    *,
    db: Database,
    blobs: BlobStore,
    settings: PlatformSettings,
    user_id: str,
    username: str,
    human: HumanMeta | None = None,
    exam_type: ExamType | None = None,
    source: Literal["upload", "dicomweb"] = "upload",
    on_decoded: Callable[[str, ImageData], None] | None = None,
) -> IngestResult:
    """Ingere um estudo completo; tudo ou nada.

    Raises:
        DeidentKeyMissingError: DICOM sem ``MEDAI_DEIDENT_KEY`` configurada.
        IngestRejectedError: formato, zip inseguro, mistura de estudos, decodificação.
        UploadTooLargeError: limites de tamanho/membros.
    """
    from radiologyai.analysis.routing import infer_exam_type
    from radiologyai.errors import RadiologyAIError
    from radiologyai.persistence.audit import append_event
    from radiologyai.persistence.models import Instance, Series, Study, utcnow

    human = human or HumanMeta()
    warnings: list[str] = []

    def key() -> bytes:
        if not settings.deident_key:
            raise DeidentKeyMissingError(
                "MEDAI_DEIDENT_KEY não configurada: DICOM não é ingerido sem des-identificação"
            )
        return settings.deident_key.encode("utf-8")

    workdir = Path(tempfile.mkdtemp(prefix="radiologyai-ingest-"))
    try:
        expanded = expand_uploads(items, workdir, settings)
        prepared, study_uid, patient_pseudo = _prepare(
            expanded, workdir=workdir, key=key, human=human, warnings=warnings
        )
        # 4. decodificação real de cada imagem, a partir dos bytes JÁ des-identificados
        for n, prep in enumerate(prepared):
            paths: list[tuple[str, Path]] = []
            for i, stored in enumerate(prep.files):
                p = workdir / f"img{n:04d}-{i:06d}-{stored.filename}"
                p.write_bytes(stored.data)
                paths.append((stored.filename, p))
            try:
                prep.image = decode_files(
                    prep.fmt,
                    paths,
                    modality=prep.modality,
                    overrides=prep.overrides,
                    workdir=workdir,
                )
            except RadiologyAIError as exc:
                raise IngestRejectedError(f"imagem {n + 1}: {exc}") from exc
        images = [p.image for p in prepared if p.image is not None]
        if exam_type is not None:
            chosen, origin = exam_type, "human"
        else:
            chosen, origin = infer_exam_type(images), "metadata"
        for image in images:
            if image.meta.burned_in_annotation or (
                image.modality.value == "US" and image.meta.burned_in_annotation is None
            ):
                warnings.append(
                    "possível texto gravado no pixel (BurnedInAnnotation); a des-identificação "
                    "não remove texto da imagem (perigo H-09)"
                )
                break
        if prepared[0].overrides:
            warnings.append(
                "metadados declarados por humano preencheram campos ausentes: "
                + ", ".join(sorted(prepared[0].overrides))
            )

        with db.session() as session:
            study = Study(
                study_uid=study_uid,
                patient_pseudo_id=patient_pseudo,
                modalities=",".join(sorted({i.modality.value for i in images})),
                description=next(
                    (i.meta.study_description for i in images if i.meta.study_description), None
                ),
                exam_type=chosen.value,
                exam_type_source=origin,
                exam_type_set_by=user_id if origin == "human" else None,
                exam_type_set_at=utcnow() if origin == "human" else None,
                source=source,
                created_by=user_id,
                warnings=warnings,
            )
            session.add(study)
            session.flush()
            series_ids: list[str] = []
            for pos, prep in enumerate(prepared):
                decoded = prep.image
                assert decoded is not None  # decodificada acima
                image = decoded
                rows, cols = image.plane_shape
                n_frames = 1 if image.kind.value == "projection" else int(image.pixels.shape[0])
                series = Series(
                    study_id=study.id,
                    position=pos,
                    series_uid=prep.series_uid,
                    modality=image.modality.value,
                    source_format=prep.fmt,
                    kind=image.kind.value,
                    description=image.meta.series_description,
                    body_part=image.meta.body_part,
                    view_position=image.meta.view_position,
                    laterality=image.meta.laterality,
                    rows=rows,
                    columns=cols,
                    n_frames=n_frames,
                    n_instances=len(prep.files),
                    content_sha256=image.meta.content_sha256,
                    meta=_series_meta(image, prep),
                )
                session.add(series)
                session.flush()
                for i, stored in enumerate(prep.files):
                    ref = blobs.put(session, stored.data, stored.media_type, kind="input")
                    session.add(
                        Instance(
                            series_id=series.id,
                            ordinal=i,
                            sop_uid=stored.sop_uid,
                            artifact_sha256=ref.sha256,
                            filename=stored.filename,
                        )
                    )
                series_ids.append(series.id)
            study_id = study.id
        append_event(
            db,
            "study.ingested",
            actor_id=user_id,
            actor_username=username,
            subject_type="study",
            subject_id=study_id,
            payload={
                "source": source,
                "n_images": len(images),
                "input_sha256": [i.meta.content_sha256 for i in images],
                "exam_type": chosen.value,
                "exam_type_source": origin,
                "human_metadata": sorted(prepared[0].overrides),
            },
        )
        if on_decoded is not None:
            for sid, image in zip(series_ids, images, strict=True):
                on_decoded(sid, image)
        log.info("estudo %s ingerido: %d imagem(ns), tipo %s", study_id, len(images), chosen.value)
        return IngestResult(study_id, len(images), chosen.value, origin, tuple(warnings))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _series_meta(image: ImageData, prep: _PreparedImage) -> dict[str, Any]:
    return {
        "image_meta": image.meta.model_dump(mode="json"),
        "spacing_mm": list(image.spacing_mm) if image.spacing_mm else None,
        "window": list(image.window) if image.window else None,
        "value_range": list(image.value_range),
        "color": image.color,
        "affine_lps": image.affine_lps.tolist() if image.affine_lps is not None else None,
        "human_overrides": prep.overrides,
    }
