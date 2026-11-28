"""Ingestão: zip seguro, limites medidos no fluxo, formatos recusados com motivo."""

from __future__ import annotations

import importlib.util
import io
import zipfile
from pathlib import Path

import pytest
from pydicom.uid import generate_uid

HAS = all(importlib.util.find_spec(m) for m in ("sqlalchemy", "alembic", "argon2", "PIL"))
pytestmark = pytest.mark.skipif(not HAS, reason="requer os extras [platform,imaging]")


@pytest.fixture
def settings(tmp_path):
    from radiologyai.persistence.config import PlatformSettings

    return PlatformSettings(
        database_url=f"sqlite:///{tmp_path / 'p.db'}",
        store_dir=tmp_path / "store",
        deident_key="chave-de-teste",
        max_zip_members=10,
        max_uncompressed_bytes=3_000_000,
    )


@pytest.fixture
def env(settings):
    from radiologyai.persistence.blobs import BlobStore
    from radiologyai.persistence.db import Database

    db = Database(settings.database_url)
    db.migrate()
    return db, BlobStore(settings.store_dir), settings


def dcm(dataset) -> bytes:
    import pydicom

    buf = io.BytesIO()
    pydicom.dcmwrite(buf, dataset, enforce_file_format=True)
    return buf.getvalue()


def write_zip(path: Path, members: dict[str, bytes], symlink: str | None = None) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = 0o120777 << 16
            zf.writestr(info, "/etc/passwd")
    return path


def run(env, items, **kw):
    from radiologyai.interop.ingest import ingest

    db, blobs, settings = env
    return ingest(
        items, db=db, blobs=blobs, settings=settings, user_id=_user(db), username="t", **kw
    )


def _user(db) -> str:
    from sqlalchemy import select

    from radiologyai.persistence.models import User
    from radiologyai.persistence.users import create_user

    with db.session() as session:
        existing = session.execute(select(User)).scalars().first()
        if existing:
            return existing.id
    return create_user(
        db, username="t", full_name="T", role="technician", password="senha-de-teste-1"
    ).id


@pytest.mark.requirement("REQ-152")
class TestZipSafety:
    @pytest.mark.parametrize("name", ["../fora.dcm", "/etc/x.dcm", "a/../../b.dcm", "C:/x.dcm"])
    def test_path_traversal_refused(self, env, tmp_path, cr_dataset, name):
        from radiologyai.interop.ingest import IngestRejectedError

        z = write_zip(tmp_path / "x.zip", {name: dcm(cr_dataset)})
        with pytest.raises(IngestRejectedError, match="zip"):
            run(env, [("x.zip", z)])

    def test_symlink_member_refused(self, env, tmp_path, cr_dataset):
        from radiologyai.interop.ingest import IngestRejectedError

        z = write_zip(tmp_path / "x.zip", {"a.dcm": dcm(cr_dataset)}, symlink="link.dcm")
        with pytest.raises(IngestRejectedError, match="link"):
            run(env, [("x.zip", z)])

    def test_nested_zip_refused(self, env, tmp_path, cr_dataset):
        from radiologyai.interop.ingest import IngestRejectedError

        inner = write_zip(tmp_path / "in.zip", {"a.dcm": dcm(cr_dataset)}).read_bytes()
        z = write_zip(tmp_path / "x.zip", {"inner.zip": inner})
        with pytest.raises(IngestRejectedError, match="aninhado"):
            run(env, [("x.zip", z)])

    def test_uncompressed_bytes_counted_while_reading(self, env, tmp_path):
        """Bomba de compressão: o limite vale sobre os bytes LIDOS, não sobre o cabeçalho."""
        from radiologyai.interop.ingest import UploadTooLargeError

        z = write_zip(tmp_path / "bomba.zip", {"zeros.dcm": b"\0" * 4_000_000})
        assert z.stat().st_size < 100_000
        with pytest.raises(UploadTooLargeError):
            run(env, [("bomba.zip", z)])

    def test_member_count_limit(self, env, tmp_path):
        from radiologyai.interop.ingest import UploadTooLargeError

        z = write_zip(tmp_path / "muitos.zip", {f"{i}.dcm": b"x" for i in range(11)})
        with pytest.raises(UploadTooLargeError, match="10 arquivos"):
            run(env, [("muitos.zip", z)])

    def test_zip_of_series_ingested(self, env, tmp_path, cr_dataset):
        z = write_zip(
            tmp_path / "ok.zip", {"pasta/a.dcm": dcm(cr_dataset), "__MACOSX/._a.dcm": b"x"}
        )
        result = run(env, [("ok.zip", z)])
        assert result.n_images == 1
        assert result.exam_type == "cxr"


@pytest.mark.requirement("REQ-152")
class TestFormats:
    def test_unknown_format_refused_not_ignored(self, env, tmp_path):
        from radiologyai.interop.ingest import IngestRejectedError

        p = tmp_path / "notas.txt"
        p.write_text("nada de imagem aqui")
        with pytest.raises(IngestRejectedError, match="não reconhecido"):
            run(env, [("notas.txt", p)])

    def test_two_studies_in_one_upload_refused(self, env, tmp_path, cr_dataset):
        from radiologyai.interop.ingest import IngestRejectedError

        a, b = tmp_path / "a.dcm", tmp_path / "b.dcm"
        a.write_bytes(dcm(cr_dataset))
        cr_dataset.StudyInstanceUID = generate_uid()
        cr_dataset.SOPInstanceUID = generate_uid()
        cr_dataset.file_meta.MediaStorageSOPInstanceUID = cr_dataset.SOPInstanceUID
        b.write_bytes(dcm(cr_dataset))
        with pytest.raises(IngestRejectedError, match="2 estudos"):
            run(env, [("a.dcm", a), ("b.dcm", b)])

    def test_dicom_and_png_not_mixed(self, env, tmp_path, cr_dataset):
        import numpy as np
        from PIL import Image

        from radiologyai.interop.ingest import IngestRejectedError

        a = tmp_path / "a.dcm"
        a.write_bytes(dcm(cr_dataset))
        b = tmp_path / "b.png"
        Image.fromarray(np.zeros((8, 8), np.uint8)).save(b)
        with pytest.raises(IngestRejectedError, match="misture"):
            run(env, [("a.dcm", a), ("b.png", b)])

    def test_nifti_requires_declared_modality(self, env, tmp_path):
        nib = pytest.importorskip("nibabel")
        import numpy as np

        from radiologyai.interop.ingest import IngestRejectedError

        img = nib.Nifti1Image(np.zeros((8, 8, 4), np.int16), np.eye(4))
        img.header["descrip"] = b"PACIENTE FULANO"
        p = tmp_path / "vol.nii.gz"
        nib.save(img, str(p))
        with pytest.raises(IngestRejectedError, match="modality"):
            run(env, [("vol.nii.gz", p)])

    def test_nifti_free_text_header_erased(self, env, tmp_path):
        nib = pytest.importorskip("nibabel")
        import numpy as np

        from radiologyai.interop.ingest import HumanMeta

        img = nib.Nifti1Image(np.arange(8 * 8 * 4, dtype=np.int16).reshape(8, 8, 4), np.eye(4))
        img.header["descrip"] = b"PACIENTE FULANO"
        p = tmp_path / "vol.nii.gz"
        nib.save(img, str(p))
        result = run(env, [("vol.nii.gz", p)], human=HumanMeta(modality="CT"))
        assert result.n_images == 1
        _, blobs, _ = env
        for n, f in enumerate(blobs.store.base_dir.rglob("*.gz")):
            copy = tmp_path / f"lido{n}.nii.gz"
            copy.write_bytes(f.read_bytes())
            loaded = nib.load(str(copy))
            assert b"FULANO" not in bytes(loaded.header["descrip"])
        assert result.exam_type == "unknown"  # NIfTI sem descrição: um humano escolhe
