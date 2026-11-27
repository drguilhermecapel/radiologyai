"""Caminhos de erro e bordas do adaptador TotalSegmentator (REQ-130, REQ-131, REQ-136, REQ-138).

O processo filho é um script FALSO escrito em ``tmp_path`` (nunca distribuído)
com modos que simulam falhas do filho real: saída ausente, tarefa divergente,
rótulos fora de faixa, demora. Os mapas que ele grava são geométricos (limiar
de HU), não predições.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ctmr_support import ct_meta, volume_image

from radiologyai.analysis.adapters import ct_totalseg
from radiologyai.analysis.adapters.ct_totalseg import (
    TotalSegmentatorAdapter,
    TotalSegmentatorRunError,
    _device_arg,
    write_nifti,
)
from radiologyai.analysis.base import AdapterOutput, LabelMap, ModelInputs, ResolvedWeights
from radiologyai.analysis.manifest import ModelManifest, load_manifest
from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR
from radiologyai.analysis.types import AbstentionReason, Modality
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import BackendUnavailableError, ManifestError

nib = pytest.importorskip("nibabel")

FAKE = r"""
import argparse, json, os, sys, time
from pathlib import Path
import nibabel as nib, numpy as np
p = argparse.ArgumentParser()
for name in ("--input", "--output", "--sidecar", "--task", "--mode", "--device"):
    p.add_argument(name)
p.add_argument("--expect", action="append", default=[])
a = p.parse_args()
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "sleep":
    time.sleep(30)
if mode == "nooutput":
    sys.exit(0)
img = nib.load(a.input)
data = np.asarray(img.dataobj)
cmap = json.loads(os.environ["FAKE_CLASSMAP"])
seg = np.zeros(data.shape, dtype=np.int16)
for value, label in json.loads(os.environ.get("FAKE_THRESHOLDS", "[]")):
    seg[np.isclose(data, value)] = label
if mode == "badlabels":
    seg[0, 0, 0] = 300
nib.save(nib.Nifti1Image(seg, img.affine), a.output)
task = "outra" if mode == "wrongtask" else a.task
Path(a.sidecar).write_text(json.dumps({"task": task, "class_map": cmap,
    "totalsegmentator_version": "2.18.0"}))
"""

SP = (5.0, 1.0, 1.0)


def _manifest(model_id: str) -> ModelManifest:
    manifest, _ = load_manifest(PACKAGE_MANIFEST_DIR / f"{model_id}.yaml")
    return manifest


def _resolved(manifest: ModelManifest, tmp_path: Path, drop: str | None = None) -> ResolvedWeights:
    files, hashes = {}, {}
    for w in manifest.weights:
        if w.key == drop:
            continue
        f = tmp_path / "w" / w.filename
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(w.key.encode())
        files[w.key] = f
        hashes[w.key] = hashlib.sha256(w.key.encode()).hexdigest()
    return ResolvedWeights(files=files, sha256=hashes)


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    script = tmp_path / "fake_child.py"
    script.write_text(FAKE, encoding="utf-8")
    monkeypatch.setattr(TotalSegmentatorAdapter, "command", (sys.executable, str(script)))
    monkeypatch.setenv("FAKE_CLASSMAP", '{"1": "intracerebral_hemorrhage"}')
    return monkeypatch


def _loaded(model_id: str, tmp_path: Path) -> TotalSegmentatorAdapter:
    adapter = TotalSegmentatorAdapter(_manifest(model_id))
    adapter.load(_resolved(adapter.manifest, tmp_path), "cpu")
    return adapter


def _head() -> np.ndarray:
    vol = np.full((20, 60, 60), -1000.0, dtype=np.float32)
    vol[2:18, 5:55, 5:55] = 30.0
    return vol


@pytest.mark.requirement("REQ-131")
def test_ctmr_device_arg_and_nifti_writer(tmp_path: Path) -> None:
    assert _device_arg("cuda:1") == "gpu:1"
    assert _device_arg("cuda") == "gpu"
    assert _device_arg("cpu") == "cpu"
    ct = volume_image(_head(), SP)
    ras = write_nifti(ct, tmp_path / "ct.nii.gz")
    assert np.asarray(nib.load(str(tmp_path / "ct.nii.gz")).dataobj).dtype == np.int16
    assert ras[0, 0] == -1.0 * ct.affine_lps[0, 0]  # type: ignore[index]
    mr = volume_image(_head() + 0.25, SP, modality=Modality.MR)
    write_nifti(mr, tmp_path / "mr.nii.gz")
    assert np.asarray(nib.load(str(tmp_path / "mr.nii.gz")).dataobj).dtype == np.float32
    flat = type(ct)(
        pixels=ct.pixels,
        kind=ct.kind,
        modality=ct.modality,
        meta=ct.meta,
        spacing_mm=ct.spacing_mm,
        value_range=ct.value_range,
    )
    with pytest.raises(TotalSegmentatorRunError, match="afim"):
        write_nifti(flat, tmp_path / "x.nii.gz")


@pytest.mark.requirement("REQ-130")
def test_ctmr_load_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest("totalseg-cerebral-bleed")
    with pytest.raises(ManifestError, match="config do TotalSegmentator inválida"):
        TotalSegmentatorAdapter(manifest.model_copy(update={"config": {"task": "x"}}))
    adapter = TotalSegmentatorAdapter(manifest)
    first_key = manifest.weights[0].key
    with pytest.raises(ManifestError, match="não declarado"):
        adapter.load(_resolved(manifest, tmp_path, drop=first_key), "cpu")
    assert adapter._command()[1:] == ["-m", "radiologyai.analysis.adapters.ct_totalseg_runner"]
    with pytest.raises(TotalSegmentatorRunError, match="não carregado"):
        adapter.run_task(volume_image(_head(), SP))

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a: None if n == "totalsegmentator" else real(n, *a)
    )
    with pytest.raises(BackendUnavailableError, match="TotalSegmentator"):
        adapter.load(_resolved(manifest, tmp_path), "cpu")
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a: None if n == "nibabel" else real(n, *a)
    )
    with pytest.raises(BackendUnavailableError, match="nibabel"):
        adapter.load(_resolved(manifest, tmp_path), "cpu")


@pytest.mark.requirement("REQ-130")
@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("nooutput", "saída ausente"),
        ("wrongtask", "tarefa divergente"),
        ("badlabels", "fora de 0..255"),
    ],
)
def test_ctmr_child_failures_raise(
    fake: pytest.MonkeyPatch, tmp_path: Path, mode: str, match: str
) -> None:
    fake.setenv("FAKE_MODE", mode)
    adapter = _loaded("totalseg-cerebral-bleed", tmp_path)
    try:
        with pytest.raises(TotalSegmentatorRunError, match=match):
            adapter.run_task(volume_image(_head(), SP))
    finally:
        adapter.unload()


@pytest.mark.requirement("REQ-130")
def test_ctmr_child_timeout_is_an_error(fake: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake.setenv("FAKE_MODE", "sleep")
    adapter = TotalSegmentatorAdapter(_manifest("totalseg-cerebral-bleed"))
    adapter.config = adapter.config.model_copy(update={"timeout_s": 1})
    adapter.load(_resolved(adapter.manifest, tmp_path), "cpu")
    try:
        with pytest.raises(TotalSegmentatorRunError, match="excedeu 1 s"):
            adapter.run_task(volume_image(_head(), SP))
    finally:
        adapter.unload()


@pytest.mark.requirement("REQ-131")
def test_ctmr_volume_spec_citing_unknown_label_is_manifest_error(
    fake: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake.setenv("FAKE_CLASSMAP", '{"1": "outra_coisa"}')
    adapter = _loaded("totalseg-cerebral-bleed", tmp_path)
    image = volume_image(_head(), SP)
    try:
        with pytest.raises(ManifestError, match="cita"):
            adapter.predict(
                ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD, upstream=_brain(image))
            )
    finally:
        adapter.unload()


def _brain(image: Any, present: bool = True) -> dict[str, AdapterOutput]:
    arr = np.zeros(image.pixels.shape, dtype=np.uint8)
    if present:
        arr[2:18, 5:55, 5:55] = 1
    return {
        "totalseg-total-3mm": AdapterOutput(
            labelmaps=[LabelMap(name="total", array=arr, labels={1: "brain"})]
        )
    }


VENTRICLE_MAP = {
    "1": "ventricle_frontal_horn_left",
    "2": "ventricle_frontal_horn_right",
    "3": "third_ventricle",
}
VENTRICLE_NAMES = (
    "ventricle_occipital_horn_left",
    "ventricle_body_left",
    "ventricle_temporal_horn_left",
    "ventricle_trigone_left",
    "ventricle_occipital_horn_right",
    "ventricle_body_right",
    "ventricle_temporal_horn_right",
    "ventricle_trigone_right",
    "fourth_ventricle",
)


@pytest.mark.requirement("REQ-138")
def test_ctmr_ventricles_volume_and_evans_paths(fake: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import json

    cmap = dict(VENTRICLE_MAP)
    cmap.update({str(4 + i): n for i, n in enumerate(VENTRICLE_NAMES)})
    fake.setenv("FAKE_CLASSMAP", json.dumps(cmap))
    fake.setenv("FAKE_THRESHOLDS", json.dumps([[5.0, 1], [6.0, 2], [7.0, 3]]))
    vol = _head()
    vol[8:12, 20:28, 20:28] = 5.0  # corno frontal esquerdo
    vol[8:12, 20:28, 32:40] = 6.0  # corno frontal direito
    vol[8:12, 28:32, 28:32] = 7.0
    image = volume_image(vol, SP, image_meta=ct_meta(patient_age_years=70.0))
    adapter = _loaded("totalseg-ventricle-parts", tmp_path)
    try:
        with_brain = ModelInputs(
            images=(image,), exam_type=ExamType.CT_HEAD, upstream=_brain(image)
        )
        assert adapter.applicability(with_brain).applicable, adapter.applicability(with_brain)
        out = adapter.predict(with_brain)
        meas = {m.key: m for m in out.measurements}
        vent = meas["ventricular_system_volume_ml"]
        assert vent.value == pytest.approx((2 * 4 * 8 * 8 + 4 * 4 * 4) * 5.0 / 1000.0)
        assert vent.preconditions["n_components"] == "1"  # os três se tocam
        evans = [m for m in out.measurements if m.key != "ventricular_system_volume_ml"]
        assert len(evans) == 1 and (evans[0].value is not None or evans[0].abstention)

        no_brain = ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD)
        qa = adapter.applicability(no_brain)
        assert AbstentionReason.UPSTREAM in qa.reasons
        out2 = adapter.predict(no_brain)
        (ev2,) = [m for m in out2.measurements if m.key != "ventricular_system_volume_ml"]
        assert ev2.abstention is AbstentionReason.UPSTREAM

        small = ModelInputs(
            images=(image,), exam_type=ExamType.CT_HEAD, upstream=_brain(image, present=False)
        )
        assert AbstentionReason.BODY_PART in adapter.applicability(small).reasons

        other_grid = {
            "totalseg-total-3mm": AdapterOutput(
                labelmaps=[
                    LabelMap(name="t", array=np.zeros((2, 2, 2), np.uint8), labels={1: "brain"})
                ]
            )
        }
        with pytest.raises(TotalSegmentatorRunError, match="grade diferente"):
            adapter.predict(
                ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD, upstream=other_grid)
            )
    finally:
        adapter.unload()
    assert adapter._root is None


@pytest.mark.requirement("REQ-131")
def test_ctmr_applicability_needs_geometry_and_slices(tmp_path: Path) -> None:
    adapter = TotalSegmentatorAdapter(_manifest("totalseg-cerebral-bleed"))
    thin = volume_image(_head()[:3], SP)
    qa = adapter.applicability(ModelInputs(images=(thin,), exam_type=ExamType.CT_HEAD))
    assert AbstentionReason.INPUT in qa.reasons
    flat = type(thin)(
        pixels=_head(),
        kind=thin.kind,
        modality=thin.modality,
        meta=thin.meta,
        value_range=thin.value_range,
    )
    qa = adapter.applicability(ModelInputs(images=(flat,), exam_type=ExamType.CT_HEAD))
    assert AbstentionReason.GEOMETRY in qa.reasons
    assert ct_totalseg.DEAD_PROXY.startswith("http://127.0.0.1:")
