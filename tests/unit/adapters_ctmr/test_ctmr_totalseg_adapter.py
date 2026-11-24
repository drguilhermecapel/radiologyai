"""Adaptador TotalSegmentator com um executável FALSO no lugar do processo filho.

O falso (escrito em ``tmp_path``, nunca distribuído) lê o NIfTI de entrada,
"segmenta" por limiar de HU e grava o mapa + JSON lateral como o filho real.
Assim se testa tudo o que é do adaptador: árvore de pesos, ambiente sem rede,
conferência de versão/grade, volumes, portão a montante e o pipeline.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from ctmr_support import ct_meta, volume_image

from radiologyai.analysis.adapters import ct_totalseg
from radiologyai.analysis.adapters.ct_totalseg import (
    DEAD_PROXY,
    TotalSegmentatorAdapter,
    TotalSegmentatorRunError,
)
from radiologyai.analysis.artifacts import ArtifactStore
from radiologyai.analysis.base import AdapterOutput, LabelMap, ModelInputs, ResolvedWeights
from radiologyai.analysis.manifest import load_manifest
from radiologyai.analysis.pipeline import analyze
from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR, Registry
from radiologyai.analysis.types import AbstentionReason, ExamInputs
from radiologyai.catalog.catalog import Catalog
from radiologyai.catalog.schema import ExamType
from radiologyai.config.settings import Settings

pytest.importorskip("nibabel")

FAKE_RUNNER = r"""
import argparse, json, os, sys
from pathlib import Path
import nibabel as nib, numpy as np
p = argparse.ArgumentParser()
for name in ("--input", "--output", "--sidecar", "--task", "--mode", "--device"):
    p.add_argument(name)
p.add_argument("--expect", action="append", default=[])
a = p.parse_args()
weights = Path(os.environ["TOTALSEG_WEIGHTS_PATH"])
tree = sorted(str(q.relative_to(weights)) for q in weights.rglob("*") if q.is_file())
dump = {k: os.environ.get(k) for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "ALL_PROXY",
        "NO_PROXY", "TOTALSEG_HOME_DIR", "TOTALSEG_WEIGHTS_PATH")}
dump["tree"] = tree
dump["expect_ok"] = all((weights / e).is_dir() for e in a.expect)
dump["argv"] = sys.argv[1:]
Path(os.environ["FAKE_ENV_OUT"]).write_text(json.dumps(dump))
mode = os.environ.get("FAKE_MODE", "ok")
if mode == "fail":
    print("falha simulada do filho", file=sys.stderr)
    sys.exit(3)
img = nib.load(a.input)
data = np.asarray(img.dataobj)
if a.task == "total":
    seg = ((data > -100) & (data < 100)).astype(np.uint8)
    cmap = {"1": "brain", "2": "skull"}
else:
    seg = (data > 500).astype(np.uint8)
    cmap = {"1": "intracerebral_hemorrhage"}
affine = img.affine.copy()
if mode == "badshape":
    seg = seg[:-1]
if mode == "badaffine":
    affine[0, 3] += 5.0
nib.save(nib.Nifti1Image(seg, affine), a.output)
Path(a.sidecar).write_text(json.dumps({"task": a.task, "class_map": cmap,
    "totalsegmentator_version": os.environ.get("FAKE_VERSION", "2.18.0")}))
"""

SP = (5.0, 1.0, 1.0)
BLEED_VOXELS = 4 * 10 * 10


@pytest.fixture
def fake_runner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    script = tmp_path / "fake_totalseg.py"
    script.write_text(FAKE_RUNNER, encoding="utf-8")
    env_out = tmp_path / "env.json"
    monkeypatch.setenv("FAKE_ENV_OUT", str(env_out))
    monkeypatch.setattr(TotalSegmentatorAdapter, "command", (sys.executable, str(script)))
    return env_out


def _head_ct() -> np.ndarray:
    vol = np.full((20, 80, 80), -1000.0, dtype=np.float32)
    vol[2:18, 5:75, 5:75] = 30.0  # "encéfalo" (392 mL)
    vol[8:12, 10:20, 10:20] = 700.0  # "sangue" de volume conhecido
    return vol


def _resolved(manifest, tmp_path: Path) -> ResolvedWeights:
    files, hashes = {}, {}
    for w in manifest.weights:
        f = tmp_path / "w" / w.filename
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(w.key.encode())
        files[w.key] = f
        hashes[w.key] = hashlib.sha256(w.key.encode()).hexdigest()
    return ResolvedWeights(files=files, sha256=hashes)


def _adapter(model_id: str, tmp_path: Path) -> TotalSegmentatorAdapter:
    manifest, _ = load_manifest(PACKAGE_MANIFEST_DIR / f"{model_id}.yaml")
    adapter = TotalSegmentatorAdapter(manifest)
    adapter.load(_resolved(manifest, tmp_path), "cpu")
    return adapter


def _upstream(image, brain=True) -> dict[str, AdapterOutput]:
    arr = np.zeros(image.pixels.shape, dtype=np.uint8)
    if brain:
        arr[2:18, 5:75, 5:75] = 1
    return {
        "totalseg-total-3mm": AdapterOutput(
            labelmaps=[LabelMap(name="total", array=arr, labels={1: "brain"})]
        )
    }


@pytest.mark.requirement("REQ-130")
class TestOfflineSubprocess:
    def test_child_env_is_offline_and_tree_is_verified_files(self, fake_runner, tmp_path):
        adapter = _adapter("totalseg-cerebral-bleed", tmp_path)
        image = volume_image(_head_ct(), SP)
        adapter.predict(
            ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD, upstream=_upstream(image))
        )
        env = json.loads(fake_runner.read_text())
        assert env["HTTPS_PROXY"] == env["HTTP_PROXY"] == env["https_proxy"] == DEAD_PROXY
        assert env["ALL_PROXY"] == DEAD_PROXY
        assert env["NO_PROXY"] == ""
        assert env["expect_ok"] is True
        assert (
            "Dataset150_icb_v0/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/"
            "checkpoint_final.pth" in env["tree"]
        )
        assert "--expect" in env["argv"]
        adapter.unload()
        assert not Path(env["TOTALSEG_WEIGHTS_PATH"]).exists()

    def test_runner_forces_usage_stats_off(self, tmp_path):
        from radiologyai.analysis.adapters.ct_totalseg_runner import force_no_usage_stats

        home = tmp_path / "home"
        home.mkdir()
        (home / "config.json").write_text(json.dumps({"send_usage_stats": True, "x": 1}))
        force_no_usage_stats(home)
        data = json.loads((home / "config.json").read_text())
        assert data["send_usage_stats"] is False
        assert data["x"] == 1
        force_no_usage_stats(tmp_path / "new")
        assert (
            json.loads((tmp_path / "new" / "config.json").read_text())["send_usage_stats"] is False
        )

    def test_runner_refuses_missing_weights_before_import(self, tmp_path, monkeypatch):
        from radiologyai.analysis.adapters import ct_totalseg_runner

        monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path / "h"))
        monkeypatch.setenv("TOTALSEG_WEIGHTS_PATH", str(tmp_path / "w"))
        with pytest.raises(SystemExit, match="download é proibido"):
            ct_totalseg_runner.main(
                [
                    "--input",
                    "x",
                    "--output",
                    "y",
                    "--sidecar",
                    "z",
                    "--task",
                    "total",
                    "--expect",
                    "Dataset297_TotalSegmentator_total_3mm_1559subj",
                ]
            )


@pytest.mark.requirement("REQ-131")
@pytest.mark.usefixtures("fake_runner")
class TestOutputChecks:
    def test_volume_in_ml_from_source_grid(self, tmp_path):
        adapter = _adapter("totalseg-cerebral-bleed", tmp_path)
        image = volume_image(_head_ct(), SP)
        out = adapter.predict(
            ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD, upstream=_upstream(image))
        )
        (lm,) = out.labelmaps
        assert lm.array.shape == image.pixels.shape
        assert lm.labels == {1: "intracerebral_hemorrhage"}
        (vol,) = out.measurements
        assert vol.key == "intracerebral_hemorrhage_volume_ml"
        assert vol.value == pytest.approx(BLEED_VOXELS * 5.0 / 1000.0)
        assert "1 componente" in (vol.note or "")

    @pytest.mark.parametrize("mode", ["badshape", "badaffine", "fail"])
    def test_bad_child_output_raises(self, tmp_path, monkeypatch, mode):
        monkeypatch.setenv("FAKE_MODE", mode)
        adapter = _adapter("totalseg-cerebral-bleed", tmp_path)
        image = volume_image(_head_ct(), SP)
        with pytest.raises(TotalSegmentatorRunError):
            adapter.predict(
                ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD, upstream=_upstream(image))
            )

    def test_version_mismatch_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("FAKE_VERSION", "2.99.0")
        adapter = _adapter("totalseg-cerebral-bleed", tmp_path)
        image = volume_image(_head_ct(), SP)
        with pytest.raises(TotalSegmentatorRunError, match="2.99.0"):
            adapter.predict(
                ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD, upstream=_upstream(image))
            )

    def test_nifti_roundtrip_keeps_geometry(self, tmp_path):
        import nibabel as nib

        image = volume_image(_head_ct(), SP)
        ras = ct_totalseg.write_nifti(image, tmp_path / "x.nii.gz")
        img = nib.load(str(tmp_path / "x.nii.gz"))
        assert np.asarray(img.dataobj).dtype == np.int16
        assert img.shape == image.pixels.shape[::-1]
        assert np.allclose(img.affine, ras)
        assert np.allclose([float(z) for z in img.header.get_zooms()], SP[::-1])


@pytest.mark.requirement("REQ-132")
@pytest.mark.usefixtures("fake_runner")
class TestGateAndDerivedFinding:
    def test_target_organ_absent_abstains(self, tmp_path):
        adapter = _adapter("totalseg-cerebral-bleed", tmp_path)
        image = volume_image(_head_ct(), SP)
        ok = adapter.applicability(
            ModelInputs(images=(image,), exam_type=ExamType.CT_HEAD, upstream=_upstream(image))
        )
        assert ok.applicable, ok.reasons
        qa = adapter.applicability(
            ModelInputs(
                images=(image,), exam_type=ExamType.CT_HEAD, upstream=_upstream(image, brain=False)
            )
        )
        assert not qa.applicable
        assert AbstentionReason.BODY_PART in qa.reasons

    def test_wrong_exam_type_abstains(self, tmp_path):
        adapter = _adapter("totalseg-cerebral-bleed", tmp_path)
        image = volume_image(_head_ct(), SP)
        qa = adapter.applicability(
            ModelInputs(images=(image,), exam_type=ExamType.CT_CHEST, upstream=_upstream(image))
        )
        assert AbstentionReason.EXAM_TYPE in qa.reasons

    def test_missing_geometry_abstains(self, tmp_path):
        from radiologyai.analysis.types import ImageData, ImageKind, Modality

        adapter = _adapter("totalseg-cerebral-bleed", tmp_path)
        image = ImageData(
            pixels=_head_ct(),
            kind=ImageKind.VOLUME,
            modality=Modality.CT,
            meta=ct_meta(),
            value_range=(-1024.0, 3071.0),
        )
        qa = adapter.applicability(
            ModelInputs(
                images=(image,),
                exam_type=ExamType.CT_HEAD,
                upstream=_upstream(volume_image(_head_ct(), SP)),
            )
        )
        assert AbstentionReason.GEOMETRY in qa.reasons

    def test_pipeline_end_to_end(self, tmp_path):
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        weights_dir = tmp_path / "weights"
        for mid in ("totalseg-total-3mm", "totalseg-cerebral-bleed"):
            data = yaml.safe_load((PACKAGE_MANIFEST_DIR / f"{mid}.yaml").read_text())
            for w in data["weights"]:
                body = f"{mid}:{w['key']}".encode()
                target = weights_dir / mid / w["filename"]
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(body)
                w["sha256"], w["bytes"] = hashlib.sha256(body).hexdigest(), len(body)
            (manifest_dir / f"{mid}.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
        settings = Settings(weights_dir=weights_dir, device="cpu", max_loaded_models=4)
        registry = Registry(manifest_dir=manifest_dir, weights_dir=weights_dir, settings=settings)
        exam = ExamInputs(images=(volume_image(_head_ct(), SP),), exam_type=ExamType.CT_HEAD)
        result = analyze(
            exam,
            registry=registry,
            catalog=Catalog([], None),
            store=ArtifactStore(tmp_path / "store"),
        )
        registry.unload_all()
        runs = {r.model_id: r.status for r in result.runs}
        assert runs == {"totalseg-total-3mm": "completed", "totalseg-cerebral-bleed": "completed"}
        (finding,) = result.findings
        assert finding.condition_id == "ct.intracranial_hemorrhage"
        assert finding.decision == "positive"
        assert finding.operating_points[0].name == "any_volume_ge_0_1ml"
        assert finding.visibility == "research"  # fora do catálogo => nível D
        assert {s.name for s in result.segmentations} == {"total", "cerebral_bleed"}
