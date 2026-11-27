"""Adaptador de biomarcadores de TC e processo filho do TotalSegmentator sem pesos reais.

* :class:`CTBiomarkerAdapter`: o mapa de rótulos a montante é GEOMÉTRICO
  (cubos de HU conhecido), nunca saída de modelo.
* ``ct_totalseg_runner.main``: os módulos ``totalsegmentator``/``torch`` são
  substituídos por objetos de teste em ``sys.modules`` que gravam um mapa
  trivial; confere-se só o contrato do filho (telemetria desligada, pesos
  exigidos, tarefa comercial recusada, JSON lateral).
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ctmr_support import ct_meta, volume_image

from radiologyai.analysis.adapters.ct_biomarkers import CTBiomarkerAdapter
from radiologyai.analysis.adapters.ct_totalseg_runner import (
    USAGE_STATS_OFF,
    force_no_usage_stats,
    main,
)
from radiologyai.analysis.base import AdapterOutput, LabelMap, ModelInputs, ResolvedWeights
from radiologyai.analysis.manifest import ModelManifest
from radiologyai.analysis.registry import PACKAGE_MANIFEST_DIR
from radiologyai.analysis.types import AbstentionReason
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import BackendUnavailableError, ManifestError, UnsupportedInputError

SP = (2.0, 1.0, 1.0)
UP = "totalseg-total-3mm"


def _manifest() -> ModelManifest:
    from radiologyai.analysis.manifest import load_manifest

    manifest, _ = load_manifest(PACKAGE_MANIFEST_DIR / "ct-biomarkers.yaml")
    return manifest


def _abdomen() -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    hu = np.full((40, 90, 90), -1000.0, dtype=np.float32)
    lab = np.zeros(hu.shape, dtype=np.uint8)
    names = {1: "liver", 2: "spleen", 3: "heart"}
    hu[5:30, 10:50, 10:50] = 60.0
    lab[5:30, 10:50, 10:50] = 1
    hu[10:25, 55:75, 55:75] = 50.0
    lab[10:25, 55:75, 55:75] = 2
    hu[8:20, 20:35, 60:80] = 40.0
    lab[8:20, 20:35, 60:80] = 3
    return hu, lab, names


def _inputs(**meta: Any) -> ModelInputs:
    hu, lab, names = _abdomen()
    image = volume_image(hu, SP, image_meta=ct_meta(**meta))
    upstream = {UP: AdapterOutput(labelmaps=[LabelMap(name="total", array=lab, labels=names)])}
    return ModelInputs(images=(image,), exam_type=ExamType.CT_ABDOMEN, upstream=upstream)


@pytest.mark.requirement("REQ-133")
def test_ctmr_biomarker_adapter_manifest_contract() -> None:
    manifest = _manifest()
    with pytest.raises(ManifestError, match="anatomy_model"):
        CTBiomarkerAdapter(manifest.model_copy(update={"depends_on": ()}))
    bad = {**manifest.config, "biomarkers": ["aorta", "pressao_arterial"]}
    with pytest.raises(ManifestError, match="desconhecidos"):
        CTBiomarkerAdapter(manifest.model_copy(update={"config": bad}))
    only = {**manifest.config, "biomarkers": ["organ_volumes"]}
    assert CTBiomarkerAdapter(manifest.model_copy(update={"config": only})).enabled == (
        "organ_volumes",
    )


@pytest.mark.requirement("REQ-133")
def test_ctmr_biomarker_adapter_load_requires_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = CTBiomarkerAdapter(_manifest())
    adapter.load(ResolvedWeights(files={}, sha256={}), "cpu")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda n, *a: None if n == "scipy" else real(n, *a)
    )
    with pytest.raises(BackendUnavailableError, match="volumetric"):
        adapter.load(ResolvedWeights(files={}, sha256={}), "cpu")


@pytest.mark.requirement("REQ-133")
def test_ctmr_biomarker_adapter_applicability_gates() -> None:
    adapter = CTBiomarkerAdapter(_manifest())
    ok = _inputs()
    assert adapter.applicability(ok).applicable, adapter.applicability(ok)
    no_up = ModelInputs(images=ok.images, exam_type=ok.exam_type)
    assert AbstentionReason.UPSTREAM in adapter.applicability(no_up).reasons
    empty_up = ModelInputs(
        images=ok.images, exam_type=ok.exam_type, upstream={UP: AdapterOutput()}
    )
    assert AbstentionReason.UPSTREAM in adapter.applicability(empty_up).reasons
    image = ok.images[0]
    flat = type(image)(
        pixels=image.pixels,
        kind=image.kind,
        modality=image.modality,
        meta=image.meta,
        spacing_mm=image.spacing_mm,
        affine_lps=None,
        value_range=image.value_range,
    )
    no_geo = ModelInputs(images=(flat,), exam_type=ok.exam_type, upstream=ok.upstream)
    assert AbstentionReason.GEOMETRY in adapter.applicability(no_geo).reasons
    with pytest.raises(UnsupportedInputError, match="geometria"):
        adapter.predict(no_geo)
    with pytest.raises(UnsupportedInputError, match="mapa de rótulos"):
        adapter.predict(empty_up)


@pytest.mark.requirement("REQ-136")
def test_ctmr_biomarker_adapter_predict_runs_every_enabled_biomarker() -> None:
    adapter = CTBiomarkerAdapter(_manifest())
    out = adapter.predict(_inputs())
    meas = {m.key: m for m in out.measurements}
    assert {"spleen_volume_ml", "liver_volume_ml", "heart_volume_ml"} <= set(meas)
    # volumes de cubos conhecidos (voxel = 2 mm³ = 0.002 mL)
    assert meas["spleen_volume_ml"].value == pytest.approx(15 * 20 * 20 * 0.002)
    assert meas["liver_volume_ml"].value == pytest.approx(25 * 40 * 40 * 0.002)
    # sem vértebra L1, pulmões ou aorta no mapa: abstenção explícita, nunca zero
    for m in out.measurements:
        assert m.value is not None or m.abstention is not None, m
    assert len(out.measurements) == len({m.key for m in out.measurements})


# --------------------------------------------------------------------------- #
# Processo filho (REQ-130)
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-130")
def test_ctmr_runner_forces_usage_stats_off(tmp_path: Path) -> None:
    home = tmp_path / "home"
    force_no_usage_stats(home)
    assert json.loads((home / "config.json").read_text())["send_usage_stats"] is False
    (home / "config.json").write_text(json.dumps({"send_usage_stats": True, "x": 1}))
    force_no_usage_stats(home)
    data = json.loads((home / "config.json").read_text())
    assert data == {"send_usage_stats": False, "x": 1}
    assert USAGE_STATS_OFF["send_usage_stats"] is False


def _install_fake_backend(monkeypatch: pytest.MonkeyPatch, calls: list[dict[str, Any]]) -> None:
    import importlib.metadata

    nib = pytest.importorskip("nibabel")

    def fake_totalsegmentator(inp: str, out: str, **kw: Any) -> Any:
        calls.append({"input": inp, "output": out, **kw})
        img = nib.load(inp)
        seg = np.zeros(img.shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(seg, img.affine), out)
        return seg

    ts = types.ModuleType("totalsegmentator")
    api = types.ModuleType("totalsegmentator.python_api")
    api.totalsegmentator = fake_totalsegmentator  # type: ignore[attr-defined]
    m2b = types.ModuleType("totalsegmentator.map_to_binary")
    m2b.class_map = {"cerebral_bleed": {1: "intracerebral_hemorrhage"}}  # type: ignore[attr-defined]
    m2b.commercial_models = {"heartchambers_highres": 1}  # type: ignore[attr-defined]
    torch = types.ModuleType("torch")
    torch.__version__ = "0.0-teste"  # type: ignore[attr-defined]
    threads = {"n": 1}
    torch.set_num_threads = lambda n: threads.update(n=n)  # type: ignore[attr-defined]
    torch.get_num_threads = lambda: threads["n"]  # type: ignore[attr-defined]
    for name, mod in (
        ("totalsegmentator", ts),
        ("totalsegmentator.python_api", api),
        ("totalsegmentator.map_to_binary", m2b),
        ("torch", torch),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    real_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda d: "9.9.9" if d in ("TotalSegmentator", "nnunetv2") else real_version(d),
    )


@pytest.mark.requirement("REQ-130")
def test_ctmr_runner_main_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    nib = pytest.importorskip("nibabel")
    calls: list[dict[str, Any]] = []
    _install_fake_backend(monkeypatch, calls)
    weights = tmp_path / "weights"
    (weights / "Dataset150_bleed").mkdir(parents=True)
    monkeypatch.setenv("TOTALSEG_HOME_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("TOTALSEG_WEIGHTS_PATH", str(weights))
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    inp = tmp_path / "in.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((4, 5, 6), np.int16), np.eye(4)), str(inp))
    args = [
        "--input", str(inp),
        "--output", str(tmp_path / "seg.nii.gz"),
        "--sidecar", str(tmp_path / "side.json"),
        "--task", "cerebral_bleed",
        "--mode", "fast",
        "--expect", "Dataset150_bleed",
    ]  # fmt: skip
    assert main(args) == 0
    side = json.loads((tmp_path / "side.json").read_text())
    assert side["class_map"] == {"1": "intracerebral_hemorrhage"}
    assert side["torch_threads"] == 2 and side["totalsegmentator_version"] == "9.9.9"
    assert side["input_shape"] == [4, 5, 6] == side["output_shape"]
    assert calls[0]["fast"] is True and calls[0]["fastest"] is False and calls[0]["ml"] is True
    cfg = json.loads((tmp_path / "home" / "config.json").read_text())
    assert cfg["send_usage_stats"] is False

    with pytest.raises(SystemExit, match="download é proibido"):
        main([*args[:-1], "Dataset999_ausente"])
    commercial = [*args]
    commercial[commercial.index("cerebral_bleed")] = "heartchambers_highres"
    with pytest.raises(SystemExit, match="licença comercial"):
        main(commercial)
