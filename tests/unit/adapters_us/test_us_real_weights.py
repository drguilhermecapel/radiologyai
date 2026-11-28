"""Execução REAL dos modelos de US/eco (pulada sem pesos/dados locais).

Rodam os pesos verdadeiros pelo registro (sha256 conferido) sobre dados públicos
do cache local. As tolerâncias abaixo são verificações de sanidade de execução
(o modelo produziu um contorno plausível), não medidas de desempenho — essas
vêm da fábrica de evidência sobre as predições gravadas pelos scripts.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from us_support import DATASETS, WEIGHTS, have

from radiologyai.analysis.types import ImageKind, Primitive
from radiologyai.catalog.schema import ExamType

MANIFESTS = Path(__file__).resolve().parents[3] / "src/radiologyai/analysis/manifests"
CAMUS = DATASETS / "camus-sample" / "patient0001"
BUSBRA = DATASETS / "bus-bra/extracted/BUSBRA"

pytestmark = pytest.mark.timeout(1800)


def _registry(tmp_path: Path, *ids: str) -> object:
    from radiologyai.analysis.registry import Registry
    from radiologyai.config.settings import Settings

    mdir = tmp_path / "manifests"
    mdir.mkdir()
    for i in ids:
        shutil.copy(MANIFESTS / f"{i}.yaml", mdir / f"{i}.yaml")
    settings = Settings(weights_dir=WEIGHTS, device="cpu", allow_noncommercial_models=True)
    return Registry(manifest_dir=mdir, weights_dir=WEIGHTS, settings=settings)


def _box(mask: np.ndarray, frame: int, label: str, margin: int = 5) -> Primitive:
    ys, xs = np.nonzero(mask)
    return Primitive(
        type="bbox",
        xy=(
            (float(xs.min() - margin), float(ys.min() - margin)),
            (float(xs.max() + margin), float(ys.max() + margin)),
        ),
        frame=frame,
        label=label,
    )


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / max(int(a.sum() + b.sum()), 1))


@pytest.mark.requirement("REQ-146")
@pytest.mark.skipif(
    not (
        have("torch", "nibabel", "scipy")
        and (WEIGHTS / "medsam2-echo-lv/MedSAM2_US_Heart.pt").exists()
        and CAMUS.is_dir()
    ),
    reason="pesos MedSAM2 ou amostra CAMUS ausentes",
)
def test_medsam2_echo_real_camus_patient(tmp_path: Path) -> None:
    from radiologyai.analysis.artifacts import ArtifactStore
    from radiologyai.analysis.imaging import load_cine
    from radiologyai.analysis.pipeline import analyze
    from radiologyai.analysis.types import ExamInputs
    from radiologyai.catalog.catalog import Catalog

    images, prompts, gts = [], [], {}
    for view, role in (("4CH", "A4C"), ("2CH", "A2C")):
        clip = load_cine(
            CAMUS / f"patient0001_{view}_half_sequence.nii.gz",
            view_position=role,
            patient_age_years=56.0,
        )
        gt = load_cine(CAMUS / f"patient0001_{view}_half_sequence_gt.nii.gz")
        assert clip.kind is ImageKind.CINE
        assert clip.spacing_mm is not None
        lv = gt.pixels == 1
        last = int(clip.pixels.shape[0]) - 1  # CAMUS: ED no 1º quadro, ES no último
        prompts += [
            _box(lv[0], 0, f"{role.lower()}:ed"),
            _box(lv[last], last, f"{role.lower()}:es"),
        ]
        gts[role.lower()] = (lv[0], lv[last])
        images.append(clip)
    registry = _registry(tmp_path, "medsam2-echo-lv")
    result = analyze(
        ExamInputs(images=tuple(images), exam_type=ExamType.ECHO, prompts=tuple(prompts)),
        registry=registry,
        catalog=Catalog([], None),
        store=ArtifactStore(tmp_path / "store"),
        model_ids=["medsam2-echo-lv"],
    )
    run = result.runs[0]
    assert run.status == "completed", run.error or run.qa
    meas = {m.name: m for m in result.measurements}
    ef = meas["lv_ef"].value
    assert ef is not None
    # Info_2CH.cfg de patient0001: EF 54 — sanidade de execução (contorno plausível)
    assert abs(ef - 54.0) < 12.0
    assert meas["lv_edv"].unit == "mL"
    assert meas["lv_edv"].value > meas["lv_esv"].value
    assert len(result.segmentations) == 2
    assert result.provenance[0].weights_sha256["checkpoint"].startswith("295c0ff8")


@pytest.mark.requirement("REQ-147")
@pytest.mark.skipif(
    not (
        have("torch", "transformers", "PIL", "scipy")
        and (WEIGHTS / "medsam-vit-base/pytorch_model.bin").exists()
        and BUSBRA.is_dir()
    ),
    reason="pesos MedSAM ou BUS-BRA ausentes",
)
def test_medsam_real_busbra_lesion(tmp_path: Path) -> None:
    from PIL import Image

    from radiologyai.analysis.base import ModelInputs
    from radiologyai.analysis.imaging import load_raster
    from radiologyai.analysis.types import Modality

    registry = _registry(tmp_path, "medsam-vit-base")
    adapter = registry.load("medsam-vit-base")
    image = load_raster(
        BUSBRA / "Images/bus_0001-l.png", modality=Modality.US, spacing_mm=(0.1, 0.1)
    )
    gt = np.asarray(Image.open(BUSBRA / "Masks/mask_0001-l.png").convert("L")) > 127
    inputs = ModelInputs(
        images=(image,), exam_type=ExamType.US_BREAST, prompts=(_box(gt, 0, "lesion"),)
    )
    assert adapter.applicability(inputs).applicable
    out = adapter.predict(inputs)
    pred = out.labelmaps[0].array == 1
    assert _dice(pred, gt) > 0.7
    meas = {m.key: m for m in out.measurements}
    assert meas["lesion1_area"].value is not None
    assert meas["lesion1_area"].value > 0
    out2 = adapter.predict(inputs)
    np.testing.assert_array_equal(out2.labelmaps[0].array, out.labelmaps[0].array)


PROBES = (
    "us-probe-breast-malignancy",
    "us-probe-liver-steatosis",
    "us-probe-liver-lesion-malignancy",
)


@pytest.mark.requirement("REQ-148")
@pytest.mark.parametrize("probe_id", PROBES)
def test_us_probe_real_weights_deterministic(probe_id: str, tmp_path: Path) -> None:
    if (
        not (MANIFESTS / f"{probe_id}.yaml").exists()
        or not (WEIGHTS / probe_id / "probe.safetensors").exists()
    ):
        pytest.skip("sonda não ajustada neste ambiente (rode scripts/probe_fit_us_probes.py)")
    if not (have("torch", "open_clip", "timm", "PIL") and BUSBRA.is_dir()):
        pytest.skip("extras ou dados ausentes")
    from PIL import Image

    from radiologyai.analysis.base import ModelInputs
    from radiologyai.analysis.imaging import load_raster
    from radiologyai.analysis.types import Modality

    registry = _registry(tmp_path, probe_id)
    adapter = registry.load(probe_id)
    image = load_raster(BUSBRA / "Images/bus_0001-l.png", modality=Modality.US)
    gt = np.asarray(Image.open(BUSBRA / "Masks/mask_0001-l.png").convert("L")) > 127
    prompts = (_box(gt, 0, "lesion", margin=0),) if adapter.roi == "box" else ()
    exam = ExamType.US_BREAST if "breast" in probe_id else ExamType.US_ABDOMEN
    inputs = ModelInputs(images=(image,), exam_type=exam, prompts=prompts)
    assert adapter.applicability(inputs).applicable
    a = adapter.predict(inputs).scores[0]
    b = adapter.predict(inputs).scores[0]
    assert a.kind == "logit"
    assert np.isfinite(a.value)
    assert a.value == b.value
