"""Encanamento de sondas de US, MedSAM e utilidades de quadro sem pesos reais.

Sem torch: o codificador é um objeto Python que devolve um vetor FIXO derivado
da média do quadro (para checar ROI, agregação de cine e aplicação da sonda); a
sonda é montada com coeficientes conhecidos. Com torch: arquiteturas mínimas
(``transformers.SamModel`` com configuração reduzida e a torre ViT do
BiomedCLIP inicializada com semente) para exercitar ``load``/``embed``/
``segment``. Nenhum desses números tem significado clínico; nada sai de
``tests/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from us_support import have, sector_frame, us_image

from radiologyai.analysis.adapters.us_common import (
    box_prompts,
    clean_mask,
    frame_uint8,
    lesion_morphometry,
    mask_bbox,
    sector_bbox,
)
from radiologyai.analysis.adapters.us_probe import (
    BIOMEDCLIP_ENCODER_ID,
    USProbeAdapter,
    _frames_for,
    expand_box,
)
from radiologyai.analysis.base import ModelInputs, ResolvedWeights
from radiologyai.analysis.manifest import ModelManifest, load_manifest
from radiologyai.analysis.types import AbstentionReason, ImageKind, Primitive
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import BackendUnavailableError, ManifestError, UnsupportedInputError

MANIFESTS = Path(__file__).resolve().parents[3] / "src/radiologyai/analysis/manifests"
DIM = 512


def _manifest(model_id: str) -> ModelManifest:
    manifest, _ = load_manifest(MANIFESTS / f"{model_id}.yaml")
    return manifest


def _probe(label: str, encoder_id: str = BIOMEDCLIP_ENCODER_ID) -> Any:
    from radiologyai.analysis.probes.artifact import ProbeArtifact

    coef = np.zeros(DIM)
    coef[0] = 2.0
    return ProbeArtifact(
        label=label,
        encoder_id=encoder_id,
        embedding_dim=DIM,
        mean=np.zeros(DIM),
        scale=np.ones(DIM),
        coef=coef,
        intercept=-1.0,
        training_data_sha256="a" * 64,
    )


class _MeanEncoder:
    """Vetor fixo: componente 0 = média do quadro canônico / 255; resto zero."""

    def __init__(self) -> None:
        self.seen: list[tuple[int, ...]] = []

    def embed(self, canonical: np.ndarray) -> np.ndarray:
        arr = np.asarray(canonical, dtype=np.float64)
        if arr.ndim == 2:
            arr = arr[None]
        self.seen.append(arr.shape)
        out = np.zeros((arr.shape[0], DIM), dtype=np.float32)
        out[:, 0] = arr.reshape(arr.shape[0], -1).mean(axis=1) / 255.0
        return out


# --------------------------------------------------------------------------- #
# Utilidades de quadro (REQ-149)
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-149")
def test_us_frame_uint8_scales_cine_color_and_rejects_bad_frames() -> None:
    clip = us_image(
        np.stack([np.full((4, 5), v) for v in (0.0, 127.5, 255.0)]), kind=ImageKind.CINE
    )
    assert frame_uint8(clip, 2).max() == 255
    assert frame_uint8(clip, 1)[0, 0] == 128
    with pytest.raises(UnsupportedInputError, match="fora do clipe"):
        frame_uint8(clip, 3)
    still = us_image(np.full((4, 5), 10.0))
    with pytest.raises(UnsupportedInputError, match="quadro 0"):
        frame_uint8(still, 1)
    rgb = us_image(
        np.dstack([np.full((3, 3), 255.0), np.zeros((3, 3)), np.zeros((3, 3))]), color=True
    )
    assert int(frame_uint8(rgb)[0, 0]) == round(255 * 0.299)
    vol = us_image(np.zeros((2, 3, 3)), kind=ImageKind.VOLUME)
    with pytest.raises(UnsupportedInputError, match="volume"):
        frame_uint8(vol)


@pytest.mark.requirement("REQ-147")
def test_us_box_and_mask_helpers() -> None:
    prompts = (
        Primitive(type="bbox", xy=((1.0, 1.0), (5.0, 5.0)), label="Lesion:1"),
        Primitive(type="point", xy=((2.0, 2.0),)),
        Primitive(type="bbox", xy=((1.0, 1.0), (5.0, 5.0)), label="a4c:ed"),
    )
    assert len(box_prompts(prompts)) == 2
    assert [p.label for p in box_prompts(prompts, label_prefix="lesion")] == ["Lesion:1"]
    m = np.zeros((6, 7), bool)
    m[2:4, 1:6] = True
    assert mask_bbox(m).tolist() == [1.0, 2.0, 5.0, 3.0]
    with pytest.raises(UnsupportedInputError):
        mask_bbox(np.zeros((3, 3), bool))
    frame = sector_frame(60, 80).astype(np.uint8)
    x0, y0, x1, y1 = sector_bbox(frame)
    assert 0 < x0 < x1 < 80
    assert 0 < y0 < y1 < 60
    assert not clean_mask(np.zeros((4, 4), bool)).any()
    with pytest.raises(UnsupportedInputError, match="espaçamento inválido"):
        lesion_morphometry(m, (0.0, 1.0))


@pytest.mark.requirement("REQ-148")
def test_us_expand_box_and_frame_sampling() -> None:
    assert expand_box(np.array([10.0, 10.0, 30.0, 20.0]), (100, 100), 0.25) == (5, 7, 36, 24)
    assert expand_box(np.array([0.0, 0.0, 10.0, 10.0]), (8, 8), 0.5) == (0, 0, 8, 8)
    with pytest.raises(UnsupportedInputError, match="degenerada"):
        expand_box(np.array([5.0, 5.0, 6.0, 6.0]), (8, 8), 0.0)
    clip = us_image(np.zeros((25, 4, 4)), kind=ImageKind.CINE)
    assert _frames_for(clip, 5) == [0, 6, 12, 18, 24]
    assert _frames_for(clip, 1) == [0]
    assert _frames_for(us_image(np.zeros((4, 4))), 5) == [0]


# --------------------------------------------------------------------------- #
# Sondas (REQ-148)
# --------------------------------------------------------------------------- #
@pytest.mark.requirement("REQ-148")
def test_us_probe_manifest_contract_errors() -> None:
    manifest = _manifest("us-probe-breast-malignancy")
    with pytest.raises(ManifestError, match="exatamente um"):
        USProbeAdapter(manifest.model_copy(update={"labels": ()}))
    with pytest.raises(ManifestError, match="config.roi"):
        USProbeAdapter(manifest.model_copy(update={"config": {**manifest.config, "roi": "x"}}))
    no_prompt = manifest.model_copy(
        update={"inputs": manifest.inputs.model_copy(update={"prompts": "none"})}
    )
    with pytest.raises(ManifestError, match="prompts: required"):
        USProbeAdapter(no_prompt)
    with pytest.raises(ManifestError, match="pesos esperados"):
        USProbeAdapter(manifest.model_copy(update={"weights": manifest.weights[:2]}))


@pytest.mark.requirement("REQ-148")
@pytest.mark.skipif(not have("PIL"), reason="Pillow ausente")
def test_us_probe_box_roi_flow_with_fixed_encoder() -> None:
    adapter = USProbeAdapter(_manifest("us-probe-breast-malignancy"))
    img = us_image(np.pad(np.full((40, 60), 200.0), 30))
    with pytest.raises(RuntimeError, match="antes de load"):
        adapter.predict(ModelInputs(images=(img,), exam_type=ExamType.US_BREAST))
    none = ModelInputs(images=(img,), exam_type=ExamType.US_BREAST)
    qa = adapter.applicability(none)
    assert AbstentionReason.MISSING_INPUT in qa.reasons
    box = Primitive(type="bbox", xy=((30.0, 30.0), (89.0, 69.0)), label="lesion")
    two = ModelInputs(images=(img,), exam_type=ExamType.US_BREAST, prompts=(box, box))
    assert AbstentionReason.INPUT in adapter.applicability(two).reasons
    one = ModelInputs(images=(img,), exam_type=ExamType.US_BREAST, prompts=(box,))
    assert adapter.applicability(one).applicable

    enc = _MeanEncoder()
    adapter.encoder = enc  # type: ignore[assignment]
    adapter.probe = _probe("breast_lesion_malignant")
    out = adapter.predict(one)
    (score,) = out.scores
    assert score.label == "breast_lesion_malignant"
    assert score.kind == "logit"
    # ROI = caixa + 25 %: a média canônica fica entre o fundo (0) e a lesão (200)
    assert -1.0 < score.value < -1.0 + 2.0 * 200 / 255
    assert enc.seen == [(1, 224, 224)]
    assert "caixa do usuário (+25%)" in out.warnings[0]
    adapter.unload()
    assert adapter.encoder is None
    assert adapter.probe is None


@pytest.mark.requirement("REQ-148")
@pytest.mark.skipif(not have("PIL"), reason="Pillow ausente")
def test_us_probe_full_roi_averages_cine_frames() -> None:
    adapter = USProbeAdapter(_manifest("us-probe-liver-steatosis"))
    frames = np.stack([np.full((150, 150), 255.0 * k / 3) for k in range(4)])
    clip = us_image(frames, kind=ImageKind.CINE)
    inputs = ModelInputs(images=(clip,), exam_type=ExamType.US_ABDOMEN)
    assert adapter.applicability(inputs).applicable
    adapter.encoder = _MeanEncoder()  # type: ignore[assignment]
    adapter.probe = _probe("liver_steatosis")
    out = adapter.predict(inputs)
    assert out.scores[0].value == pytest.approx(-1.0 + 2.0 * 0.5, abs=0.01)
    assert "média de 4 quadro(s)" in out.warnings[0]


@pytest.mark.requirement("REQ-148")
@pytest.mark.skipif(not have("safetensors", "PIL"), reason="safetensors/Pillow ausentes")
def test_us_probe_load_checks_encoder_and_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from radiologyai.analysis.adapters import us_probe
    from radiologyai.analysis.probes.artifact import save_probe

    class _Enc(_MeanEncoder):
        def __init__(self, checkpoint: Any, device: str = "cpu") -> None:
            super().__init__()
            self.checkpoint = checkpoint
            self.device = device

    monkeypatch.setattr(us_probe, "BiomedCLIPImageEncoder", _Enc)

    def weights_for(probe: Any) -> ResolvedWeights:
        t, j = save_probe(probe, tmp_path / probe.label)
        return ResolvedWeights(
            files={"probe_tensors": t, "probe_meta": j, "encoder_weights": tmp_path / "enc.pt"},
            sha256={},
        )

    adapter = USProbeAdapter(_manifest("us-probe-liver-steatosis"))
    with pytest.raises(ManifestError, match="ajustada sobre"):
        adapter.load(weights_for(_probe("liver_steatosis", encoder_id="outro@x")), "cpu")
    with pytest.raises(ManifestError, match="rótulo da sonda"):
        adapter.load(weights_for(_probe("outro_rotulo")), "cpu")
    adapter.load(weights_for(_probe("liver_steatosis")), "cpu")
    assert adapter.loaded
    assert isinstance(adapter.encoder, _Enc)


@pytest.mark.requirement("REQ-148")
def test_us_probe_backend_missing_names_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.util

    from radiologyai.analysis.adapters import us_medsam, us_probe

    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name == "torch" else real(name, *a)
    )
    with pytest.raises(BackendUnavailableError, match="ml,cpu"):
        us_probe.BiomedCLIPImageEncoder(Path("nada.pt"))
    with pytest.raises(BackendUnavailableError, match="ml,cpu"):
        us_medsam.MedSAMSegmenter(Path("nada"))


@pytest.mark.requirement("REQ-148")
@pytest.mark.skipif(not have("torch", "open_clip", "timm"), reason="torch/open_clip/timm ausentes")
def test_us_biomedclip_tower_loads_strict_and_embeds(tmp_path: Path) -> None:
    import torch
    from open_clip.timm_model import TimmModel

    from radiologyai.analysis.adapters.us_probe import BiomedCLIPImageEncoder

    empty = tmp_path / "empty.pt"
    torch.save({"text.w": torch.zeros(1)}, str(empty))
    with pytest.raises(ManifestError, match="visual"):
        BiomedCLIPImageEncoder(empty)
    torch.manual_seed(0)
    tower = TimmModel(
        "vit_base_patch16_224",
        embed_dim=512,
        image_size=224,
        pool="",
        proj="linear",
        proj_bias=False,
    )
    ckpt = tmp_path / "biomedclip.pt"
    torch.save({f"visual.{k}": v for k, v in tower.state_dict().items()}, str(ckpt))
    enc = BiomedCLIPImageEncoder(ckpt)
    one = enc.embed(np.zeros((224, 224), dtype=np.uint8))
    batch = enc.embed(np.zeros((2, 224, 224), dtype=np.uint8))
    assert one.shape == (1, 512)
    assert batch.shape == (2, 512)
    np.testing.assert_allclose(batch[0], one[0], rtol=1e-4, atol=1e-5)


# --------------------------------------------------------------------------- #
# MedSAM (REQ-147)
# --------------------------------------------------------------------------- #
class _EmptySeg:
    def embed(self, frame: np.ndarray) -> str:
        return f"emb{frame.shape}"

    def segment(self, frame: np.ndarray, box: np.ndarray, embedding: object = None) -> Any:
        assert box.shape == (4,)
        assert embedding == f"emb{frame.shape}"
        return np.zeros(frame.shape, bool), 0.1


@pytest.mark.requirement("REQ-147")
def test_medsam_adapter_guards_and_empty_mask_abstains() -> None:
    from radiologyai.analysis.adapters.us_medsam import MedSAMUSAdapter, _spacing2

    assert _spacing2(None) is None
    assert _spacing2((0.5,)) is None
    assert _spacing2((1.0, 0.2, 0.3)) == (0.2, 0.3)
    adapter = MedSAMUSAdapter(_manifest("medsam-vit-base"))
    clip = us_image(np.stack([sector_frame(60, 80)] * 3), kind=ImageKind.CINE)
    with pytest.raises(RuntimeError, match="antes de load"):
        adapter.predict(ModelInputs(images=(clip,), exam_type=ExamType.US_OTHER))
    nobox = ModelInputs(images=(clip,), exam_type=ExamType.US_OTHER)
    assert AbstentionReason.MISSING_INPUT in adapter.applicability(nobox).reasons
    late = Primitive(type="bbox", xy=((10.0, 10.0), (50.0, 40.0)), frame=7)
    bad = ModelInputs(images=(clip,), exam_type=ExamType.US_OTHER, prompts=(late,))
    assert AbstentionReason.INPUT in adapter.applicability(bad).reasons

    adapter.segmenter = _EmptySeg()  # type: ignore[assignment]
    box = Primitive(type="bbox", xy=((10.0, 10.0), (50.0, 40.0)), frame=2)
    out = adapter.predict(
        ModelInputs(images=(clip,), exam_type=ExamType.US_OTHER, prompts=(box, box))
    )
    assert len(out.measurements) == 6
    assert all(m.abstention is AbstentionReason.QUALITY for m in out.measurements)
    assert out.labelmaps[0].array.shape == (3, 60, 80)
    assert not out.labelmaps[0].array.any()
    adapter.unload()
    assert adapter.segmenter is None


def _tiny_sam_dir(tmp_path: Path) -> Path:
    import torch
    from transformers import SamConfig, SamModel

    cfg = SamConfig(
        vision_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "mlp_dim": 64,
            "output_channels": 32,
            "global_attn_indexes": [0],
            "window_size": 0,
            "image_size": 1024,
            "patch_size": 16,
            "num_pos_feats": 16,
        },
        prompt_encoder_config={"hidden_size": 32, "image_size": 1024, "patch_size": 16},
        mask_decoder_config={
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "mlp_dim": 64,
            "iou_head_hidden_dim": 32,
        },
    )
    torch.manual_seed(0)
    model = SamModel(cfg)
    out = tmp_path / "medsam"
    model.save_pretrained(str(out))
    return out


@pytest.mark.requirement("REQ-147")
@pytest.mark.skipif(not have("torch", "transformers", "PIL", "scipy"), reason="extras ausentes")
def test_medsam_segmenter_and_adapter_with_tiny_sam(tmp_path: Path) -> None:
    from radiologyai.analysis.adapters.us_medsam import MedSAMSegmenter, MedSAMUSAdapter

    model_dir = _tiny_sam_dir(tmp_path)
    seg = MedSAMSegmenter(model_dir)
    frame = sector_frame(60, 80).astype(np.uint8)
    emb = seg.embed(frame)
    mask, iou = seg.segment(frame, np.array([10.0, 10.0, 60.0, 50.0]), emb)
    assert mask.shape == (60, 80)
    assert mask.dtype == np.bool_
    assert np.isfinite(iou)

    adapter = MedSAMUSAdapter(_manifest("medsam-vit-base"))
    weights_file = next(model_dir.glob("*.safetensors"))
    adapter.load(ResolvedWeights(files={"model": weights_file}, sha256={"model": "0" * 64}), "cpu")
    box = Primitive(type="bbox", xy=((10.0, 10.0), (60.0, 50.0)))
    out = adapter.predict(
        ModelInputs(
            images=(us_image(sector_frame(60, 80), spacing=(0.2, 0.2)),),
            exam_type=ExamType.US_OTHER,
            prompts=(box,),
        )
    )
    assert {m.key for m in out.measurements} == {
        "lesion1_area",
        "lesion1_max_diameter",
        "lesion1_perp_diameter",
    }


def test_us_support_digest_is_content_hash() -> None:
    img = us_image(np.zeros((2, 2)))
    expected = hashlib.sha256(np.zeros((2, 2), np.float32).tobytes()).hexdigest()
    assert img.sha256 == expected
    assert json.loads(json.dumps({"ok": True}))["ok"]
