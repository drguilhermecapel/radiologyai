"""SAM 2 vendorizado: construção, carga estrita e caminhos de inferência SEM pesos reais.

A arquitetura Hiera-T 512 do MedSAM2 é instanciada com inicialização
determinística (``torch.manual_seed``) e gravada num checkpoint temporário.
Isso exercita o encanamento (construção pela lista branca, ``strict=True``,
pré/pós-processamento do preditor, cabeças SAM, codificador e atenção de
memória) — as máscaras produzidas por pesos não treinados NÃO têm significado e
os testes só conferem forma, tipo e contratos de erro. Nada disto é distribuído
nem alcança ``src/``.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from us_support import have, sector_frame, us_image

pytestmark = pytest.mark.skipif(not have("torch", "scipy"), reason="torch/scipy ausentes")


def _untrained_model() -> Any:
    import torch

    from radiologyai.analysis.vendor.sam2 import build

    torch.manual_seed(0)
    config = copy.deepcopy(build.HIERA_T512_CONFIG)
    config.update(copy.deepcopy(build._EVAL_OVERRIDES))
    return build._instantiate(config).eval()


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import torch

    path = tmp_path_factory.mktemp("sam2") / "untrained.pt"
    torch.save({"model": _untrained_model().state_dict()}, str(path))
    return path


@pytest.fixture(scope="module")
def model(checkpoint: Path) -> Any:
    from radiologyai.analysis.vendor.sam2.build import build_sam2_hiera_t512

    return build_sam2_hiera_t512(checkpoint)


@pytest.fixture(scope="module")
def predictor(model: Any) -> Any:
    from radiologyai.analysis.adapters.echo_medsam2 import _FEAT_SIZES_512
    from radiologyai.analysis.vendor.sam2.sam2_image_predictor import SAM2ImagePredictor

    pred = SAM2ImagePredictor(model)
    pred._bb_feat_sizes = list(_FEAT_SIZES_512)
    return pred


def _rgb(h: int = 96, w: int = 128) -> np.ndarray:
    frame = sector_frame(h, w).astype(np.uint8)
    return np.repeat(frame[..., None], 3, axis=2)


@pytest.mark.requirement("REQ-146")
def test_builder_rejects_unknown_class_and_bad_checkpoints(tmp_path: Path) -> None:
    import torch

    from radiologyai.analysis.vendor.sam2 import build

    with pytest.raises(ValueError, match="lista branca"):
        build._instantiate({"_target_": "os.system"})
    assert build._instantiate({"a": [1, {"b": 2}]}) == {"a": [1, {"b": 2}]}

    no_model_key = tmp_path / "nokey.pt"
    torch.save({"weights": {}}, str(no_model_key))
    with pytest.raises(KeyError):
        build.build_sam2_hiera_t512(no_model_key)

    partial = tmp_path / "partial.pt"
    state = _untrained_model().state_dict()
    state.pop(next(iter(state)))
    torch.save({"model": state}, str(partial))
    with pytest.raises(RuntimeError, match="Missing key"):
        build.build_sam2_hiera_t512(partial)


@pytest.mark.requirement("REQ-146")
def test_strict_load_reproduces_saved_parameters(model: Any, checkpoint: Path) -> None:
    import torch

    saved = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
    current = model.state_dict()
    assert set(saved) == set(current)
    for key in list(saved)[:: max(1, len(saved) // 20)]:
        assert torch.equal(saved[key], current[key])
    assert not model.training
    assert model.image_size == 512


@pytest.mark.requirement("REQ-146")
def test_predictor_box_point_and_mask_prompts(predictor: Any) -> None:
    with pytest.raises(RuntimeError):
        predictor.reset_predictor()
        predictor.predict(box=np.array([1.0, 1.0, 20.0, 20.0], dtype=np.float32))
    with pytest.raises(RuntimeError):
        predictor.get_image_embedding()

    img = _rgb()
    predictor.set_image(img)
    emb = predictor.get_image_embedding()
    assert tuple(emb.shape) == (1, 256, 32, 32)
    assert str(predictor.device) == "cpu"

    box = np.array([30.0, 20.0, 90.0, 80.0], dtype=np.float32)
    masks, iou, low = predictor.predict(box=box, multimask_output=False)
    assert masks.shape == (1, 96, 128)
    assert iou.shape == (1,)
    assert low.shape == (1, 128, 128)

    pts = np.array([[60.0, 50.0], [40.0, 30.0]], dtype=np.float32)
    lab = np.array([1, 0], dtype=np.int32)
    masks3, iou3, low3 = predictor.predict(point_coords=pts, point_labels=lab, multimask_output=True)
    assert masks3.shape == (3, 96, 128)
    assert iou3.shape == (3,)

    masks_m, _, _ = predictor.predict(
        point_coords=pts,
        point_labels=lab,
        mask_input=low3[:1],
        multimask_output=False,
        return_logits=True,
    )
    assert masks_m.dtype == np.float32

    with pytest.raises(AssertionError):
        predictor.predict(point_coords=pts, multimask_output=False)


@pytest.mark.requirement("REQ-146")
def test_predictor_batch_api(predictor: Any) -> None:
    imgs = [_rgb(64, 80), _rgb(72, 72)]
    predictor.set_image_batch(imgs)
    boxes = [np.array([5.0, 5.0, 50.0, 40.0], dtype=np.float32)] * 2
    masks, ious, lows = predictor.predict_batch(box_batch=boxes, multimask_output=False)
    assert [m.shape for m in masks] == [(1, 64, 80), (1, 72, 72)]
    assert len(ious) == len(lows) == 2
    with pytest.raises(NotImplementedError):
        predictor.set_image("não é imagem")


@pytest.mark.requirement("REQ-146")
def test_track_step_with_memory_bank(model: Any) -> None:
    """Dois quadros: o 1º é condicionante (ponto), o 2º usa o banco de memória."""
    import torch

    torch.manual_seed(1)
    imgs = torch.rand(2, 3, 512, 512)
    outputs: dict[str, dict[int, Any]] = {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
    with torch.inference_mode():
        for idx in range(2):
            backbone = model.forward_image(imgs[idx : idx + 1])
            _, feats, pos, sizes = model._prepare_backbone_features(backbone)
            points = (
                {
                    "point_coords": torch.tensor([[[256.0, 256.0]]]),
                    "point_labels": torch.tensor([[1]], dtype=torch.int32),
                }
                if idx == 0
                else None
            )
            out = model.track_step(
                frame_idx=idx,
                is_init_cond_frame=idx == 0,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=sizes,
                point_inputs=points,
                mask_inputs=None,
                output_dict=outputs,
                num_frames=2,
            )
            key = "cond_frame_outputs" if idx == 0 else "non_cond_frame_outputs"
            outputs[key][idx] = out
            assert tuple(out["pred_masks"].shape) == (1, 1, 128, 128)
            assert tuple(out["pred_masks_high_res"].shape) == (1, 1, 512, 512)
            assert out["maskmem_features"] is not None

        # máscara de entrada usada diretamente como saída (sem cabeças SAM)
        backbone = model.forward_image(imgs[:1])
        _, feats, pos, sizes = model._prepare_backbone_features(backbone)
        mask_in = torch.zeros(1, 1, 512, 512)
        mask_in[..., 200:300, 200:300] = 1.0
        out = model.track_step(
            frame_idx=0,
            is_init_cond_frame=True,
            current_vision_feats=feats,
            current_vision_pos_embeds=pos,
            feat_sizes=sizes,
            point_inputs=None,
            mask_inputs=mask_in,
            output_dict={"cond_frame_outputs": {}, "non_cond_frame_outputs": {}},
            num_frames=1,
            run_mem_encoder=False,
        )
        assert out["maskmem_features"] is None
        assert bool((out["pred_masks_high_res"][..., 250, 250] > 0).all())
        constrained = model._apply_non_overlapping_constraints(torch.randn(2, 1, 8, 8))
        assert tuple(constrained.shape) == (2, 1, 8, 8)


@pytest.mark.requirement("REQ-146")
def test_sam2_utils_helpers() -> None:
    import torch

    from radiologyai.analysis.vendor.sam2.modeling import sam2_utils
    from radiologyai.analysis.vendor.sam2.utils import misc

    cond = {i: {"i": i} for i in (0, 4, 9, 12)}
    sel, unsel = sam2_utils.select_closest_cond_frames(10, cond, max_cond_frame_num=2)
    assert set(sel) == {9, 12} and set(unsel) == {0, 4}
    sel_all, unsel_all = sam2_utils.select_closest_cond_frames(10, cond, max_cond_frame_num=-1)
    assert set(sel_all) == set(cond) and not unsel_all
    pe = sam2_utils.get_1d_sine_pe(torch.arange(5.0), dim=8)
    assert tuple(pe.shape) == (5, 8)
    for name in ("relu", "gelu", "glu"):
        assert callable(sam2_utils.get_activation_fn(name))
    with pytest.raises(RuntimeError):
        sam2_utils.get_activation_fn("nada")
    assert len(sam2_utils.get_clones(torch.nn.Linear(2, 2), 3)) == 3
    drop = sam2_utils.DropPath(0.5).eval()
    x = torch.ones(2, 3)
    assert torch.equal(drop(x), x)
    mlp = sam2_utils.MLP(4, 8, 2, num_layers=2, sigmoid_output=True)
    assert tuple(mlp(torch.zeros(1, 4)).shape) == (1, 2)
    assert tuple(sam2_utils.LayerNorm2d(3)(torch.ones(1, 3, 2, 2)).shape) == (1, 3, 2, 2)

    masks = torch.zeros(2, 1, 10, 12, dtype=torch.bool)
    masks[0, 0, 2:5, 3:9] = True
    boxes = misc.mask_to_box(masks)
    assert boxes[0, 0].tolist() == [3, 2, 8, 4]
    assert boxes[1, 0].tolist() == [0, 0, 0, 0]
    pts, labs = misc.concat_points(None, torch.zeros(1, 1, 2), torch.ones(1, 1))
    pts2, labs2 = misc.concat_points({"point_coords": pts, "point_labels": labs}, pts, labs)
    assert tuple(pts2["point_coords"].shape) == (1, 2, 2) if isinstance(pts2, dict) else True
    assert isinstance(misc.get_sdpa_settings(), tuple)
    with pytest.raises(ImportError):
        misc.get_connected_components(masks)


@pytest.mark.requirement("REQ-146")
def test_medsam2_segmenter_and_adapter_load(checkpoint: Path) -> None:
    from radiologyai.analysis.adapters.echo_medsam2 import (
        MedSAM2EchoLVAdapter,
        MedSAM2Segmenter,
    )
    from radiologyai.analysis.base import ModelInputs, ResolvedWeights
    from radiologyai.analysis.manifest import load_manifest
    from radiologyai.analysis.types import ImageKind, Primitive
    from radiologyai.catalog.schema import ExamType

    seg = MedSAM2Segmenter(checkpoint)
    frame = sector_frame(80, 100).astype(np.uint8)
    mask, iou = seg.segment(frame, np.array([20.0, 15.0, 70.0, 60.0]))
    assert mask.shape == (80, 100) and mask.dtype == np.bool_
    assert np.isfinite(iou)

    manifests = Path(__file__).resolve().parents[3] / "src/radiologyai/analysis/manifests"
    manifest, _ = load_manifest(manifests / "medsam2-echo-lv.yaml")
    adapter = MedSAM2EchoLVAdapter(manifest)
    with pytest.raises(RuntimeError):
        adapter.predict(ModelInputs(images=(), exam_type=ExamType.ECHO))
    weights = ResolvedWeights(files={"checkpoint": checkpoint}, sha256={"checkpoint": "0" * 64})
    adapter.load(weights, "cpu")
    assert adapter.loaded and adapter.segmenter is not None
    clip = us_image(
        np.stack([sector_frame(80, 100)] * 3), kind=ImageKind.CINE, spacing=(0.3, 0.3)
    )
    prompts = (
        Primitive(type="bbox", xy=((25.0, 15.0), (75.0, 65.0)), frame=0, label="a4c:ed"),
        Primitive(type="bbox", xy=((30.0, 20.0), (70.0, 60.0)), frame=2, label="a4c:es"),
    )
    out = adapter.predict(
        ModelInputs(images=(clip,), exam_type=ExamType.ECHO, roles={"a4c": 0}, prompts=prompts)
    )
    assert {m.key for m in out.measurements} >= {"lv_ef", "lv_edv", "lv_esv"}
    assert out.labelmaps[0].array.shape == (3, 80, 100)
    adapter.unload()
    assert adapter.segmenter is None and not adapter.loaded
