"""Construção do SAM 2.1 Hiera-T a 512 px (configuração do MedSAM2) sem hydra.

A configuração abaixo transcreve ``sam2.1_hiera_t512.yaml`` de bowang-lab/MedSAM2
(Apache-2.0). ``_target_`` só pode nomear classes da lista branca :data:`_CLASSES`
(nenhuma importação dinâmica de texto). O checkpoint é lido com
``torch.load(weights_only=True)`` e carregado com ``strict=True``: chave faltante
ou sobrando é erro, nunca peso aleatório silencioso.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from radiologyai.analysis.vendor.sam2.modeling.backbones.hieradet import Hiera
from radiologyai.analysis.vendor.sam2.modeling.backbones.image_encoder import (
    FpnNeck,
    ImageEncoder,
)
from radiologyai.analysis.vendor.sam2.modeling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)
from radiologyai.analysis.vendor.sam2.modeling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)
from radiologyai.analysis.vendor.sam2.modeling.position_encoding import PositionEmbeddingSine
from radiologyai.analysis.vendor.sam2.modeling.sam.transformer import RoPEAttention
from radiologyai.analysis.vendor.sam2.modeling.sam2_base import SAM2Base

__all__ = ["HIERA_T512_CONFIG", "build_sam2_hiera_t512"]

_CLASSES: dict[str, Any] = {
    "SAM2Base": SAM2Base,
    "ImageEncoder": ImageEncoder,
    "Hiera": Hiera,
    "FpnNeck": FpnNeck,
    "PositionEmbeddingSine": PositionEmbeddingSine,
    "MemoryAttention": MemoryAttention,
    "MemoryAttentionLayer": MemoryAttentionLayer,
    "RoPEAttention": RoPEAttention,
    "MemoryEncoder": MemoryEncoder,
    "MaskDownSampler": MaskDownSampler,
    "Fuser": Fuser,
    "CXBlock": CXBlock,
}

_ROPE_SELF: dict[str, Any] = {
    "_target_": "RoPEAttention",
    "rope_theta": 10000.0,
    "feat_sizes": [32, 32],
    "embedding_dim": 256,
    "num_heads": 1,
    "downsample_rate": 1,
    "dropout": 0.1,
}

HIERA_T512_CONFIG: dict[str, Any] = {
    "_target_": "SAM2Base",
    "image_encoder": {
        "_target_": "ImageEncoder",
        "scalp": 1,
        "trunk": {
            "_target_": "Hiera",
            "embed_dim": 96,
            "num_heads": 1,
            "stages": [1, 2, 7, 2],
            "global_att_blocks": [5, 7, 9],
            "window_pos_embed_bkg_spatial_size": [7, 7],
        },
        "neck": {
            "_target_": "FpnNeck",
            "position_encoding": {
                "_target_": "PositionEmbeddingSine",
                "num_pos_feats": 256,
                "normalize": True,
                "scale": None,
                "temperature": 10000,
            },
            "d_model": 256,
            "backbone_channel_list": [768, 384, 192, 96],
            "fpn_top_down_levels": [2, 3],
            "fpn_interp_model": "nearest",
        },
    },
    "memory_attention": {
        "_target_": "MemoryAttention",
        "d_model": 256,
        "pos_enc_at_input": True,
        "layer": {
            "_target_": "MemoryAttentionLayer",
            "activation": "relu",
            "dim_feedforward": 2048,
            "dropout": 0.1,
            "pos_enc_at_attn": False,
            "self_attention": _ROPE_SELF,
            "d_model": 256,
            "pos_enc_at_cross_attn_keys": True,
            "pos_enc_at_cross_attn_queries": False,
            "cross_attention": {**_ROPE_SELF, "rope_k_repeat": True, "kv_in_dim": 64},
        },
        "num_layers": 4,
    },
    "memory_encoder": {
        "_target_": "MemoryEncoder",
        "out_dim": 64,
        "position_encoding": {
            "_target_": "PositionEmbeddingSine",
            "num_pos_feats": 64,
            "normalize": True,
            "scale": None,
            "temperature": 10000,
        },
        "mask_downsampler": {
            "_target_": "MaskDownSampler",
            "kernel_size": 3,
            "stride": 2,
            "padding": 1,
        },
        "fuser": {
            "_target_": "Fuser",
            "layer": {
                "_target_": "CXBlock",
                "dim": 256,
                "kernel_size": 7,
                "padding": 3,
                "layer_scale_init_value": 1e-6,
                "use_dwconv": True,
            },
            "num_layers": 2,
        },
    },
    "num_maskmem": 7,
    "image_size": 512,
    "sigmoid_scale_for_mem_enc": 20.0,
    "sigmoid_bias_for_mem_enc": -10.0,
    "use_mask_input_as_output_without_sam": True,
    "directly_add_no_mem_embed": True,
    "no_obj_embed_spatial": True,
    "use_high_res_features_in_sam": True,
    "multimask_output_in_sam": True,
    "iou_prediction_use_sigmoid": True,
    "use_obj_ptrs_in_encoder": True,
    "add_tpos_enc_to_obj_ptrs": True,
    "proj_tpos_enc_in_obj_ptrs": True,
    "use_signed_tpos_enc_to_obj_ptrs": True,
    "only_obj_ptrs_in_the_past_for_eval": True,
    "pred_obj_scores": True,
    "pred_obj_scores_mlp": True,
    "fixed_no_obj_ptr": True,
    "multimask_output_for_tracking": True,
    "use_multimask_token_for_obj_ptr": True,
    "multimask_min_pt_num": 0,
    "multimask_max_pt_num": 1,
    "use_mlp_for_obj_ptr_proj": True,
    "compile_image_encoder": False,
}

# Sobrescritas que build_sam.py aplica em modo de avaliação (apply_postprocessing=True).
_EVAL_OVERRIDES: dict[str, Any] = {
    "sam_mask_decoder_extra_args": {
        "dynamic_multimask_via_stability": True,
        "dynamic_multimask_stability_delta": 0.05,
        "dynamic_multimask_stability_thresh": 0.98,
    }
}


def _instantiate(node: Any) -> Any:
    if isinstance(node, dict):
        kwargs = {k: _instantiate(v) for k, v in node.items() if k != "_target_"}
        target = node.get("_target_")
        if target is None:
            return kwargs
        cls = _CLASSES.get(str(target))
        if cls is None:
            raise ValueError(f"classe fora da lista branca: {target!r}")
        return cls(**kwargs)
    if isinstance(node, list):
        return [_instantiate(v) for v in node]
    return node


def build_sam2_hiera_t512(checkpoint: Path, device: str = "cpu") -> Any:
    """Constrói o ``SAM2Base`` e carrega o checkpoint do MedSAM2 (chave ``model``).

    Raises:
        RuntimeError: chaves do ``state_dict`` não batem com a arquitetura.
        KeyError: checkpoint sem a chave ``model``.
    """
    config = copy.deepcopy(HIERA_T512_CONFIG)
    config.update(copy.deepcopy(_EVAL_OVERRIDES))
    model = _instantiate(config)
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)["model"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model
