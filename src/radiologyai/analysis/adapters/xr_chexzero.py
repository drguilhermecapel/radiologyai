"""CheXzero (Tiu et al., Nat Biomed Eng 2022) — classificação zero-shot de RX de tórax.

SOMENTE MODO PESQUISA (``task: zero_shot``): não há ponto de operação nem
validação por condição; o nível de evidência é D até medição externa aprovada.

Reprodução do ``zero_shot.py`` do repositório ``rajpurkarlab/CheXzero``:

* imagem → 8 bits pela faixa representável → maior lado redimensionado para 320
  (LANCZOS), preenchida com zeros até 320x320 centralizada, tons de cinza;
* tensor 3 canais, ``Normalize(101.48761, 83.43944)``, ``Resize(224, bicúbico)``;
* para cada rótulo, par de prompts ``"{rótulo}"`` / ``"no {rótulo}"``; embeddings
  de texto e de imagem normalizados; escore = softmax dos dois cossenos (sem
  ``logit_scale``), média entre os checkpoints do manifesto.

A arquitetura é o CLIP ViT-B/32 do ``open_clip`` (MIT, extra ``foundation``),
que aceita o ``state_dict`` do CheXzero com todas as chaves; o tokenizador é o
BPE do CLIP distribuído pelo ``open_clip``. Nenhum código do repositório
original é executado.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.xr_common import load_torch_verified, require_backend, to_gray
from radiologyai.analysis.adapters.xr_xrv import gate_checks
from radiologyai.analysis.base import (
    AdapterOutput,
    ModelAdapter,
    ModelInputs,
    ResolvedWeights,
    Score,
)
from radiologyai.analysis.types import ImageData, QAReport
from radiologyai.errors import InferenceError, ManifestError, UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from radiologyai.analysis.manifest import ModelManifest

__all__ = ["ChexzeroAdapter", "build_chexzero_clip", "chexzero_image_tensor"]

_PAD_SIZE = 320
_INPUT_SIZE = 224
_MEAN = 101.48761
_STD = 83.43944
_NON_WEIGHT_KEYS = ("input_resolution", "context_length", "vocab_size")


def build_chexzero_clip(state: dict[str, Any]) -> torch.nn.Module:
    """CLIP ViT-B/32 (``open_clip``) com o ``state_dict`` do CheXzero, em float32.

    Raises:
        ManifestError: o checkpoint não casa com a arquitetura (chave faltando/sobrando).
    """
    from open_clip.model import CLIP, CLIPTextCfg, CLIPVisionCfg

    model = CLIP(
        embed_dim=512,
        vision_cfg=CLIPVisionCfg(width=768, layers=12, patch_size=32, image_size=224),
        text_cfg=CLIPTextCfg(context_length=77, vocab_size=49408, width=512, heads=8, layers=12),
        quick_gelu=True,
    )
    weights = {k: v for k, v in state.items() if k not in _NON_WEIGHT_KEYS}
    result = model.load_state_dict(weights, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ManifestError(
            "checkpoint CheXzero incompatível com CLIP ViT-B/32: "
            f"faltando {result.missing_keys[:5]}, sobrando {result.unexpected_keys[:5]}"
        )
    out: torch.nn.Module = model.float().eval()
    return out


def _to_uint8(image: ImageData) -> npt.NDArray[np.uint8]:
    import numpy as np

    plane = to_gray(image)
    lo, hi = image.value_range
    if not np.isfinite(plane).all():
        raise UnsupportedInputError("imagem com pixels não finitos")
    scaled = (plane - np.float32(lo)) / np.float32(hi - lo) * np.float32(255.0)
    if scaled.min() < -0.5 or scaled.max() > 255.5:
        raise UnsupportedInputError("pixels fora da faixa representável declarada")
    out: npt.NDArray[np.uint8] = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    return out


def chexzero_image_tensor(image: ImageData) -> torch.Tensor:
    """Pré-processamento do CheXzero → tensor ``[1, 3, 224, 224]``."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize

    pil = Image.fromarray(_to_uint8(image), mode="L")
    ratio = float(_PAD_SIZE) / max(pil.size)
    new_size = (int(pil.size[0] * ratio), int(pil.size[1] * ratio))
    pil = pil.resize(new_size, Image.Resampling.LANCZOS)
    canvas = Image.new("L", (_PAD_SIZE, _PAD_SIZE))
    canvas.paste(pil, ((_PAD_SIZE - new_size[0]) // 2, (_PAD_SIZE - new_size[1]) // 2))
    x = torch.from_numpy(np.asarray(canvas, dtype=np.float32))[None].repeat(3, 1, 1)
    transform = Compose(
        [
            Normalize((_MEAN,) * 3, (_STD,) * 3),
            Resize(_INPUT_SIZE, interpolation=InterpolationMode.BICUBIC, antialias=False),
        ]
    )
    out: torch.Tensor = transform(x)[None]
    return out


class ChexzeroAdapter(ModelAdapter):
    """Escore zero-shot por par de prompts; média dos checkpoints declarados."""

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        if manifest.task != "zero_shot":
            raise ManifestError(f"{manifest.id}: CheXzero só pode ser declarado como zero_shot")
        self._models: list[Any] = []
        self._text: list[tuple[Any, Any]] = []  # por checkpoint: (positivos, negativos)
        cfg = manifest.config
        templates = tuple(str(t) for t in cfg.get("templates", ("{}", "no {}")))
        if len(templates) != 2:
            raise ManifestError(f"{manifest.id}: config.templates deve ter 2 itens (pos, neg)")
        self.templates: tuple[str, str] = (templates[0], templates[1])
        self.gate_ids: tuple[str, ...] = tuple(cfg.get("gates", ()))

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_backend("torch", "torchvision", "open_clip", "PIL", extra="foundation")
        import torch
        from open_clip.tokenizer import tokenize

        prompts = [spec.source_label for spec in self.manifest.labels]
        pos_tokens = tokenize([self.templates[0].format(p) for p in prompts])
        neg_tokens = tokenize([self.templates[1].format(p) for p in prompts])
        dev = torch.device(device)
        models, text = [], []
        for weight in self.manifest.weights:
            state = load_torch_verified(weights, weight.key, allow_pickle=False)
            model: Any = build_chexzero_clip(state).to(dev)
            with torch.inference_mode():
                pos = model.encode_text(pos_tokens.to(dev)).float()
                neg = model.encode_text(neg_tokens.to(dev)).float()
            text.append(
                (pos / pos.norm(dim=-1, keepdim=True), neg / neg.norm(dim=-1, keepdim=True))
            )
            models.append(model)
        self._models, self._text = models, text

    def unload(self) -> None:
        self._models, self._text = [], []

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        if not base.applicable or not self.gate_ids:
            return base
        checks, reasons = gate_checks(inputs, self.gate_ids)
        all_checks = (*base.checks, *checks)
        if reasons:
            return QAReport.reject(*dict.fromkeys(reasons), checks=all_checks)
        return QAReport.ok(all_checks)

    def probabilities(self, image: ImageData) -> npt.NDArray[np.float64]:
        """Softmax (prompt positivo vs negativo) por rótulo, média dos checkpoints."""
        import numpy as np
        import torch

        if not self._models:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        per_ckpt = []
        for model, (pos, neg) in zip(self._models, self._text, strict=True):
            x = chexzero_image_tensor(image).to(pos.device)
            with torch.inference_mode():
                f = model.encode_image(x).float()
                f = f / f.norm(dim=-1, keepdim=True)
                s_pos = (f @ pos.T)[0].double()
                s_neg = (f @ neg.T)[0].double()
            p = torch.exp(s_pos) / (torch.exp(s_pos) + torch.exp(s_neg))
            per_ckpt.append(p.cpu().numpy())
        out: npt.NDArray[np.float64] = np.mean(np.stack(per_ckpt), axis=0)
        return out

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        probs = self.probabilities(inputs.single)
        scores = [
            Score(label=spec.key, value=float(probs[i]), kind="softmax")
            for i, spec in enumerate(self.manifest.labels)
        ]
        return AdapterOutput(
            scores=scores,
            warnings=["zero-shot (CheXzero): exploratório, sem ponto de operação; modo pesquisa"],
            extras={"image_sha256": inputs.single.sha256, "n_checkpoints": len(self._models)},
        )
