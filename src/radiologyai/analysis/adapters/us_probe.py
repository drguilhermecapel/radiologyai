"""Sondas lineares de US sobre *embeddings* congelados do BiomedCLIP (torre de imagem).

Codificador: ``microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`` —
só a torre visual (ViT-B/16 do timm + projeção linear para 512), construída com
``open_clip.timm_model.TimmModel`` e carregada com ``strict=True`` a partir das
chaves ``visual.*`` do checkpoint conferido (``torch.load(weights_only=True)``).
Licença MIT; o cartão restringe o uso a pesquisa → ``commercial_use: no``.

Pré-processamento (fixo; o MESMO no ajuste e na inferência —
:func:`us_canonical` é chamado pelo script de ajuste):

1. quadro → 8 bits pela faixa representável (cor → luminância);
2. ``roi: box`` — recorte da caixa do usuário ampliada em ``margin`` (fração
   do lado da caixa, em cada direção), recortada à imagem; ``roi: full`` —
   quadro inteiro;
3. preenchimento com zeros até quadrado (centralizado) e redimensionamento
   PIL bicúbico para 224 × 224, RGB;
4. ``/255`` e normalização (média/desvio do ``open_clip_config.json``);
5. vetor de 512 da projeção (sem normalização L2).

Clipe (cine) com ``roi: full``: média dos logits de até ``max_frames`` quadros
igualmente espaçados — a mesma agregação usada na avaliação por paciente.
O escore é o logit da sonda; decisão, calibração e nível de evidência são do
pipeline.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.us_common import bbox_array, box_prompts, frame_uint8
from radiologyai.analysis.base import (
    AdapterOutput,
    ModelAdapter,
    ModelInputs,
    ResolvedWeights,
    Score,
)
from radiologyai.analysis.types import (
    AbstentionReason,
    ImageData,
    ImageKind,
    Primitive,
    QACheck,
    QAReport,
)
from radiologyai.errors import BackendUnavailableError, ManifestError, UnsupportedInputError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    import torch

    from radiologyai.analysis.manifest import ModelManifest
    from radiologyai.analysis.probes.artifact import ProbeArtifact

__all__ = [
    "BIOMEDCLIP_ENCODER_ID",
    "BiomedCLIPImageEncoder",
    "USProbeAdapter",
    "expand_box",
    "us_canonical",
]

BIOMEDCLIP_ENCODER_ID = (
    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    "@9f341de24bfb00180f1b847274256e9b65a3a32e"
)
_SIZE = 224
_MEAN = (0.48145466, 0.4578275, 0.40821073)
_STD = (0.26862954, 0.26130258, 0.27577711)
_WEIGHT_KEYS = {"encoder_weights", "probe_tensors", "probe_meta"}


def _require() -> None:
    for mod, extra in (
        ("torch", "ml,cpu"),
        ("open_clip", "foundation"),
        ("timm", "foundation"),
        ("PIL", "imaging"),
        ("safetensors", "foundation"),
    ):
        if importlib.util.find_spec(mod) is None:
            raise BackendUnavailableError(
                f"{mod} não instalado; instale os extras: pip install 'radiologyai[{extra}]'"
            )


def expand_box(
    box: npt.NDArray[np.float64], shape: tuple[int, int], margin: float
) -> tuple[int, int, int, int]:
    """Caixa ``[x0, y0, x1, y1]`` ampliada em ``margin`` × lado, inteira e recortada à imagem."""
    import math

    h, w = shape
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0 = max(0, int(math.floor(box[0] - margin * bw)))
    y0 = max(0, int(math.floor(box[1] - margin * bh)))
    x1 = min(w, int(math.ceil(box[2] + margin * bw)) + 1)
    y1 = min(h, int(math.ceil(box[3] + margin * bh)) + 1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        raise UnsupportedInputError(f"região de interesse degenerada: {(x0, y0, x1, y1)}")
    return x0, y0, x1, y1


def us_canonical(
    frame_u8: npt.NDArray[np.uint8],
    box: npt.NDArray[np.float64] | None,
    margin: float,
) -> npt.NDArray[np.uint8]:
    """Passos 2-3: recorte (opcional), quadrado com zeros e 224 × 224 bicúbico (um canal)."""
    import numpy as np
    from PIL import Image

    arr = frame_u8
    if box is not None:
        x0, y0, x1, y1 = expand_box(box, arr.shape[:2], margin)
        arr = arr[y0:y1, x0:x1]
    h, w = arr.shape
    s = max(h, w)
    canvas = np.zeros((s, s), dtype=np.uint8)
    top, left = (s - h) // 2, (s - w) // 2
    canvas[top : top + h, left : left + w] = arr
    out: npt.NDArray[np.uint8] = np.asarray(
        Image.fromarray(canvas, mode="L").resize((_SIZE, _SIZE), Image.Resampling.BICUBIC),
        dtype=np.uint8,
    )
    return out


class BiomedCLIPImageEncoder:
    """Torre visual do BiomedCLIP (512), sem a torre de texto."""

    def __init__(self, checkpoint: Any, device: str = "cpu") -> None:
        _require()
        import torch
        from open_clip.timm_model import TimmModel

        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        visual = {k[len("visual.") :]: v for k, v in state.items() if k.startswith("visual.")}
        if not visual:
            raise ManifestError("checkpoint do BiomedCLIP sem chaves visual.*")
        model = TimmModel(
            "vit_base_patch16_224",
            embed_dim=512,
            image_size=224,
            pool="",
            proj="linear",
            proj_bias=False,
        )
        model.load_state_dict(visual, strict=True)
        self.model = model.to(device).eval()
        self.device = device

    def embed(self, canonical: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """``uint8`` 224 × 224 (ou lote ``[N, 224, 224]``) → ``[N, 512]``."""
        import numpy as np
        import torch

        arr = np.asarray(canonical, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[None]
        x = np.repeat(arr[:, None] / np.float32(255.0), 3, axis=1)
        mean = np.asarray(_MEAN, dtype=np.float32)[None, :, None, None]
        std = np.asarray(_STD, dtype=np.float32)[None, :, None, None]
        t: torch.Tensor = torch.from_numpy(((x - mean) / std).astype(np.float32)).to(self.device)
        with torch.inference_mode():
            out = self.model(t)
        values: npt.NDArray[np.float32] = out.float().cpu().numpy()
        return values


def _frames_for(image: ImageData, max_frames: int) -> list[int]:
    if image.kind is not ImageKind.CINE:
        return [0]
    n = int(image.pixels.shape[0])
    k = min(n, max_frames)
    return sorted({int(round(i * (n - 1) / max(k - 1, 1))) for i in range(k)})


class USProbeAdapter(ModelAdapter):
    """BiomedCLIP congelado + sonda linear fixada; escore logit."""

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        if len(manifest.labels) != 1:
            raise ManifestError(f"{manifest.id}: sonda de US emite exatamente um rótulo")
        roi = str(manifest.config.get("roi", ""))
        if roi not in ("box", "full"):
            raise ManifestError(f"{manifest.id}: config.roi deve ser 'box' ou 'full'")
        if roi == "box" and manifest.inputs.prompts != "required":
            raise ManifestError(f"{manifest.id}: roi 'box' exige inputs.prompts: required")
        keys = {w.key for w in manifest.weights}
        if keys != _WEIGHT_KEYS:
            raise ManifestError(f"{manifest.id}: pesos esperados {sorted(_WEIGHT_KEYS)}")
        self.roi = roi
        self.margin = float(manifest.config.get("margin", 0.0))
        self.max_frames = int(manifest.config.get("max_frames", 10))
        self.encoder: BiomedCLIPImageEncoder | None = None
        self.probe: ProbeArtifact | None = None

    def load(self, weights: ResolvedWeights, device: str) -> None:
        from radiologyai.analysis.probes.artifact import load_probe

        probe = load_probe(
            weights.path("probe_tensors").read_bytes(), weights.path("probe_meta").read_bytes()
        )
        if probe.encoder_id != BIOMEDCLIP_ENCODER_ID:
            raise ManifestError(
                f"{self.manifest.id}: sonda ajustada sobre {probe.encoder_id!r}, "
                f"não sobre {BIOMEDCLIP_ENCODER_ID!r}"
            )
        if probe.label != self.manifest.labels[0].key:
            raise ManifestError(f"{self.manifest.id}: rótulo da sonda {probe.label!r} difere")
        self.encoder = BiomedCLIPImageEncoder(weights.path("encoder_weights"), device=device)
        self.probe = probe
        self.device = device
        self.loaded = True

    def unload(self) -> None:
        self.encoder = None
        self.probe = None
        self.loaded = False

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        if self.roi != "box":
            return base
        checks = list(base.checks)
        reasons = list(base.reasons)
        boxes = box_prompts(inputs.prompts)
        checks.append(QACheck(name="roi_box", passed=len(boxes) == 1, value=float(len(boxes))))
        if len(boxes) != 1:
            reasons.append(AbstentionReason.MISSING_INPUT if not boxes else AbstentionReason.INPUT)
        if reasons:
            return QAReport.reject(*dict.fromkeys(reasons), checks=checks)
        return QAReport.ok(checks)

    def _logit(self, canonical: npt.NDArray[np.uint8]) -> float:
        from radiologyai.analysis.probes.artifact import probe_logit

        assert self.encoder is not None
        assert self.probe is not None
        return float(probe_logit(self.probe, self.encoder.embed(canonical))[0])

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        if self.encoder is None or self.probe is None:
            raise RuntimeError(f"{self.manifest.id}: predict antes de load()")
        image = inputs.single
        label = self.manifest.labels[0].key
        if self.roi == "box":
            prompt: Primitive = box_prompts(inputs.prompts)[0]
            frame = frame_uint8(image, prompt.frame)
            box = bbox_array(prompt, frame.shape)
            value = self._logit(us_canonical(frame, box, self.margin))
            note = f"região: caixa do usuário (+{self.margin:.0%}), quadro {prompt.frame}"
        else:
            frames = _frames_for(image, self.max_frames)
            logits = [self._logit(us_canonical(frame_uint8(image, f), None, 0.0)) for f in frames]
            value = sum(logits) / len(logits)
            note = f"imagem inteira; média de {len(frames)} quadro(s)"
        return AdapterOutput(
            scores=[Score(label=label, value=value, kind="logit")], warnings=[note]
        )
