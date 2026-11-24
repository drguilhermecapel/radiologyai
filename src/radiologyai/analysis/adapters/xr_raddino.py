"""Sonda linear sobre *embeddings* congelados do RAD-DINO (``microsoft/rad-dino``).

Codificador escolhido por medição (``scripts/probe_fit_xr_compare_encoders.py``):
com a mesma receita de sonda, o RAD-DINO transferiu entre datasets muito melhor
que a DenseNet do TorchXRayVision. Licença MIT, mas o cartão restringe a
pesquisa ("shared for research purposes only") — ``commercial_use: no``.

Pré-processamento (fixo; reproduz o usado para ajustar as sondas):

1. imagem → 8 bits pela faixa REPRESENTÁVEL (``value_range``), arredondando;
2. quadrado: recorte central (``square: crop``, RX de tórax) ou preenchimento
   com zeros centralizado (``square: pad``, RX musculoesquelético);
3. redimensionamento PIL bicúbico para 518x518, RGB;
4. ``/255``, normalização (0,5307; 0,2583) do ``preprocessor_config`` do modelo;
5. ``pooler_output`` do ``Dinov2Model`` (CLS após layernorm, 768).

Pesos (chaves fixas no manifesto): ``encoder_weights`` (``model.safetensors``),
``encoder_config`` (``config.json``), ``probe_tensors``, ``probe_meta``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.xr_common import read_verified, require_backend, to_gray
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
    from radiologyai.analysis.probes.artifact import ProbeArtifact

__all__ = [
    "RADDINO_ENCODER_ID",
    "RadDinoLinearProbeAdapter",
    "build_raddino",
    "raddino_canonical",
    "raddino_embeddings",
    "raddino_tensor",
]

RADDINO_ENCODER_ID = "microsoft/rad-dino@110cbc18d5133582e320b43d53bf5c44e410c936"
_SIZE = 518
_MEAN = 0.5307
_STD = 0.2583
_WEIGHT_KEYS = {"encoder_weights", "encoder_config", "probe_tensors", "probe_meta"}


def raddino_canonical(image: ImageData, square: str) -> npt.NDArray[np.uint8]:
    """Imagem → ``uint8`` 518x518 (passos 1-3 do pré-processamento; um canal).

    Raises:
        UnsupportedInputError: pixels não finitos ou fora da faixa representável.
        ValueError: ``square`` diferente de ``crop``/``pad``.
    """
    import numpy as np
    from PIL import Image

    plane = to_gray(image)
    lo, hi = image.value_range
    if not np.isfinite(plane).all():
        raise UnsupportedInputError("imagem com pixels não finitos")
    scaled = (plane - np.float32(lo)) / np.float32(hi - lo) * np.float32(255.0)
    if scaled.min() < -0.5 or scaled.max() > 255.5:
        raise UnsupportedInputError("pixels fora da faixa representável declarada")
    pil = Image.fromarray(np.clip(np.rint(scaled), 0, 255).astype(np.uint8), mode="L")
    w, h = pil.size
    if square == "crop":
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        pil = pil.crop((left, top, left + s, top + s))
    elif square == "pad":
        s = max(w, h)
        canvas = Image.new("L", (s, s), 0)
        canvas.paste(pil, ((s - w) // 2, (s - h) // 2))
        pil = canvas
    else:
        raise ValueError(f"square deve ser 'crop' ou 'pad'; recebido {square!r}")
    out: npt.NDArray[np.uint8] = np.asarray(
        pil.resize((_SIZE, _SIZE), Image.Resampling.BICUBIC), dtype=np.uint8
    )
    return out


def raddino_tensor(canonical: npt.NDArray[np.uint8]) -> torch.Tensor:
    """``uint8`` 518x518 (ou lote ``[N, 518, 518]``) → tensor normalizado ``[N, 3, 518, 518]``."""
    import numpy as np
    import torch

    arr = np.asarray(canonical, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None]
    x = (arr / np.float32(255.0) - np.float32(_MEAN)) / np.float32(_STD)
    return torch.from_numpy(np.repeat(x[:, None], 3, axis=1).astype(np.float32))


def build_raddino(config_bytes: bytes, weights_bytes: bytes) -> torch.nn.Module:
    """``Dinov2Model`` a partir dos bytes verificados (sem acesso ao Hugging Face Hub).

    Os bytes já conferidos são gravados num diretório temporário e carregados com
    ``from_pretrained(..., local_files_only=True)``, que aplica a conversão de nomes
    de chaves da versão instalada do ``transformers`` (o checkpoint usa os nomes
    antigos ``attention.attention.query``).

    Raises:
        ManifestError: pesos não casam com a configuração.
    """
    import tempfile
    from pathlib import Path

    from transformers import Dinov2Model

    with tempfile.TemporaryDirectory(prefix="raddino-") as tmp:
        (Path(tmp) / "config.json").write_bytes(config_bytes)
        (Path(tmp) / "model.safetensors").write_bytes(weights_bytes)
        try:
            loaded: Any = Dinov2Model.from_pretrained(
                tmp, local_files_only=True, output_loading_info=True
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise ManifestError(f"pesos RAD-DINO ilegíveis: {exc}") from exc
    model, info = loaded[0], loaded[1]
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise ManifestError(
            f"pesos RAD-DINO incompatíveis: faltando {list(info.get('missing_keys', []))[:5]}, "
            f"sobrando {list(info.get('unexpected_keys', []))[:5]}"
        )
    module: torch.nn.Module = model
    module.eval()
    return module


def raddino_embeddings(model: torch.nn.Module, batch: torch.Tensor) -> npt.NDArray[np.float32]:
    """``pooler_output`` (768) para um lote ``[N, 3, 518, 518]``."""
    import torch

    with torch.inference_mode():
        out = model(pixel_values=batch).pooler_output
    values: npt.NDArray[np.float32] = out.float().cpu().numpy()
    return values


class RadDinoLinearProbeAdapter(ModelAdapter):
    """RAD-DINO congelado + sonda linear fixada; escore logit."""

    def __init__(self, manifest: ModelManifest) -> None:
        super().__init__(manifest)
        if len(manifest.labels) != 1:
            raise ManifestError(f"{manifest.id}: uma sonda emite exatamente um rótulo")
        keys = {w.key for w in manifest.weights}
        if keys != _WEIGHT_KEYS:
            raise ManifestError(
                f"{manifest.id}: pesos devem ser {sorted(_WEIGHT_KEYS)}; recebido {sorted(keys)}"
            )
        cfg = manifest.config
        self.square = str(cfg.get("square", ""))
        if self.square not in ("crop", "pad"):
            raise ManifestError(f"{manifest.id}: config.square deve ser 'crop' ou 'pad'")
        self.gate_ids: tuple[str, ...] = tuple(cfg.get("gates", ()))
        missing = [g for g in self.gate_ids if g not in manifest.depends_on]
        if missing:
            raise ManifestError(f"{manifest.id}: portões {missing} não estão em depends_on")
        self._model: Any = None
        self._probe: ProbeArtifact | None = None

    def load(self, weights: ResolvedWeights, device: str) -> None:
        require_backend("torch", "transformers", "safetensors", "PIL", extra="foundation")
        import torch

        from radiologyai.analysis.probes.artifact import load_probe

        probe = load_probe(
            read_verified(weights, "probe_tensors"), read_verified(weights, "probe_meta")
        )
        if probe.label != self.manifest.labels[0].key:
            raise ManifestError(
                f"{self.manifest.id}: artefato é da sonda {probe.label!r}, manifesto declara "
                f"{self.manifest.labels[0].key!r}"
            )
        if probe.encoder_id != RADDINO_ENCODER_ID:
            raise ManifestError(
                f"{self.manifest.id}: sonda ajustada sobre {probe.encoder_id!r}, "
                f"adaptador usa {RADDINO_ENCODER_ID!r}"
            )
        if probe.meta.get("square") != self.square:
            raise ManifestError(f"{self.manifest.id}: sonda ajustada com outro enquadramento")
        model = build_raddino(
            read_verified(weights, "encoder_config"), read_verified(weights, "encoder_weights")
        )
        self._model = model.to(torch.device(device))
        self._probe = probe

    def unload(self) -> None:
        self._model, self._probe = None, None

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        if not base.applicable or not self.gate_ids:
            return base
        checks, reasons = gate_checks(inputs, self.gate_ids)
        all_checks = (*base.checks, *checks)
        if reasons:
            return QAReport.reject(*dict.fromkeys(reasons), checks=all_checks)
        return QAReport.ok(all_checks)

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        from radiologyai.analysis.probes.artifact import probe_logit

        if self._model is None or self._probe is None:
            raise InferenceError(f"{self.manifest.id}: modelo não carregado")
        device = next(self._model.parameters()).device
        batch = raddino_tensor(raddino_canonical(inputs.single, self.square)).to(device)
        emb = raddino_embeddings(self._model, batch)
        value = float(probe_logit(self._probe, emb)[0])
        return AdapterOutput(
            scores=[Score(label=self.manifest.labels[0].key, value=value, kind="logit")],
            extras={"image_sha256": inputs.single.sha256, "encoder": RADDINO_ENCODER_ID},
        )
