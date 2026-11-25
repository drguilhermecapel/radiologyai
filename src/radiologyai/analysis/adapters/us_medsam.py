"""MedSAM (``wanglab/medsam-vit-base``): segmentação interativa por caixa em US.

O usuário desenha uma caixa (``Primitive`` bbox; ``frame`` escolhe o quadro de
um clipe) e o MedSAM devolve a máscara da estrutura/lesão dentro dela. Daí
saem área, maior diâmetro (Feret) e o maior diâmetro perpendicular a ele —
em mm SOMENTE com espaçamento físico (``SequenceOfUltrasoundRegions``);
sem ele as medidas se abstêm por geometria e só a máscara é devolvida.

Pré-processamento do MedSAM original (bowang-lab/MedSAM, ``MedSAM_Inference.py``):
imagem RGB esticada para 1024 × 1024 (bicúbica), min-max para [0, 1], SEM
média/desvio do ImageNet; caixa reescalada para 1024; logits de baixa
resolução → sigmoide → bilinear para (H, W) → > 0,5. O redimensionamento usa
o bicúbico do Pillow (com antisserrilhamento) no lugar do ``skimage.transform.resize``
do original — desvio declarado.

Não classifica nada: não há achado, só máscara e medidas.
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any

from radiologyai.analysis.adapters.us_common import (
    bbox_array,
    box_prompts,
    clean_mask,
    frame_uint8,
    lesion_morphometry,
)
from radiologyai.analysis.base import (
    AdapterOutput,
    LabelMap,
    ModelAdapter,
    ModelInputs,
    RawMeasurement,
    ResolvedWeights,
)
from radiologyai.analysis.types import (
    AbstentionReason,
    ImageKind,
    Primitive,
    QACheck,
    QAReport,
)
from radiologyai.errors import BackendUnavailableError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

__all__ = ["MedSAMSegmenter", "MedSAMUSAdapter"]

INPUT_SIZE = 1024


def _require() -> None:
    for mod, extra in (("torch", "ml,cpu"), ("transformers", "foundation"), ("PIL", "imaging")):
        if importlib.util.find_spec(mod) is None:
            raise BackendUnavailableError(
                f"{mod} não instalado; instale os extras: pip install 'radiologyai[{extra}]'"
            )


class MedSAMSegmenter:
    """MedSAM ViT-B via ``transformers.SamModel`` (sem ``trust_remote_code``)."""

    def __init__(self, model_dir: Any, device: str = "cpu") -> None:
        _require()
        from transformers import SamModel

        self.device = device
        model: Any = SamModel.from_pretrained(str(model_dir), local_files_only=True)
        self.model: Any = model.to(device)
        self.model.eval()

    def embed(self, frame_u8: npt.NDArray[np.uint8]) -> Any:
        """Embedding da imagem (pré-processamento do MedSAM original)."""
        import numpy as np
        import torch
        from PIL import Image

        rgb = np.repeat(frame_u8[..., None], 3, axis=2)
        im = np.asarray(
            Image.fromarray(rgb).resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC),
            dtype=np.float32,
        )
        span = max(float(im.max() - im.min()), 1e-8)
        im = (im - float(im.min())) / span
        px = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            return self.model.get_image_embeddings(px)

    def segment(
        self, frame_u8: npt.NDArray[np.uint8], box: npt.NDArray[np.float64], embedding: Any = None
    ) -> tuple[npt.NDArray[np.bool_], float]:
        """Máscara binária (limpa) e IoU previsto pelo modelo para a caixa ``[x0, y0, x1, y1]``."""
        import torch
        import torch.nn.functional as F  # noqa: N812 - convenção do PyTorch

        h, w = frame_u8.shape
        emb = self.embed(frame_u8) if embedding is None else embedding
        scale = torch.tensor([w, h, w, h], dtype=torch.float32)
        box1024 = torch.as_tensor(box, dtype=torch.float32) / scale * INPUT_SIZE
        with torch.inference_mode():
            out = self.model(
                image_embeddings=emb,
                input_boxes=box1024[None, None, :].to(self.device),
                multimask_output=False,
            )
            logits = out.pred_masks[0, 0].float()
            prob = F.interpolate(
                torch.sigmoid(logits)[None], size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]
        mask = (prob > 0.5).cpu().numpy()
        return clean_mask(mask), float(out.iou_scores.flatten()[0])


def _spacing2(spacing: tuple[float, ...] | None) -> tuple[float, float] | None:
    if spacing is None or len(spacing) < 2:
        return None
    return float(spacing[-2]), float(spacing[-1])


class MedSAMUSAdapter(ModelAdapter):
    """Segmentação interativa (caixa) + morfometria em US. Rota interativa."""

    segmenter: MedSAMSegmenter | None = None

    def load(self, weights: ResolvedWeights, device: str) -> None:
        model_dir = weights.path("model").parent
        self.segmenter = MedSAMSegmenter(model_dir, device=device)
        self.device = device
        self.loaded = True

    def unload(self) -> None:
        self.segmenter = None
        self.loaded = False

    def applicability(self, inputs: ModelInputs) -> QAReport:
        base = super().applicability(inputs)
        checks = list(base.checks)
        reasons = list(base.reasons)
        boxes = box_prompts(inputs.prompts)
        checks.append(QACheck(name="box_prompts", passed=bool(boxes), value=float(len(boxes))))
        if not boxes:
            reasons.append(AbstentionReason.MISSING_INPUT)
        if inputs.images:
            image = inputs.images[0]
            n = int(image.pixels.shape[0]) if image.kind is ImageKind.CINE else 1
            bad = [p.frame for p in boxes if not 0 <= p.frame < n]
            checks.append(
                QACheck(name="box_frames", passed=not bad, value=str(bad) if bad else None)
            )
            if bad:
                reasons.append(AbstentionReason.INPUT)
        if reasons:
            return QAReport.reject(*dict.fromkeys(reasons), checks=checks)
        return QAReport.ok(checks)

    def predict(self, inputs: ModelInputs) -> AdapterOutput:
        import numpy as np

        if self.segmenter is None:
            raise RuntimeError("MedSAMUSAdapter.predict antes de load()")
        image = inputs.single
        boxes = box_prompts(inputs.prompts)
        spacing = _spacing2(image.spacing_mm)
        cine = image.kind is ImageKind.CINE
        n = int(image.pixels.shape[0]) if cine else 1
        labels = np.zeros((n, *image.plane_shape), dtype=np.uint8)
        measurements: list[RawMeasurement] = []
        warnings: list[str] = []
        embeddings: dict[int, Any] = {}
        for k, prompt in enumerate(boxes, start=1):
            frame = frame_uint8(image, prompt.frame)
            if prompt.frame not in embeddings:
                embeddings[prompt.frame] = self.segmenter.embed(frame)
            mask, iou = self.segmenter.segment(
                frame, bbox_array(prompt, frame.shape), embeddings[prompt.frame]
            )
            tag = f"lesion{k}"
            warnings.append(f"{tag}: IoU previsto pelo modelo {iou:.3f}")
            method = f"MedSAM (caixa do usuário {k}), quadro {prompt.frame}"
            if not mask.any():
                for key, unit in (("area", "mm2"), ("max_diameter", "mm"), ("perp_diameter", "mm")):
                    measurements.append(
                        RawMeasurement(
                            key=f"{tag}_{key}",
                            value=None,
                            unit=unit,
                            method=method,
                            abstention=AbstentionReason.QUALITY,
                            note="o modelo não segmentou estrutura na caixa",
                        )
                    )
                continue
            labels[prompt.frame][mask & (labels[prompt.frame] == 0)] = min(k, 255)
            morph = lesion_morphometry(mask, spacing)
            geo_major = [
                Primitive(type="line", xy=morph.max_diameter_line, frame=prompt.frame, label=tag)
            ]
            geo_perp = (
                [Primitive(type="line", xy=morph.perpendicular_line, frame=prompt.frame, label=tag)]
                if morph.perpendicular_line
                else []
            )
            no_mm = None if spacing else AbstentionReason.GEOMETRY
            note = None if spacing else f"sem espaçamento físico; área = {morph.area_px} pixels"
            pre: dict[str, str] = {"pixel_spacing_source": str(image.meta.pixel_spacing_source)}
            for key, value, unit, geo in (
                ("area", morph.area_mm2, "mm2", []),
                ("max_diameter", morph.max_diameter_mm, "mm", geo_major),
                ("perp_diameter", morph.perpendicular_diameter_mm, "mm", geo_perp),
            ):
                measurements.append(
                    RawMeasurement(
                        key=f"{tag}_{key}",
                        value=value,
                        unit=unit,
                        method=method
                        + (
                            "; maior diâmetro de Feret"
                            if key == "max_diameter"
                            else "; maior extensão perpendicular ao diâmetro de Feret"
                            if key == "perp_diameter"
                            else ""
                        ),
                        geometry=geo,
                        derived_from=["lesion"],
                        preconditions=pre,
                        abstention=no_mm,
                        note=note,
                    )
                )
        lm = LabelMap(
            name="lesion",
            array=labels if cine else labels[0],
            labels={k: f"lesion{k}" for k in range(1, min(len(boxes), 255) + 1)},
        )
        return AdapterOutput(labelmaps=[lm], measurements=measurements, warnings=warnings)
