"""Explicabilidade — derivada dos gradientes reais, nunca simulada."""

from __future__ import annotations

from radiologyai.explain.gradcam import Explanation, GradCAM, overlay_heatmap

__all__ = ["Explanation", "GradCAM", "overlay_heatmap"]
