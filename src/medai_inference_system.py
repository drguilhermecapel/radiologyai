"""Versão simplificada do sistema de inferência usada nos testes.
As implementações aqui não dependem de bibliotecas de IA externas."""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import logging

logger = logging.getLogger("MedAI.Inference")

class InferenceEngine:
    """Engine de inferência fictício para os testes."""

    def __init__(self) -> None:
        self.models: Dict[str, Dict] = {}
        self.initialized = True
        logger.info("InferenceEngine inicializado (stub)")

    def load_model(self, model_name: str) -> bool:
        """Carrega um modelo simbólico."""
        self.models[model_name] = {"name": model_name}
        logger.info("Modelo %s carregado", model_name)
        return True

    def predict(self, image_data, model_name: str = "default") -> Dict[str, float]:
        """Gera uma predição aleatória para a imagem fornecida."""
        confidence = float(np.random.uniform(0.85, 0.98))
        return {
            "class": "normal" if confidence > 0.9 else "pneumonia",
            "confidence": confidence,
            "processing_time": float(np.random.uniform(0.5, 2.0)),
        }

@dataclass
class PredictionResult:
    image_path: str
    predictions: Dict[str, float]
    predicted_class: str
    confidence: float
    processing_time: float
    heatmap: Optional[np.ndarray] = None
    attention_map: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
