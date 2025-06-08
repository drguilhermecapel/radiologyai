"""
Versão simplificada dos gerenciadores de modelos SOTA usados nos testes.
Esta implementação evita dependências pesadas como TensorFlow.
"""

from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger("MedAI.SOTA")

class SOTAModelManager:
    """Gerencia modelos fictícios utilizados apenas para testes."""

    def __init__(self) -> None:
        self.available_models: Dict[str, str] = {
            "medical_vit": "Vision Transformer simplificado",
            "hybrid_cnn_transformer": "Modelo híbrido fictício",
            "ensemble_model": "Modelo ensemble fictício",
        }
        self.loaded_models: Dict[str, Any] = {}

    def get_available_models(self) -> List[str]:
        """Retorna a lista de modelos disponíveis."""
        return list(self.available_models.keys())

    def load_model(self, model_name: str, input_shape: Tuple[int, int, int] | None = None,
                   num_classes: int | None = None) -> Dict[str, Any]:
        """Carrega um modelo fictício e o armazena internamente."""
        if model_name not in self.available_models:
            raise ValueError(f"Modelo {model_name} não disponível")
        model_info = {
            "name": model_name,
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
        self.loaded_models[model_name] = model_info
        logger.info("Modelo %s carregado (stub)", model_name)
        return model_info

    def get_model(self, model_name: str) -> Any:
        """Obtém um modelo previamente carregado."""
        return self.loaded_models.get(model_name)

class StateOfTheArtModels:
    """Classe fictícia usada apenas para inicialização em testes."""

    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int) -> None:
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_medical_vision_transformer(self):
        logger.info("build_medical_vision_transformer chamado (stub)")
        return {"model": "medical_vit", "input_shape": self.input_shape}
