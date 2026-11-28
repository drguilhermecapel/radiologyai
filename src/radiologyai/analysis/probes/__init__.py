"""Sondas lineares sobre *embeddings* congelados de um codificador de imagem.

* :mod:`.artifact` — formato do artefato (``safetensors`` com os vetores + JSON
  com metadados, hash dos dados de treino e pontos de operação) e a aplicação
  da sonda (escala logit), só com numpy.
* :mod:`.fit` — ajuste determinístico (scikit-learn, extra ``eval``): escolha de
  ``C`` por validação cruzada interna agrupada por paciente e pontos de operação
  calculados SOMENTE nas predições fora-da-dobra do conjunto de treino.

Sondas não são modelos "prontos": o nível de evidência de cada uma vem das
medições registradas pela fábrica de evidência, nunca deste pacote.
"""

from __future__ import annotations

__all__ = ["artifact", "fit"]
