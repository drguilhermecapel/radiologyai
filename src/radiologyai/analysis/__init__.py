"""Análise multimodelo de exames: contratos, registro de modelos e pipeline.

Este subpacote é o núcleo do produto multimodal (ROADMAP §4): um exame pode
passar por vários modelos reais, com dependências entre si, que produzem
escores, segmentações, medidas e explicações. A decisão (positivo, negativo,
indeterminado) é tomada SOMENTE pelo pipeline, a partir de pontos de operação
com origem declarada, e o nível de evidência vem do catálogo de condições.

Tudo o que sai daqui é *achado sugerido para confirmação médica* — nunca
diagnóstico. Software de pesquisa; não é dispositivo médico.

Importar este pacote é leve: nada de numpy, torch, pydicom ou yaml no topo.
Os módulos pesados são importados sob demanda pelos submódulos.

Módulos:

* :mod:`radiologyai.analysis.types` — ImageData, Finding, AnalysisResult ...
* :mod:`radiologyai.analysis.manifest` — esquema dos manifestos YAML.
* :mod:`radiologyai.analysis.base` — ``ModelAdapter`` e tipos de saída.
* :mod:`radiologyai.analysis.registry` — manifestos, pesos, integridade, LRU.
* :mod:`radiologyai.analysis.pipeline` — ``analyze``.
* :mod:`radiologyai.analysis.routing` — tipo de exame e roteamento.
* :mod:`radiologyai.analysis.artifacts` — repositório endereçado por conteúdo.
* :mod:`radiologyai.analysis.imaging` — DICOM/série/NIfTI/raster/cine → ImageData.
* :mod:`radiologyai.analysis.policy` — interface de calibração e pontos ajustados.
"""

from __future__ import annotations

__all__: list[str] = []
