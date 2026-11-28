"""Cópia vendorizada e reduzida de ``sam2`` 1.1.0 (Apache-2.0; ver ``NOTICE``).

Só o caminho de imagem (``SAM2ImagePredictor``) e os módulos de que ``SAM2Base``
precisa para ser construído e receber o ``state_dict`` completo do checkpoint.
Sem hydra, iopath, tqdm, extensão CUDA nem preditor de vídeo. A construção do
modelo fica em :mod:`.build` (configuração embutida, lista branca de classes).

Importa torch nos submódulos: só é importado por ``load`` do adaptador de eco.
"""

from __future__ import annotations
