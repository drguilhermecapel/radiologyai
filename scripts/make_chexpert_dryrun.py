#!/usr/bin/env python3
"""Gera um CheXpert **sintético** para ensaiar o notebook 02 a seco.

Por que isto existe: o CheXpert real está atrás do *Research Use Agreement* da
Stanford, que proíbe redistribuição. Ninguém pode te mandar o ``valid.csv`` — ele
vem com o download, no seu nome. Mas a espera pelo aceite não precisa bloquear a
verificação de que o notebook roda na sua máquina, com o seu Python, até o fim.

O que este script produz tem o **esquema** exato do CheXpert (as 14 observações,
``Frontal/Lateral``, ``AP/PA``, ``Sex``, ``Age``, o prefixo da coluna ``Path``) e
**nenhum dado**: as imagens são ruído pseudoaleatório e os rótulos são sorteados
com as prevalências aproximadas do conjunto real.

Consequência que precisa ficar explícita: o AUROC que sai daqui fica perto de
0,5 porque é isso que ruído produz. **Não é uma medição**, e ver 0,5 é o sinal
de que a montagem está certa — um número alto aqui indicaria vazamento. Para que
nenhum artefato gerado a partir daqui possa ser confundido com medição, o script
grava um marcador ``SINTETICO.md`` que faz o manifest, e portanto o
``metrics.json``, se declararem sintéticos sozinhos.

Uso:
    python scripts/make_chexpert_dryrun.py --out /tmp/chexpert_dryrun
    # depois, no notebook 02, célula 1.1:
    CAMINHO_LOCAL = '/tmp/chexpert_dryrun/CheXpert-v1.0-small'
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from radiologyai.data.chexpert import (  # noqa: E402
    CHEXPERT_LABELS,
    DRY_RUN_MARKER,
)

CAMPOS = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA", *CHEXPERT_LABELS]

# Prevalências aproximadas do conjunto de validação real, para que a conferência
# da célula 1.2 exercite tanto o caminho "suporte suficiente" quanto o
# "suporte insuficiente". Não são medições — são parâmetros do gerador.
PREVALENCIAS: tuple[tuple[str, float], ...] = (
    ("Enlarged Cardiomediastinum", 0.45),
    ("Cardiomegaly", 0.30),
    ("Lung Opacity", 0.45),
    ("Lung Lesion", 0.04),
    ("Edema", 0.25),
    ("Consolidation", 0.14),
    ("Pneumonia", 0.04),
    ("Atelectasis", 0.33),
    ("Pneumothorax", 0.04),
    ("Pleural Effusion", 0.30),
    ("Fracture", 0.03),
)

CARD = """# Conjunto SINTÉTICO — não é o CheXpert

Este diretório foi **gerado** por `scripts/make_chexpert_dryrun.py`.

- As imagens são ruído pseudoaleatório, não radiografias.
- Os rótulos são sorteados; não descrevem as imagens.
- Nenhum dado do CheXpert real foi usado, copiado ou derivado aqui.

## O que ele serve para fazer

Verificar que o notebook `02_chexpert_e_credenciamento.ipynb` executa de ponta a
ponta — ambiente, pesos, carregador, inferência, artefato — antes de o
*Research Use Agreement* da Stanford ser aceito.

## O que ele não serve para fazer

Produzir qualquer número. O AUROC que sai daqui fica perto de 0,5 porque é isso
que ruído produz contra rótulo sorteado. Um valor alto indicaria vazamento no
pipeline, não desempenho.

A presença do arquivo `SINTETICO.md` faz o manifest se chamar
`CheXpert-SINTETICO (ensaio a seco — NÃO é medição)`, e esse nome entra no
`metrics.json`. Ainda assim: **não commite artefatos gerados a partir daqui.**

## Onde obter o dado real

https://aimi.stanford.edu/datasets/chexpert-chest-x-rays — registro, aceite do
Research Use Agreement, e o portal entrega um link individual em seu nome. O
`valid.csv` vem no download; ele não é redistribuível por terceiros.
"""


def gerar(destino: Path, *, n_pacientes: int, seed: int) -> Path:
    """Escreve o conjunto sintético e devolve a raiz que o notebook deve receber.

    Raises:
        RuntimeError: Pillow ou numpy ausentes (necessários para as imagens).
    """
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - ambiente sem o extra imaging
        raise RuntimeError(
            "este gerador precisa de numpy e Pillow: pip install 'radiologyai[imaging]'"
        ) from exc

    raiz = destino / "CheXpert-v1.0-small"
    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / DRY_RUN_MARKER).write_text(CARD, encoding="utf-8")

    rng = random.Random(seed)
    npr = np.random.default_rng(seed)
    linhas: list[dict[str, str]] = []

    for i in range(n_pacientes):
        pid = f"patient{64541 + i}"
        # ~17% dos pacientes com dois estudos, como no conjunto real.
        for estudo in range(2 if i % 6 == 0 else 1):
            lateral = rng.random() < 0.12
            vista = "lateral" if lateral else "frontal"
            rel = f"CheXpert-v1.0-small/valid/{pid}/study{estudo + 1}/view1_{vista}.jpg"
            imagem = raiz / rel.split("/", 1)[1]
            imagem.parent.mkdir(parents=True, exist_ok=True)
            ruido = npr.integers(0, 256, (390, 320), dtype=np.uint8)
            Image.fromarray(ruido, mode="L").save(imagem, quality=90)

            rotulos = dict.fromkeys(CHEXPERT_LABELS, "")
            for nome, prevalencia in PREVALENCIAS:
                rotulos[nome] = "1.0" if rng.random() < prevalencia else "0.0"
            if rng.random() < 0.05:
                rotulos["Consolidation"] = "-1.0"  # incerto: o carregador exclui

            linhas.append(
                {
                    "Path": rel,
                    "Sex": rng.choice(["Male", "Female"]),
                    "Age": str(rng.randint(20, 89)),
                    "Frontal/Lateral": "Lateral" if lateral else "Frontal",
                    "AP/PA": "" if lateral else rng.choice(["AP", "PA"]),
                    **rotulos,
                }
            )

    with (raiz / "valid.csv").open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(linhas)

    frontais = sum(1 for linha in linhas if linha["Frontal/Lateral"] == "Frontal")
    print(f"{len(linhas)} linhas ({frontais} frontais) de {n_pacientes} pacientes")
    print(f"raiz: {raiz}")
    print(f"csv:  {raiz / 'valid.csv'}")
    print()
    print("No notebook 02, célula 1.1:")
    print(f"    CAMINHO_LOCAL = {str(raiz)!r}")
    print()
    print("AUROC esperado: ~0,5. É ruído. Não é medição, e não deve ser commitado.")
    return raiz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="diretório de saída")
    parser.add_argument("--pacientes", type=int, default=200, help="nº de pacientes")
    parser.add_argument("--seed", type=int, default=20260101, help="semente")
    args = parser.parse_args()
    gerar(args.out, n_pacientes=args.pacientes, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
