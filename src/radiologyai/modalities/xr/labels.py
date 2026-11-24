"""Alinhamento de rótulos entre NIH ChestX-ray14 e torchxrayvision.

O legado colapsava as 18 saídas do torchxrayvision em 5 categorias, de forma
clinicamente perigosa (HONEST_STATUS.md §4): pneumotórax virava pneumonia,
cardiomegalia e edema viravam "normal". Aqui não há colapso: cada achado é
reportado nativamente, e o que não tem contrapartida no conjunto de referência
é marcado como não avaliado — nunca descartado em silêncio.
"""

from __future__ import annotations

# As 14 classes do NIH ChestX-ray14 (Data_Entry_2017).
NIH_CXR14_LABELS: tuple[str, ...] = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
)

# Ordem canônica das patologias do torchxrayvision
# (``xrv.datasets.default_pathologies``).
XRV_PATHOLOGIES: tuple[str, ...] = (
    "Atelectasis",
    "Consolidation",
    "Infiltration",
    "Pneumothorax",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Effusion",
    "Pneumonia",
    "Pleural_Thickening",
    "Cardiomegaly",
    "Nodule",
    "Mass",
    "Hernia",
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)

# Saídas do xrv sem contrapartida direta no NIH ChestX-ray14.
# São reportadas com ``evaluated: false`` no relatório de avaliação.
XRV_NOT_IN_NIH: tuple[str, ...] = (
    "Lung Lesion",
    "Fracture",
    "Lung Opacity",
    "Enlarged Cardiomediastinum",
)

# Achados cuja comparabilidade entre datasets é conhecidamente fraca.
# Declarados em ``limitations[]`` de todo metrics.json que os inclua.
WEAKLY_COMPARABLE: tuple[str, ...] = ("Infiltration",)


def xrv_to_nih_indices() -> dict[str, int]:
    """Mapa ``rótulo NIH -> índice na saída do torchxrayvision``.

    Raises:
        RuntimeError: se algum rótulo do NIH não existir em XRV_PATHOLOGIES.
            Isso indicaria mudança de versão do torchxrayvision e deve quebrar
            o build, nunca degradar silenciosamente.
    """
    index = {p: i for i, p in enumerate(XRV_PATHOLOGIES)}
    missing = [label for label in NIH_CXR14_LABELS if label not in index]
    if missing:
        raise RuntimeError(
            "rótulos do NIH ausentes nas patologias do torchxrayvision: "
            f"{missing}. Verifique a versão de torchxrayvision."
        )
    return {label: index[label] for label in NIH_CXR14_LABELS}
