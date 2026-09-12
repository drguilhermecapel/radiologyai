"""CheXpert — conjunto de validação com rótulos adjudicados por radiologistas.

Por que este dataset importa aqui: a primeira linha de base mediu AUROC 0,664
no NIH ChestX-ray14, cujos rótulos são **minerados por NLP dos laudos**. Um
número baixo nessas condições é ambíguo — pode ser modelo fraco, pode ser
rótulo ruidoso. Fibrose saiu em 0,448 com IC95 excluindo 0,5, o que sugere
fortemente que o rótulo "Fibrosis" do NIH e o do PadChest não nomeiam o mesmo
achado.

O conjunto de validação do CheXpert tem 234 imagens de 200 pacientes, rotuladas
pelo **voto majoritário de 3 radiologistas certificados** olhando a imagem — não
o laudo. É pequeno, então os intervalos de confiança são largos; mas é a
diferença entre medir contra ruído e medir contra consenso.

Acesso: registro na Stanford AIMI e aceite do Research Use Agreement. **Não
exige credenciamento PhysioNet.**

Referência: Irvin J, Rajpurkar P, et al. *CheXpert: A Large Chest Radiograph
Dataset with Uncertainty Labels and Expert Comparison.* AAAI 2019.
"""

from __future__ import annotations

import csv
from pathlib import Path

from radiologyai.data.manifest import Manifest, ManifestRow
from radiologyai.errors import EvaluationError

VALID_CSV_NAMES: tuple[str, ...] = ("valid.csv", "CheXpert-v1.0/valid.csv")

# As 14 observações do CheXpert, na ordem do CSV oficial.
CHEXPERT_LABELS: tuple[str, ...] = (
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
)

# CheXpert -> nomenclatura do torchxrayvision. Só entram os pares em que as duas
# definições descrevem o mesmo achado; o resto é declarado como não avaliado em
# vez de forçado a um vizinho aproximado, que foi o erro do sistema legado
# (ele mapeava pneumotórax para pneumonia).
CHEXPERT_TO_XRV: dict[str, str] = {
    "Cardiomegaly": "Cardiomegaly",
    "Edema": "Edema",
    "Consolidation": "Consolidation",
    "Pneumonia": "Pneumonia",
    "Atelectasis": "Atelectasis",
    "Pneumothorax": "Pneumothorax",
    "Pleural Effusion": "Effusion",
    "Lung Lesion": "Lung Lesion",
    "Lung Opacity": "Lung Opacity",
    "Enlarged Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Fracture": "Fracture",
}

# Sem contrapartida em nenhum modelo do torchxrayvision.
NOT_MAPPED: tuple[str, ...] = ("No Finding", "Pleural Other", "Support Devices")

LIMITATIONS: tuple[str, ...] = (
    "Conjunto de validação do CheXpert: 234 imagens de 200 pacientes. "
    "O tamanho pequeno produz intervalos de confiança largos; achados de baixa "
    "prevalência podem não ter suporte para avaliação.",
    "Rótulos por voto majoritário de 3 radiologistas certificados olhando a "
    "IMAGEM — não minerados de laudo. É esta a diferença em relação ao NIH.",
    "Uma única instituição (Stanford Hospital), população dos EUA.",
    "Inclui incidências frontais e laterais; o uso pretendido declarado cobre "
    "apenas frontais, então laterais devem ser filtradas antes da avaliação.",
    "Esta é uma medição retrospectiva de desempenho de algoritmo isolado, "
    "NÃO uma validação clínica.",
)


def find_valid_csv(root: str | Path) -> Path:
    """Localiza o ``valid.csv``, aceitando o layout com e sem o diretório raiz.

    Raises:
        EvaluationError: nenhum dos nomes conhecidos existe.
    """
    root = Path(root)
    for name in VALID_CSV_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    raise EvaluationError(
        f"valid.csv não encontrado em {root} (procurei {list(VALID_CSV_NAMES)}). "
        "Baixe o CheXpert em https://aimi.stanford.edu/datasets/chexpert-chest-x-rays "
        "após aceitar o Research Use Agreement."
    )


def _parse_label(value: str) -> int | None:
    """Converte a célula do CSV em rótulo binário.

    CheXpert usa ``1.0`` positivo, ``0.0`` negativo, ``-1.0`` incerto e vazio
    para não mencionado. No conjunto de validação só há 0 e 1 — os demais
    devolvem ``None`` e a imagem é excluída daquele achado, nunca contada como
    negativa por omissão.
    """
    text = (value or "").strip()
    if text in ("1", "1.0"):
        return 1
    if text in ("0", "0.0"):
        return 0
    return None


def _patient_id(path: str) -> str:
    """Extrai ``patientNNNNN`` do caminho — exigido para o split por paciente."""
    for part in Path(path).parts:
        if part.startswith("patient"):
            return part
    return path


def build_valid_manifest(data_root: str | Path, *, frontal_only: bool = True) -> Manifest:
    """Constrói o manifest do conjunto de validação.

    Args:
        data_root: diretório contendo ``valid.csv`` (com ou sem ``CheXpert-v1.0/``).
        frontal_only: mantém apenas incidências frontais. Default True porque o
            uso pretendido declarado (REG-01 §2) cobre apenas frontais.

    Raises:
        EvaluationError: CSV ausente, colunas inesperadas, ou nenhuma linha.
    """
    csv_path = find_valid_csv(data_root)

    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        campos = reader.fieldnames or []
        faltando = [c for c in CHEXPERT_LABELS if c not in campos]
        if faltando:
            raise EvaluationError(
                f"{csv_path.name} não tem as colunas esperadas do CheXpert: {faltando}"
            )

        rows: list[ManifestRow] = []
        for record in reader:
            if frontal_only and (record.get("Frontal/Lateral") or "").strip() != "Frontal":
                continue

            labels: dict[str, int] = {}
            for chexpert_name, xrv_name in CHEXPERT_TO_XRV.items():
                valor = _parse_label(record.get(chexpert_name, ""))
                if valor is not None:
                    labels[xrv_name] = valor

            idade = (record.get("Age") or "").strip()
            try:
                idade_anos = float(idade) if idade else None
            except ValueError:
                idade_anos = None

            caminho = record["Path"]
            rows.append(
                ManifestRow(
                    image_id=caminho,
                    patient_id=_patient_id(caminho),
                    labels=labels,
                    view_position=(record.get("AP/PA") or "").strip().upper() or None,
                    patient_sex=(record.get("Sex") or "").strip().upper()[:1] or None,
                    patient_age_years=idade_anos if idade_anos and 0 < idade_anos < 120 else None,
                )
            )

    if not rows:
        raise EvaluationError(f"nenhuma linha utilizável em {csv_path}")

    return Manifest(
        name="CheXpert",
        split="validação oficial (rótulos por consenso de 3 radiologistas)",
        label_names=tuple(sorted(set(CHEXPERT_TO_XRV.values()))),
        rows=rows,
    )


def resolve_image_path(image_id: str, data_root: str | Path) -> Path:
    """Resolve o caminho da imagem, tolerando o prefixo ``CheXpert-v1.0/``."""
    root = Path(data_root)
    candidatos = [root / image_id]
    if image_id.startswith("CheXpert-v1.0/"):
        candidatos.append(root / image_id[len("CheXpert-v1.0/") :])
    else:
        candidatos.append(root / "CheXpert-v1.0" / image_id)

    for c in candidatos:
        if c.is_file():
            return c
    raise EvaluationError(f"imagem {image_id!r} não encontrada sob {root}")
