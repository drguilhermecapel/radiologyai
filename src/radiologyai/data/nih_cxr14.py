"""NIH ChestX-ray14 — construção do manifest a partir dos arquivos oficiais.

Substitui o stub de ``legacy/src/medai_dataset_loaders.py`` ("NIH ChestX-ray14
loader not fully implemented").

Fonte: 112.120 radiografias frontais de 30.805 pacientes, do NIH Clinical Center.
Requer ``Data_Entry_2017_v2020.csv`` e ``test_list.txt`` (o split oficial, que já
é disjunto por paciente).

Os rótulos são **minerados por NLP** dos laudos, não adjudicados por radiologista.
Essa limitação é propagada para todo relatório de avaliação.
"""

from __future__ import annotations

import csv
from pathlib import Path

from radiologyai.data.manifest import Manifest, ManifestRow
from radiologyai.errors import EvaluationError
from radiologyai.modalities.xr.labels import NIH_CXR14_LABELS

DATA_ENTRY_CSV = "Data_Entry_2017_v2020.csv"
TEST_LIST = "test_list.txt"
TRAIN_VAL_LIST = "train_val_list.txt"

NO_FINDING = "No Finding"

LIMITATIONS: tuple[str, ...] = (
    "Rótulos minerados por NLP dos laudos, não adjudicados por radiologista; "
    "a acurácia estimada da rotulagem é de aproximadamente 90%.",
    "Apenas incidências frontais (PA e AP). Perfil não está representado.",
    "Sem sujeitos pediátricos identificados separadamente.",
    "'Infiltration' do NIH não é diretamente comparável ao rótulo homônimo de "
    "outros datasets; a concordância entre definições é fraca.",
    "Esta é uma medição retrospectiva de desempenho de algoritmo isolado, "
    "NÃO uma validação clínica.",
)


def _parse_age(value: str) -> float | None:
    """Interpreta a coluna Patient Age do NIH.

    O dataset teve dois formatos: ``058Y`` (release de 2017) e ``58`` (v2020).
    Ambos são aceitos. Idades implausíveis são descartadas: a fonte contém
    valores acima de 400 por erro de digitação no registro original.
    """
    text = (value or "").strip().upper()
    if not text:
        return None

    factor = 1.0
    if text[-1] in "YMWD":
        factor = {"Y": 1.0, "M": 1 / 12, "W": 1 / 52, "D": 1 / 365.25}[text[-1]]
        text = text[:-1]

    try:
        age = float(text) * factor
    except ValueError:
        return None
    return age if 0 < age < 120 else None


def build_manifest(data_root: str | Path, *, split: str = "test") -> Manifest:
    """Constrói o manifest do split oficial pedido.

    Args:
        data_root: diretório com ``Data_Entry_2017_v2020.csv``, ``test_list.txt``
            e ``train_val_list.txt``.
        split: ``"test"``, ``"train_val"`` ou ``"all"``.

    Raises:
        EvaluationError: arquivos oficiais ausentes ou split desconhecido.
    """
    root = Path(data_root)
    entry_csv = root / DATA_ENTRY_CSV
    if not entry_csv.is_file():
        raise EvaluationError(
            f"{DATA_ENTRY_CSV} não encontrado em {root}. "
            "Baixe o dataset oficial do NIH ChestX-ray14."
        )

    selected: set[str] | None = None
    if split in ("test", "train_val"):
        list_file = root / (TEST_LIST if split == "test" else TRAIN_VAL_LIST)
        if not list_file.is_file():
            raise EvaluationError(
                f"{list_file.name} não encontrado em {root}. "
                "O split oficial é necessário: ele é disjunto por paciente."
            )
        selected = {
            line.strip()
            for line in list_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    elif split != "all":
        raise EvaluationError(f"split desconhecido {split!r}; use test, train_val ou all")

    rows: list[ManifestRow] = []
    with entry_csv.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            image_id = record["Image Index"]
            if selected is not None and image_id not in selected:
                continue

            findings = {f.strip() for f in record["Finding Labels"].split("|")}
            labels = {name: int(name in findings) for name in NIH_CXR14_LABELS}

            rows.append(
                ManifestRow(
                    image_id=image_id,
                    patient_id=record["Patient ID"],
                    labels=labels,
                    view_position=(record.get("View Position") or "").strip().upper() or None,
                    patient_sex=(record.get("Patient Gender") or "").strip().upper() or None,
                    patient_age_years=_parse_age(record.get("Patient Age", "")),
                )
            )

    if not rows:
        raise EvaluationError(f"nenhuma imagem encontrada para o split {split!r}")

    return Manifest(
        name="NIH ChestX-ray14",
        split=f"oficial {split}_list.txt" if selected is not None else "completo",
        label_names=NIH_CXR14_LABELS,
        rows=rows,
    )


def find_image(image_id: str, image_dirs: list[Path]) -> Path:
    """Localiza um arquivo de imagem entre os diretórios ``images_0xx/images/``.

    Raises:
        EvaluationError: imagem não encontrada.
    """
    for directory in image_dirs:
        candidate = directory / image_id
        if candidate.is_file():
            return candidate
    raise EvaluationError(f"imagem {image_id!r} não encontrada em {len(image_dirs)} diretório(s)")


def discover_image_dirs(data_root: str | Path) -> list[Path]:
    """Encontra os diretórios de imagem da distribuição oficial (12 arquivos tar)."""
    root = Path(data_root)
    dirs = [p for p in sorted(root.glob("images_*/images")) if p.is_dir()]
    if not dirs:
        for candidate in (root / "images", root):
            if candidate.is_dir() and any(candidate.glob("*.png")):
                dirs = [candidate]
                break
    return dirs
