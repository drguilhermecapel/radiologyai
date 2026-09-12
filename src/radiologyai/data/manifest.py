"""Manifest de dataset — o registro de reprodutibilidade.

Dados de pixel nunca são commitados. O manifest é: quais imagens, de qual
paciente, com quais rótulos, e o sha256 de cada arquivo. Custa poucos megabytes
e é o que permite a um terceiro reproduzir uma avaliação exatamente.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from radiologyai.errors import EvaluationError

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True)
class ManifestRow:
    """Uma imagem do dataset."""

    image_id: str
    patient_id: str
    labels: dict[str, int]
    view_position: str | None = None
    patient_sex: str | None = None
    patient_age_years: float | None = None
    sha256: str | None = None

    def age_band(self) -> str:
        """Faixa etária para análise de subgrupos."""
        age = self.patient_age_years
        if age is None:
            return "desconhecida"
        if age < 40:
            return "18-39"
        if age < 60:
            return "40-59"
        if age < 75:
            return "60-74"
        return "75+"


@dataclass
class Manifest:
    """Coleção de :class:`ManifestRow` com os rótulos declarados."""

    name: str
    split: str
    label_names: tuple[str, ...]
    rows: list[ManifestRow] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self) -> Iterator[ManifestRow]:
        return iter(self.rows)

    @property
    def patient_ids(self) -> set[str]:
        return {r.patient_id for r in self.rows}

    def label_matrix(self) -> list[list[int]]:
        """Matriz ``n_imagens x n_rótulos``, na ordem de :attr:`label_names`."""
        return [[row.labels.get(name, 0) for name in self.label_names] for row in self.rows]

    def positives(self, label: str) -> int:
        return sum(row.labels.get(label, 0) for row in self.rows)

    def to_csv(self, path: str | Path) -> Path:
        """Grava o manifest. Este arquivo é commitado; os pixels não."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "image_id",
            "patient_id",
            "view_position",
            "patient_sex",
            "patient_age_years",
            "sha256",
            *self.label_names,
        ]
        with p.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for row in self.rows:
                writer.writerow(
                    [
                        row.image_id,
                        row.patient_id,
                        row.view_position or "",
                        row.patient_sex or "",
                        "" if row.patient_age_years is None else f"{row.patient_age_years:.1f}",
                        row.sha256 or "",
                        *[row.labels.get(name, 0) for name in self.label_names],
                    ]
                )
        return p

    @classmethod
    def from_csv(cls, path: str | Path, *, name: str, split: str) -> Manifest:
        """Lê um manifest de disco."""
        p = Path(path)
        if not p.is_file():
            raise EvaluationError(f"manifest não encontrado: {p}")

        with p.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fixed = {
                "image_id",
                "patient_id",
                "view_position",
                "patient_sex",
                "patient_age_years",
                "sha256",
            }
            label_names = tuple(c for c in (reader.fieldnames or []) if c not in fixed)
            rows = [
                ManifestRow(
                    image_id=r["image_id"],
                    patient_id=r["patient_id"],
                    labels={name: int(r[name]) for name in label_names},
                    view_position=r.get("view_position") or None,
                    patient_sex=r.get("patient_sex") or None,
                    patient_age_years=(
                        float(r["patient_age_years"]) if r.get("patient_age_years") else None
                    ),
                    sha256=r.get("sha256") or None,
                )
                for r in reader
            ]
        return cls(name=name, split=split, label_names=label_names, rows=rows)

    def sha256(self) -> str:
        """Hash do conteúdo do manifest — gravado em todo artefato de avaliação.

        Independente da ordem das linhas: as linhas são hasheadas ordenadas por
        ``image_id``. O CSV oficial do NIH e o espelho do Kaggle trazem as
        mesmas 25.596 linhas em ordens diferentes; um hash sensível à ordem
        fazia dois manifests de conteúdo idêntico parecerem dados diferentes.
        """
        digest = hashlib.sha256()
        for row in sorted(self.rows, key=lambda r: r.image_id):
            digest.update(row.image_id.encode())
            digest.update(row.patient_id.encode())
            for name in self.label_names:
                digest.update(str(row.labels.get(name, 0)).encode())
        return digest.hexdigest()
