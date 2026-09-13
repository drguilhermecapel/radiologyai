"""Model card com integridade verificável.

Substitui ``models/model_registry.json`` do legado, que declarava
``accuracy: 0.92``, ``auc: 0.94`` e ``sha256_hash`` para arquivos que nunca
existiram — e cujo "sha256" continha as letras ``g``, ``h``, ``i``, portanto
não era sequer hexadecimal (HONEST_STATUS.md).

Duas regras estruturais impedem a repetição disso:

1. ``weights_sha256`` é validado como 64 caracteres hex minúsculos. Um
   placeholder é rejeitado na carga do card, não em produção.
2. Não existe campo de desempenho no card. Métrica vive em
   ``artifacts/eval/<run_id>/metrics.json``, produzida por execução medida, e o
   card apenas **aponta** para ela via ``evaluation_runs``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from radiologyai.errors import ModelCardError, WeightsIntegrityError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

Backend = Literal["torch", "onnx"]


class ModelCard(BaseModel):
    """Descrição versionada e verificável de um modelo.

    Deliberadamente **não** possui campos ``accuracy``/``auc``/``sensitivity``.
    Alegar desempenho é papel de um artefato de avaliação, não de um manifesto.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    card_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    display_name: str
    version: str
    modality: str = Field(description="Código do plugin de modalidade, ex.: 'XR'")
    backend: Backend
    task: Literal["multilabel-classification", "classification", "segmentation"]

    labels: tuple[str, ...] = Field(min_length=1)
    input_shape: tuple[int, ...] = Field(min_length=2)

    weights_uri: str = Field(description="URI dos pesos (file://, https://, hf://)")
    weights_sha256: str
    weights_size_bytes: int | None = Field(default=None, ge=1)

    trained_on: tuple[str, ...] = Field(
        min_length=1, description="Datasets de treino, para checagem de vazamento"
    )
    license: str
    citation: str | None = None

    evaluation_runs: tuple[str, ...] = Field(
        default=(),
        description="run_ids em artifacts/eval/ que medem este modelo. "
        "Vazio significa: nenhum desempenho medido, e nenhum pode ser alegado.",
    )
    intended_use_ref: str = Field(
        default="docs/regulatory/01-intended-use.md",
        description="Documento que define o uso pretendido aplicável",
    )
    notes: str | None = None

    @field_validator("weights_sha256")
    @classmethod
    def _validate_sha256(cls, v: str) -> str:
        if not SHA256_RE.match(v):
            raise ValueError(
                f"weights_sha256 deve ser 64 caracteres hexadecimais minúsculos; "
                f"recebido {v!r}. Placeholders são rejeitados por projeto."
            )
        return v

    @field_validator("labels")
    @classmethod
    def _validate_labels_unique(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(v)) != len(v):
            raise ValueError("labels contém duplicatas")
        return v

    @property
    def has_measured_performance(self) -> bool:
        """True somente quando existe ao menos uma execução de avaliação vinculada."""
        return bool(self.evaluation_runs)

    def verify_weights(self, path: str | Path, *, chunk_size: int = 1 << 20) -> None:
        """Verifica o sha256 do arquivo de pesos. Falha fechada (perigo H-04).

        Lê em blocos — o legado fazia MD5 do arquivo inteiro em memória.

        Raises:
            WeightsIntegrityError: arquivo ausente ou hash divergente.
        """
        p = Path(path)
        if not p.is_file():
            raise WeightsIntegrityError(f"arquivo de pesos não encontrado: {p}")

        digest = hashlib.sha256()
        with p.open("rb") as fh:
            while chunk := fh.read(chunk_size):
                digest.update(chunk)

        actual = digest.hexdigest()
        if actual != self.weights_sha256:
            raise WeightsIntegrityError(
                f"integridade dos pesos falhou para {self.card_id}: "
                f"esperado {self.weights_sha256}, encontrado {actual}. "
                "Nenhuma inferência será executada."
            )

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelCard:
        """Carrega um card de arquivo YAML."""
        import yaml

        p = Path(path)
        try:
            data: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ModelCardError(f"model card não encontrado: {p}") from exc
        except yaml.YAMLError as exc:
            raise ModelCardError(f"YAML inválido em {p}: {exc}") from exc

        if not isinstance(data, dict):
            raise ModelCardError(f"model card deve ser um mapeamento: {p}")

        try:
            return cls.model_validate(data)
        except Exception as exc:  # noqa: BLE001 - reembalado com o caminho
            raise ModelCardError(f"model card inválido em {p}: {exc}") from exc
