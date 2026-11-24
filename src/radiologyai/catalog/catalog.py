"""Leitura do catálogo de condições e dos níveis de evidência gerados."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from radiologyai.catalog.schema import (
    Condition,
    EvidenceLevel,
    LevelEntry,
    LevelsFile,
    Protocol,
    Visibility,
    visibility_for,
)
from radiologyai.errors import CatalogError

PACKAGE_DIR = Path(__file__).parent
CONDITIONS_FILE = "conditions.yaml"
LEVELS_FILE = "levels.json"
PROTOCOL_FILE = "protocol.yaml"


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    """Lê um YAML cuja raiz é um mapeamento. Falha alto em qualquer outro caso."""
    import yaml

    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise CatalogError(f"arquivo do catálogo não encontrado: {path}") from exc
    except yaml.YAMLError as exc:
        raise CatalogError(f"YAML inválido em {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CatalogError(f"{path} deve conter um mapeamento na raiz")
    return data


def load_protocol(directory: Path = PACKAGE_DIR) -> Protocol:
    """Carrega o protocolo de evidência pré-registrado.

    Raises:
        CatalogError: arquivo ausente, YAML inválido ou campos fora do esquema.
    """
    path = directory / PROTOCOL_FILE
    data = _read_yaml_mapping(path)
    try:
        return Protocol.model_validate(data)
    except ValidationError as exc:
        raise CatalogError(f"protocolo inválido em {path}: {exc}") from exc


class Catalog:
    """Condições mais o nível de evidência de cada trio (condição, modelo, rótulo)."""

    def __init__(self, conditions: list[Condition], levels: LevelsFile | None) -> None:
        self._conditions: dict[str, Condition] = {}
        for condition in conditions:
            if condition.id in self._conditions:
                raise CatalogError(f"id de condição duplicado: {condition.id!r}")
            self._conditions[condition.id] = condition
        self.levels_file = levels
        self._levels: dict[tuple[str, str, str], LevelEntry] = {}
        for entry in levels.entries if levels else ():
            if entry.condition_id not in self._conditions:
                raise CatalogError(
                    f"níveis referenciam condição desconhecida: {entry.condition_id!r}"
                )
            key = (entry.condition_id, entry.model_id, entry.label)
            if key in self._levels:
                raise CatalogError(f"nível duplicado para o trio {key!r}")
            self._levels[key] = entry

    @classmethod
    def load(cls, directory: Path = PACKAGE_DIR) -> Catalog:
        """Carrega ``conditions.yaml`` e, se existir, o ``levels.json`` gerado.

        ``levels.json`` ausente não é erro: sem níveis gerados, toda condição é
        nível D (modo pesquisa), que é o padrão conservador. Já ``conditions.yaml``
        ausente ou malformado interrompe a carga.

        Raises:
            CatalogError: arquivo ausente, malformado ou inconsistente.
        """
        conditions_path = directory / CONDITIONS_FILE
        raw = _read_yaml_mapping(conditions_path)
        items = raw.get("conditions")
        if not isinstance(items, list):
            raise CatalogError(f"{conditions_path}: a chave 'conditions' deve ser uma lista")
        try:
            conditions = [Condition.model_validate(item) for item in items]
        except ValidationError as exc:
            raise CatalogError(f"condição inválida em {conditions_path}: {exc}") from exc

        levels_path = directory / LEVELS_FILE
        levels: LevelsFile | None = None
        if levels_path.is_file():
            try:
                levels = LevelsFile.model_validate(
                    json.loads(levels_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, ValidationError) as exc:
                raise CatalogError(f"níveis inválidos em {levels_path}: {exc}") from exc
        return cls(conditions, levels)

    # -- consultas ---------------------------------------------------------- #
    def conditions(self) -> list[Condition]:
        """Todas as condições, na ordem do arquivo."""
        return list(self._conditions.values())

    def has(self, condition_id: str) -> bool:
        return condition_id in self._conditions

    def condition(self, condition_id: str) -> Condition:
        """Condição pelo id.

        Raises:
            CatalogError: id desconhecido.
        """
        try:
            return self._conditions[condition_id]
        except KeyError:
            raise CatalogError(f"condição desconhecida: {condition_id!r}") from None

    def entry(self, condition_id: str, model_id: str, label: str) -> LevelEntry | None:
        return self._levels.get((condition_id, model_id, label))

    def level(self, condition_id: str, model_id: str, label: str) -> EvidenceLevel:
        """Nível de evidência; ``D`` sempre que nenhuma entrada gerada sustentar mais."""
        entry = self.entry(condition_id, model_id, label)
        return entry.level if entry else EvidenceLevel.D

    def visibility(self, condition_id: str, model_id: str, label: str) -> Visibility:
        """Visibilidade derivada do nível: D só em modo pesquisa."""
        return visibility_for(self.level(condition_id, model_id, label))

    def entries(self) -> list[LevelEntry]:
        return list(self._levels.values())
