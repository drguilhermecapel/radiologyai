"""Read access to the condition catalog and the generated evidence levels."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from medai.catalog.schema import Condition, LevelEntry, LevelsFile, Protocol
from medai.core.errors import CatalogError
from medai.core.types import EvidenceLevel, Visibility

PACKAGE_DIR = Path(__file__).parent
CONDITIONS_FILE = "conditions.yaml"
LEVELS_FILE = "levels.json"
PROTOCOL_FILE = "protocol.yaml"


def load_protocol(directory: Path = PACKAGE_DIR) -> Protocol:
    path = directory / PROTOCOL_FILE
    try:
        return Protocol.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except Exception as exc:
        raise CatalogError(f"{path}: {exc}") from exc


class Catalog:
    """Conditions plus the evidence level of every (condition, model, label)."""

    def __init__(self, conditions: list[Condition], levels: LevelsFile | None) -> None:
        self._conditions: dict[str, Condition] = {}
        for condition in conditions:
            if condition.id in self._conditions:
                raise CatalogError(f"duplicate condition id {condition.id!r}")
            self._conditions[condition.id] = condition
        self.levels_file = levels
        self._levels: dict[tuple[str, str, str], LevelEntry] = {}
        for entry in levels.entries if levels else []:
            if entry.condition_id not in self._conditions:
                raise CatalogError(f"levels refer to unknown condition {entry.condition_id!r}")
            self._levels[(entry.condition_id, entry.model_id, entry.label)] = entry

    @classmethod
    def load(cls, directory: Path = PACKAGE_DIR) -> Catalog:
        conditions_path = directory / CONDITIONS_FILE
        try:
            raw = yaml.safe_load(conditions_path.read_text(encoding="utf-8")) or {}
            conditions = [Condition.model_validate(item) for item in raw.get("conditions", [])]
        except FileNotFoundError:
            conditions = []
        except Exception as exc:
            raise CatalogError(f"{conditions_path}: {exc}") from exc
        levels_path = directory / LEVELS_FILE
        levels: LevelsFile | None = None
        if levels_path.is_file():
            try:
                levels = LevelsFile.model_validate(
                    json.loads(levels_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise CatalogError(f"{levels_path}: {exc}") from exc
        return cls(conditions, levels)

    # -- queries ----------------------------------------------------------- #
    def conditions(self) -> list[Condition]:
        return list(self._conditions.values())

    def has(self, condition_id: str) -> bool:
        return condition_id in self._conditions

    def condition(self, condition_id: str) -> Condition:
        try:
            return self._conditions[condition_id]
        except KeyError:
            raise CatalogError(f"unknown condition {condition_id!r}") from None

    def entry(self, condition_id: str, model_id: str, label: str) -> LevelEntry | None:
        return self._levels.get((condition_id, model_id, label))

    def level(self, condition_id: str, model_id: str, label: str) -> EvidenceLevel:
        """Evidence level; ``D`` whenever no generated entry supports more."""
        entry = self.entry(condition_id, model_id, label)
        return entry.level if entry else EvidenceLevel.D

    def visibility(self, condition_id: str, model_id: str, label: str) -> Visibility:
        return (
            "research" if self.level(condition_id, model_id, label) is EvidenceLevel.D else "main"
        )

    def entries(self) -> list[LevelEntry]:
        return list(self._levels.values())
