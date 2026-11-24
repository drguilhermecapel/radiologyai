"""Model manifest schema (one YAML file per model under ``models/manifests/``).

FROZEN CONTRACT. A manifest is the single source of truth about a model:
where its weights come from and their pinned SHA-256, its licence, what it is
applicable to, which labels it emits and which catalog condition each label
maps to. It deliberately carries NO performance numbers: evidence lives in
``evidence/`` and the level shown to users is computed from those files.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from medai.core.errors import ManifestError
from medai.core.types import (
    CodedConcept,
    ExamType,
    ImageKind,
    Modality,
    OperatingPoint,
    ScoreKind,
    Visibility,
)

ALLOWED_ADAPTER_PREFIXES = ("medai.models.adapters.", "medai.probes.")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{1,62}[a-z0-9]$")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class WeightFile(_Strict):
    key: str
    url: str
    filename: str
    sha256: str = ""  # written by `medai models pin` from the real bytes; never typed by hand
    bytes: int = 0
    extract: Literal["none", "zip", "tar"] = "none"
    unsafe_pickle: bool = False  # checkpoint needs full unpickling (torch.load weights_only=False)
    note: str | None = None

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: str) -> str:
        if v and not _SHA256.match(v):
            raise ValueError("sha256 must be 64 lowercase hex characters (or empty when unpinned)")
        return v

    @field_validator("filename")
    @classmethod
    def _plain_name(cls, v: str) -> str:
        if "/" in v or "\\" in v or v in ("", ".", ".."):
            raise ValueError("filename must be a plain file name")
        return v


class License(_Strict):
    code: str
    weights: str
    training_data_terms: str | None = None
    commercial_use: Literal["yes", "no", "unclear"]
    evidence: str | None = None  # where the statement was read (URL / file + quote)


class Applicability(_Strict):
    exam_types: list[ExamType]
    modalities: list[Modality]
    image_kinds: list[ImageKind]
    views: list[str] = Field(default_factory=list)  # empty = any
    min_pixels: int | None = None  # shortest in-plane side
    adult_only: bool = False
    acquisition: list[str] = Field(
        default_factory=list
    )  # e.g. "non_contrast", "slice_thickness<=3"


class InputSpec(_Strict):
    mode: Literal["single", "multi"] = "single"
    roles: list[str] = Field(default_factory=list)  # required roles for multi-input models
    optional_roles: list[str] = Field(default_factory=list)
    prompts: Literal["none", "optional", "required"] = "none"


class LabelSpec(_Strict):
    key: str  # label key the adapter emits
    source_label: str  # the name used by the upstream model / library
    condition_id: str  # id in the condition catalog
    kind: Literal["binary", "categorical", "derived"] = "binary"
    score_kind: ScoreKind = "sigmoid"
    code: CodedConcept
    measurement: str | None = None  # measurement name, for derived labels
    classes: list[str] | None = None  # for categorical labels

    @model_validator(mode="after")
    def _consistency(self) -> Self:
        if self.kind == "derived" and not self.measurement:
            raise ValueError("derived labels must name the measurement they threshold")
        if self.kind == "categorical" and not self.classes:
            raise ValueError("categorical labels must list their classes")
        return self


class LabelOperatingPoint(_Strict):
    label: str
    point: OperatingPoint


class Reference(_Strict):
    title: str
    identifier: str | None = None  # DOI / arXiv id
    url: str | None = None


class ModelManifest(_Strict):
    id: str
    version: str
    name: str
    status: Literal["available", "planned", "retired"]
    adapter: str | None = None  # "module.path:ClassName"; required unless planned
    task: Literal[
        "classification", "segmentation", "measurement", "gate", "detection", "zero_shot", "encoder"
    ]
    capabilities: list[Literal["classify", "segment", "measure", "explain", "prompt", "gate"]]
    summary: str
    intended_use: Literal["research_use_only"] = "research_use_only"
    visibility_default: Visibility = "main"  # zero-shot / unvalidated models: "research"

    source: dict[str, str] = Field(default_factory=dict)  # library, name, revision ...
    weights: list[WeightFile] = Field(default_factory=list)
    license: License
    applicability: Applicability
    inputs: InputSpec = Field(default_factory=InputSpec)
    depends_on: list[str] = Field(default_factory=list)  # model ids whose output this one consumes

    training_data: list[str] = Field(default_factory=list)  # dataset keys, for leakage control
    labels: list[LabelSpec] = Field(default_factory=list)
    operating_points: list[LabelOperatingPoint] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    references: list[Reference] = Field(default_factory=list)
    planned_reason: str | None = None  # why a planned model is not available yet
    config: dict[str, Any] = Field(default_factory=dict)  # adapter-specific settings

    @field_validator("id")
    @classmethod
    def _valid_id(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError("id must be lowercase [a-z0-9._-], 3-64 characters")
        return v

    @model_validator(mode="after")
    def _rules(self) -> Self:
        if self.status == "planned":
            if not self.planned_reason:
                raise ValueError("planned models must state planned_reason")
            return self
        if not self.adapter:
            raise ValueError("available models must name an adapter")
        if not self.adapter.startswith(ALLOWED_ADAPTER_PREFIXES) or ":" not in self.adapter:
            raise ValueError(
                f"adapter must be 'module:Class' under {ALLOWED_ADAPTER_PREFIXES}, got {self.adapter!r}"
            )
        label_keys = [label.key for label in self.labels]
        if len(set(label_keys)) != len(label_keys):
            raise ValueError("duplicate label keys")
        for op in self.operating_points:
            if op.label not in label_keys:
                raise ValueError(f"operating point refers to unknown label {op.label!r}")
        weight_keys = [w.key for w in self.weights]
        if len(set(weight_keys)) != len(weight_keys):
            raise ValueError("duplicate weight keys")
        if self.visibility_default == "main" and self.task == "zero_shot":
            raise ValueError("zero-shot models are research-mode only")
        return self

    # -- helpers ----------------------------------------------------------- #
    def label(self, key: str) -> LabelSpec:
        for spec in self.labels:
            if spec.key == key:
                return spec
        raise KeyError(key)

    def points_for(self, label_key: str) -> list[OperatingPoint]:
        return [op.point for op in self.operating_points if op.label == label_key]

    @property
    def is_pinned(self) -> bool:
        return all(w.sha256 for w in self.weights)


def load_manifest(path: Path) -> tuple[ModelManifest, str]:
    """Parse one manifest file. Returns the manifest and the SHA-256 of the file bytes."""
    raw = path.read_bytes()
    try:
        data = yaml.safe_load(raw)
        manifest = ModelManifest.model_validate(data)
    except Exception as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    if path.stem != manifest.id:
        raise ManifestError(f"{path}: file name must equal the model id {manifest.id!r}")
    return manifest, hashlib.sha256(raw).hexdigest()
