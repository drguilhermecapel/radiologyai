"""Exception hierarchy. Failures are explicit; nothing falls back to made-up output."""

from __future__ import annotations


class MedAIError(Exception):
    """Base class for every error raised on purpose by MedAI."""


class ConfigurationError(MedAIError):
    """Settings, manifests or routing are invalid."""


class ManifestError(ConfigurationError):
    """A model manifest is malformed or violates a registry rule."""


class ModelUnavailable(MedAIError):  # noqa: N818 - domain name, mapped to HTTP 503
    """A model cannot be used: planned, weights missing, or integrity check failed.

    The API maps this to HTTP 503. It must never be swallowed and replaced by a result.
    """

    def __init__(self, model_id: str, reason: str) -> None:
        super().__init__(f"model {model_id!r} unavailable: {reason}")
        self.model_id = model_id
        self.reason = reason


class IntegrityError(ModelUnavailable):
    """A weight file does not match its pinned SHA-256."""


class UnsupportedInput(MedAIError):  # noqa: N818
    """The input cannot be decoded or is outside what the system handles."""


class ExamTypeUnknown(MedAIError):  # noqa: N818
    """The exam type could not be established from metadata; a human must choose."""


class CatalogError(ConfigurationError):
    """The condition catalog or an evidence file is inconsistent."""


class EvidenceError(MedAIError):
    """An evidence record is missing provenance or fails validation."""
