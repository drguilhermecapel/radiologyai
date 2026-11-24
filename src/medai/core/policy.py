"""Safety policy interface used by the pipeline.

FROZEN CONTRACT. The concrete, artifact-backed implementation lives in
``medai.safety``; the pipeline only depends on this protocol.
"""

from __future__ import annotations

from typing import Protocol

from medai.core.types import OperatingPoint, ScoreKind


class SafetyPolicy(Protocol):
    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: ScoreKind
    ) -> tuple[float, str] | None:
        """Return ``(calibrated_probability, calibration_id)`` or ``None``.

        ``None`` means no fitted calibration artifact exists for this model and
        label; the pipeline then reports no probability at all.
        """
        ...

    def operating_points(self, model_id: str, label: str) -> list[OperatingPoint]:
        """Operating points fitted under the evidence protocol.

        An empty list makes the pipeline fall back to the points published in
        the model manifest (for example the library's own thresholds).
        """
        ...


class ManifestOnlyPolicy:
    """Policy with no fitted artifacts: no calibration, manifest operating points only."""

    def calibrate(
        self, model_id: str, label: str, raw_score: float, score_kind: ScoreKind
    ) -> tuple[float, str] | None:
        return None

    def operating_points(self, model_id: str, label: str) -> list[OperatingPoint]:
        return []
