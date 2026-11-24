"""Camada de ingestão DICOM — agnóstica de modalidade."""

from __future__ import annotations

from radiologyai.io.deident import DeidentificationProfile, deidentify
from radiologyai.io.metadata import StudyMetadata
from radiologyai.io.reader import read_dicom, to_pixel_array
from radiologyai.io.windowing import WINDOW_PRESETS, WindowPreset, apply_window

__all__ = [
    "WINDOW_PRESETS",
    "DeidentificationProfile",
    "StudyMetadata",
    "WindowPreset",
    "apply_window",
    "deidentify",
    "read_dicom",
    "to_pixel_array",
]
