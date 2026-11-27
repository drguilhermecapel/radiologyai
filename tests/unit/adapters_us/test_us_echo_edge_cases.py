"""Casos de borda do traçado assistido do VE e do método dos discos (REQ-145, REQ-146).

As máscaras são geométricas (conhecidas); nenhuma vem de modelo.
"""

from __future__ import annotations

import numpy as np
import pytest
from us_support import half_ellipse, have

from radiologyai.analysis.adapters.echo_medsam2 import (
    EchoPromptError,
    _parse_prompts,
    _spacing2,
    atlas_box,
    lv_measurements,
)
from radiologyai.analysis.adapters.echo_simpson import (
    SimpsonError,
    disc_geometry,
    ejection_fraction,
    single_plane_volume,
)
from radiologyai.analysis.base import ModelInputs
from radiologyai.analysis.types import AbstentionReason, Primitive
from radiologyai.catalog.schema import ExamType
from radiologyai.errors import UnsupportedInputError

pytestmark = pytest.mark.skipif(not have("scipy"), reason="scipy ausente")

SP = (0.3, 0.3)


@pytest.mark.requirement("REQ-146")
def test_echo_atlas_box_maps_normalized_box_to_sector() -> None:
    frame = np.zeros((100, 200), np.uint8)
    frame[10:90, 20:180] = 50
    box = atlas_box(frame, (0.0, 0.5, 1.0, 1.0))
    assert box.tolist() == [20.0, 49.5, 179.0, 89.0]
    with pytest.raises(UnsupportedInputError, match="setor"):
        atlas_box(np.zeros((5, 5), np.uint8), (0.1, 0.1, 0.9, 0.9))


@pytest.mark.requirement("REQ-146")
def test_echo_prompt_parsing_ignores_foreign_labels_and_rejects_duplicates() -> None:
    def box(label: str | None, kind: str = "bbox") -> Primitive:
        xy = ((1.0, 1.0), (9.0, 9.0)) if kind == "bbox" else ((1.0, 1.0),)
        return Primitive(type=kind, xy=xy, label=label)  # type: ignore[arg-type]

    inputs = ModelInputs(
        images=(),
        exam_type=ExamType.ECHO,
        prompts=(
            box("A4C:ED"),
            box(None),
            box("a4c:ed", kind="point"),
            box("a4c:mid"),
            box("plax:ed"),
            box("a4c"),
        ),
    )
    assert set(_parse_prompts(inputs)) == {("a4c", "ed")}
    dup = ModelInputs(images=(), exam_type=ExamType.ECHO, prompts=(box("a4c:es"), box("a4c:ES")))
    with pytest.raises(EchoPromptError, match="duplicado"):
        _parse_prompts(dup)


@pytest.mark.requirement("REQ-145")
def test_echo_lv_measurements_report_area_when_discs_are_inapplicable() -> None:
    good = half_ellipse(40.0, 18.0, SP)
    blob = np.zeros((60, 60), bool)
    blob[20, 20] = True  # um pixel: contorno degenerado, discos inaplicáveis
    traced = {
        ("a4c", "ed"): (good, SP, 0),
        ("a4c", "es"): (blob, SP, 4),
    }
    meas = {m.key: m for m in lv_measurements(traced)}
    es = meas["lv_cavity_area_a4c_es"]
    assert es.value == pytest.approx(SP[0] * SP[1])
    assert es.note is not None and "inaplicável" in es.note
    assert meas["lv_ef"].value is None
    assert meas["lv_ef"].abstention is AbstentionReason.QUALITY
    assert meas["lv_esv"].abstention is AbstentionReason.QUALITY


@pytest.mark.requirement("REQ-145")
def test_echo_simpson_input_validation() -> None:
    with pytest.raises(SimpsonError, match="2D"):
        disc_geometry(np.zeros((2, 2, 2), bool), SP)
    with pytest.raises(SimpsonError, match="espaçamento"):
        disc_geometry(half_ellipse(40.0, 18.0, SP), (0.3,))  # type: ignore[arg-type]
    with pytest.raises(SimpsonError, match="espaçamento inválido"):
        disc_geometry(half_ellipse(40.0, 18.0, SP), (0.3, -1.0))
    with pytest.raises(SimpsonError, match="n_discs"):
        disc_geometry(half_ellipse(40.0, 18.0, SP), SP, n_discs=0)
    geom = disc_geometry(half_ellipse(40.0, 18.0, SP), SP)
    assert single_plane_volume(geom).volume_ml > 0
    with pytest.raises(SimpsonError):
        ejection_fraction(0.0, 0.0)


def test_echo_spacing2_reads_last_two_axes() -> None:
    from us_support import us_image

    assert _spacing2(us_image(np.zeros((4, 4)), spacing=None)) is None
    assert _spacing2(us_image(np.zeros((4, 4)), spacing=(0.2, 0.4))) == (0.2, 0.4)
