from __future__ import annotations

import itertools

import pytest

from medai.core.types import OperatingPoint
from medai.pipeline import decide, op_normalize


def point(
    role: str, threshold: float, on: str = "sigmoid", direction: str = "ge"
) -> OperatingPoint:
    return OperatingPoint.model_validate(
        {
            "name": role,
            "role": role,
            "threshold": threshold,
            "on": on,
            "direction": direction,
            "source": "test",
        }
    )


def test_no_operating_point_means_score_only() -> None:
    assert decide({"sigmoid": 0.99}, []) == ("score_only", [])


def test_points_on_unavailable_scale_are_ignored() -> None:
    decision, used = decide({"sigmoid": 0.99}, [point("balanced", 0.5, on="calibrated")])
    assert decision == "score_only"
    assert used == []


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.10, "negative"),
        (0.29, "negative"),
        (0.30, "indeterminate"),
        (0.69, "indeterminate"),
        (0.70, "positive"),
        (0.99, "positive"),
    ],
)
def test_three_way_decision_with_indeterminate_band(score: float, expected: str) -> None:
    points = [point("rule_out", 0.30), point("rule_in", 0.70)]
    decision, used = decide({"sigmoid": score}, points)
    assert decision == expected
    assert [p.role for p in used] == ["rule_out", "rule_in"]


@pytest.mark.parametrize(("score", "expected"), [(0.49, "negative"), (0.5, "positive")])
def test_single_point_is_two_way(score: float, expected: str) -> None:
    assert decide({"sigmoid": score}, [point("balanced", 0.5)])[0] == expected


@pytest.mark.parametrize(
    ("value", "expected"), [(100.0, "positive"), (110.0, "positive"), (111.0, "negative")]
)
def test_le_direction_for_measurements(value: float, expected: str) -> None:
    hu = point("guideline", 110.0, on="measurement", direction="le")
    assert decide({"measurement": value}, [hu])[0] == expected


def test_calibrated_scale_is_used_when_available() -> None:
    points = [point("balanced", 0.2, on="calibrated")]
    assert decide({"sigmoid": 0.9, "calibrated": 0.1}, points)[0] == "negative"


def test_op_normalize_puts_threshold_at_half() -> None:
    p = point("balanced", 0.2)
    assert op_normalize({"sigmoid": 0.2}, [p]) == pytest.approx(0.5)
    assert op_normalize({"sigmoid": 0.0}, [p]) == pytest.approx(0.0)
    assert op_normalize({"sigmoid": 1.0}, [p]) == pytest.approx(1.0)
    assert op_normalize({"sigmoid": 0.1}, [p]) == pytest.approx(0.25)


def test_op_normalize_is_monotonic() -> None:
    p = point("balanced", 0.37)
    values = [op_normalize({"sigmoid": s / 100}, [p]) for s in range(101)]
    assert all(a is not None and b is not None and a <= b for a, b in itertools.pairwise(values))


def test_op_normalize_none_for_unbounded_scales() -> None:
    assert op_normalize({"measurement": 0.6}, [point("guideline", 0.5, on="measurement")]) is None
    assert op_normalize({"sigmoid": 0.6}, []) is None
