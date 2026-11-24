"""Decisão por ponto de operação com origem declarada."""

from __future__ import annotations

import itertools

import pytest
from pydantic import ValidationError

from radiologyai.calibration.decision import OperatingPoint, band_for, decide, op_normalize
from radiologyai.errors import EvaluationError
from radiologyai.inference.types import Band


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


@pytest.mark.requirement("REQ-092")
def test_no_operating_point_means_score_only() -> None:
    assert decide({"sigmoid": 0.99}, []) == ("score_only", [])


@pytest.mark.requirement("REQ-092")
def test_points_on_unavailable_scale_are_ignored() -> None:
    decision, used = decide({"sigmoid": 0.99}, [point("balanced", 0.5, on="calibrated")])
    assert decision == "score_only"
    assert used == []


@pytest.mark.requirement("REQ-092")
@pytest.mark.parametrize("source", ["", "   "])
def test_operating_point_requires_declared_source(source: str) -> None:
    fields = {"name": "p", "role": "balanced", "threshold": 0.5, "on": "sigmoid"}
    with pytest.raises(ValidationError):
        OperatingPoint.model_validate(fields)
    with pytest.raises(ValidationError, match="origem declarada"):
        OperatingPoint.model_validate({**fields, "source": source})


@pytest.mark.requirement("REQ-092")
def test_every_verdict_names_the_points_it_used() -> None:
    points = [point("rule_out", 0.30), point("rule_in", 0.70), point("balanced", 0.5)]
    for score in (0.1, 0.5, 0.9):
        decision, used = decide({"sigmoid": score}, points)
        assert decision != "score_only"
        assert used
        assert all(p.source for p in used)


def test_operating_point_carries_no_performance_fields() -> None:
    forbidden = {"accuracy", "auc", "sensitivity", "specificity"}
    assert not forbidden & set(OperatingPoint.model_fields)
    with pytest.raises(ValidationError):
        OperatingPoint.model_validate(
            {
                "name": "p",
                "role": "balanced",
                "threshold": 0.5,
                "on": "sigmoid",
                "source": "test",
                "sensitivity": 0.5,
            }
        )


def test_operating_point_rejects_non_finite_threshold() -> None:
    with pytest.raises(ValidationError):
        point("balanced", float("nan"))


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


@pytest.mark.parametrize("role", ["rule_out", "rule_in"])
def test_lone_rule_point_is_two_way(role: str) -> None:
    p = point(role, 0.4)
    assert decide({"sigmoid": 0.39}, [p]) == ("negative", [p])
    assert decide({"sigmoid": 0.40}, [p]) == ("positive", [p])


def test_rule_points_on_unavailable_scale_fall_back_to_single_point() -> None:
    balanced = point("balanced", 0.5)
    points = [point("rule_out", 0.3, on="calibrated"), point("rule_in", 0.7), balanced]
    assert decide({"sigmoid": 0.6}, points) == ("positive", [balanced])


@pytest.mark.parametrize(
    ("value", "expected"), [(100.0, "positive"), (110.0, "positive"), (111.0, "negative")]
)
def test_le_direction_for_measurements(value: float, expected: str) -> None:
    hu = point("guideline", 110.0, on="measurement", direction="le")
    assert decide({"measurement": value}, [hu])[0] == expected


def test_calibrated_scale_is_used_when_available() -> None:
    points = [point("balanced", 0.2, on="calibrated")]
    assert decide({"sigmoid": 0.9, "calibrated": 0.1}, points)[0] == "negative"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_score_is_never_decided(bad: float) -> None:
    """NaN compararia como falso e viraria "negativo" em silêncio."""
    with pytest.raises(EvaluationError, match="não finito"):
        decide({"sigmoid": bad}, [point("balanced", 0.5)])
    with pytest.raises(EvaluationError, match="não finito"):
        decide({"sigmoid": bad}, [point("rule_out", 0.3), point("rule_in", 0.7)])
    with pytest.raises(EvaluationError, match="não finito"):
        op_normalize({"sigmoid": bad}, [point("balanced", 0.5)])


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


def test_op_normalize_le_direction_is_mirrored() -> None:
    p = point("balanced", 0.8, direction="le")
    assert op_normalize({"sigmoid": 0.8}, [p]) == pytest.approx(0.5)
    assert op_normalize({"sigmoid": 0.0}, [p]) == pytest.approx(1.0)
    assert op_normalize({"sigmoid": 1.0}, [p]) == pytest.approx(0.0)


def test_op_normalize_prefers_balanced_then_rule_in() -> None:
    rule_out, rule_in = point("rule_out", 0.2), point("rule_in", 0.8)
    balanced = point("balanced", 0.5)
    assert op_normalize({"sigmoid": 0.5}, [rule_out, rule_in, balanced]) == pytest.approx(0.5)
    assert op_normalize({"sigmoid": 0.8}, [rule_out, rule_in]) == pytest.approx(0.5)
    assert op_normalize({"sigmoid": 0.2}, [rule_out]) == pytest.approx(0.5)


def test_op_normalize_none_when_not_applicable() -> None:
    assert op_normalize({"measurement": 0.6}, [point("guideline", 0.5, on="measurement")]) is None
    assert op_normalize({"sigmoid": 0.6}, []) is None
    assert op_normalize({"sigmoid": 0.6}, [point("balanced", 0.5, on="calibrated")]) is None
    assert op_normalize({"sigmoid": 0.6}, [point("balanced", 1.0)]) is None
    assert op_normalize({"sigmoid": 0.6}, [point("balanced", 0.0)]) is None


def test_decision_maps_onto_band() -> None:
    assert band_for("positive") is Band.LIKELY
    assert band_for("negative") is Band.UNLIKELY
    assert band_for("indeterminate") is Band.INDETERMINATE
    # sem ponto de operação não há base para afirmar nada: não avaliável
    assert band_for("score_only") is Band.INDETERMINATE
