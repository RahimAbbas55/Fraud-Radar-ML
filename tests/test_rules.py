"""
Tests for src/rules.py.

Focus is on boundary behavior (exactly at a threshold, just above,
just below) since that's where off-by-one style bugs typically hide.
"""

from src.rules import (
    high_amount_rule,
    unusual_hour_borderline_score_rule,
    evaluate_rules,
    HIGH_AMOUNT_THRESHOLD,
    BORDERLINE_SCORE_LOW,
    BORDERLINE_SCORE_HIGH,
)


def test_high_amount_rule_fires_above_threshold():
    result = high_amount_rule({"Amount": HIGH_AMOUNT_THRESHOLD + 0.01})
    assert result is not None
    assert result["rule"] == "high_amount"

# Rule uses > not >=, so exactly at threshold should NOT fire
def test_high_amount_rule_does_not_fire_at_exact_threshold():
    result = high_amount_rule({"Amount": HIGH_AMOUNT_THRESHOLD})
    assert result is None

def test_high_amount_rule_does_not_fire_below_threshold():
    result = high_amount_rule({"Amount": HIGH_AMOUNT_THRESHOLD - 100})
    assert result is None

def test_high_amount_rule_handles_missing_amount():
    result = high_amount_rule({})
    assert result is None

def test_unusual_hour_rule_fires_when_both_conditions_met():
    result = unusual_hour_borderline_score_rule(
        {"hour_of_day": 2}, ml_probability=0.35
    )
    assert result is not None
    assert result["rule"] == "unusual_hour_borderline_score"

def test_unusual_hour_rule_does_not_fire_at_normal_hour():
    result = unusual_hour_borderline_score_rule(
        {"hour_of_day": 14}, ml_probability=0.35
    )
    assert result is None

def test_unusual_hour_rule_does_not_fire_outside_borderline_range():
    # Unusual hour, but ML score is confidently NOT fraud
    result = unusual_hour_borderline_score_rule(
        {"hour_of_day": 2}, ml_probability=0.01
    )
    assert result is None

def test_unusual_hour_rule_fires_at_score_boundaries():
    # Exactly at BORDERLINE_SCORE_LOW and BORDERLINE_SCORE_HIGH should fire
    # (rule uses <=, inclusive on both ends)
    result_low = unusual_hour_borderline_score_rule(
        {"hour_of_day": 3}, ml_probability=BORDERLINE_SCORE_LOW
    )
    result_high = unusual_hour_borderline_score_rule(
        {"hour_of_day": 3}, ml_probability=BORDERLINE_SCORE_HIGH
    )
    assert result_low is not None
    assert result_high is not None

def test_unusual_hour_rule_handles_missing_inputs():
    assert unusual_hour_borderline_score_rule({}, ml_probability=0.35) is None
    assert unusual_hour_borderline_score_rule({"hour_of_day": 2}, ml_probability=None) is None

def test_evaluate_rules_returns_empty_list_when_nothing_fires():
    result = evaluate_rules({"Amount": 50, "hour_of_day": 14}, ml_probability=0.05)
    assert result == []

def test_evaluate_rules_returns_multiple_fired_rules():
    # Construct a transaction that fires BOTH rules simultaneously
    result = evaluate_rules(
        {"Amount": HIGH_AMOUNT_THRESHOLD + 1, "hour_of_day": 2},
        ml_probability=0.35,
    )
    assert len(result) == 2
    fired_rule_names = {r["rule"] for r in result}
    assert fired_rule_names == {"high_amount", "unusual_hour_borderline_score"}