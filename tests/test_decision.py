"""
Tests for src/decision.py.
Covers band boundaries (exact threshold values) and the escalation mechanism specifically
"""

from src.decision import make_decision, _band_from_probability, _escalate, ALLOW_THRESHOLD, BLOCK_THRESHOLD

def test_band_from_probability_allow_below_threshold():
    assert _band_from_probability(ALLOW_THRESHOLD - 0.01) == "allow"

# Exactly at ALLOW_THRESHOLD should be "review", not "allow" (rule uses >=)
def test_band_from_probability_review_at_allow_threshold():
    assert _band_from_probability(ALLOW_THRESHOLD) == "review"


# Exactly at BLOCK_THRESHOLD should be "review", not "block" (rule uses >)
def test_band_from_probability_review_at_block_threshold():
    assert _band_from_probability(BLOCK_THRESHOLD) == "review"

def test_band_from_probability_block_above_threshold():
    assert _band_from_probability(BLOCK_THRESHOLD + 0.01) == "block"

def test_escalate_moves_toward_more_caution():
    assert _escalate("allow", "review") == "review"
    assert _escalate("review", "block") == "block"
    assert _escalate("allow", "block") == "block"

def test_escalate_never_downgrades():
    # Escalating an already-severe decision with a LOWER minimum
    # should leave it unchanged, not downgrade it.
    assert _escalate("block", "review") == "block"
    assert _escalate("review", "allow") == "review"
    assert _escalate("block", "allow") == "block"

def test_make_decision_allow_when_no_rules_and_low_probability():
    result = make_decision({"Amount": 50, "hour_of_day": 14}, ml_probability=0.05)
    assert result["decision"] == "allow"
    assert result["fired_rules"] == []

def test_make_decision_block_from_probability_alone():
    result = make_decision({"Amount": 50, "hour_of_day": 14}, ml_probability=0.95)
    assert result["decision"] == "block"

def test_make_decision_escalates_allow_to_review_when_rule_fires():
    result = make_decision({"Amount": 5000, "hour_of_day": 14}, ml_probability=0.05)
    assert result["decision"] == "review"
    assert len(result["fired_rules"]) == 1

def test_make_decision_stays_block_when_rule_fires_on_top_of_block():
    result = make_decision({"Amount": 5000, "hour_of_day": 14}, ml_probability=0.95)
    assert result["decision"] == "block"

def test_make_decision_includes_ml_probability_in_output():
    result = make_decision({"Amount": 50, "hour_of_day": 14}, ml_probability=0.42)
    assert result["ml_probability"] == 0.42