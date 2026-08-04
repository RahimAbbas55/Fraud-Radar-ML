"""
    Decision layer combining the ML model's probability with the rules
    engine's output into a single, actionable decision band.
"""

"""
    <DESIGN PRINCIPLES>
        A fired rule can only escalate a decision toward
        more caution (allow -> review -> block), never reduce it. This
        reflects that rules exist as guardrails/tiebreakers alongside the ML
        score, not replacements for it — see rules.py for the reasoning
        behind each individual rule.
"""

from src.rules import evaluate_rules

ALLOW_THRESHOLD = 0.3
BLOCK_THRESHOLD = 0.8
DECISION_ORDER = ["allow", "review", "block"]  # escalation order, low to high caution


"""Pure ML-driven decision band, before rules are considered."""
def _band_from_probability(probability: float) -> str:
    if probability > BLOCK_THRESHOLD:
        return "block"
    if probability >= ALLOW_THRESHOLD:
        return "review"
    return "allow"

"""
    Returns whichever band represents MORE caution between the two —
    used so a fired rule can only push a decision up, never down.
"""
def _escalate(current_band: str, minimum_band: str) -> str:
    current_idx = DECISION_ORDER.index(current_band)
    minimum_idx = DECISION_ORDER.index(minimum_band)
    return DECISION_ORDER[max(current_idx, minimum_idx)]


def make_decision(features: dict, ml_probability: float) -> dict:
    """
    Combine the ML probability and fired rules into a final decision.
    """
    band = _band_from_probability(ml_probability)
    fired_rules = evaluate_rules(features, ml_probability)

    if fired_rules:
        # Any fired rule escalates the decision to at least "review" —
        # a rule alone isn't enough to justify an outright block, but
        # it's enough to justify not silently allowing the transaction.
        band = _escalate(band, "review")

    return {
        "decision": band,
        "ml_probability": ml_probability,
        "fired_rules": fired_rules,
    }