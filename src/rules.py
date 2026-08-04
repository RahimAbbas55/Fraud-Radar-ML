# Rules engine for the Radar-style decision layer.

HIGH_AMOUNT_THRESHOLD = 2000.0  # above the max fraud amount seen in Day 1 EDA
UNUSUAL_HOURS = {0, 1, 2, 3, 4}  # overnight hours where fraud clustered in EDA
BORDERLINE_SCORE_LOW = 0.2
BORDERLINE_SCORE_HIGH = 0.5


def high_amount_rule(features: dict, ml_probability: float = None) -> dict | None:
    # Flags amounts above threshold; a guardrail, not a strong fraud signal per EDA.
    amount = features.get("Amount", 0)
    if amount > HIGH_AMOUNT_THRESHOLD:
        return {
            "rule": "high_amount",
            "reason": f"Transaction amount £{amount:.2f} exceeds £{HIGH_AMOUNT_THRESHOLD:.2f} threshold",
        }
    return None


def unusual_hour_borderline_score_rule(features: dict, ml_probability: float = None) -> dict | None:
    # Fires only when unusual hour AND ML score is borderline — a tiebreaker, not standalone.
    hour = features.get("hour_of_day")
    if hour is None or ml_probability is None:
        return None

    is_unusual_hour = hour in UNUSUAL_HOURS
    is_borderline = BORDERLINE_SCORE_LOW <= ml_probability <= BORDERLINE_SCORE_HIGH

    if is_unusual_hour and is_borderline:
        return {
            "rule": "unusual_hour_borderline_score",
            "reason": f"Transaction at hour {hour} combined with borderline ML score ({ml_probability:.3f})",
        }
    return None


ALL_RULES = [high_amount_rule, unusual_hour_borderline_score_rule]  # add new rules here


def evaluate_rules(features: dict, ml_probability: float = None) -> list[dict]:
    # Runs every rule, returns list of fired-rule dicts (empty if none fired).
    fired = []
    for rule_fn in ALL_RULES:
        result = rule_fn(features, ml_probability)
        if result is not None:
            fired.append(result)
    return fired