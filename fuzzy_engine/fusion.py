"""
SHIELD - Hierarchical Fusion Layer

The four subsystems (fire, gas, flood, intrusion) each already produce a
0-100 risk score via their own Mamdani FIS. This module is a *second-tier*
Mamdani FIS that fuses those four scores into a single "Overall Home Risk"
score, turning SHIELD into a hierarchical fuzzy inference system rather than
four independent, unrelated dashboards.

DESIGN NOTE - why not a 3^4 = 81 rule combinatorial FIS?
A rule base built directly on the four raw scores (Low/Medium/High each)
would need 81 rules and would be sparse, hard to justify, and hard to
maintain. Instead we first reduce the four scores to two derived,
information-dense features:

    max_score  - the worst-performing subsystem (captures "any single
                 catastrophic sensor should dominate the decision")
    avg_score  - the mean across all four subsystems (captures "overall
                 combined load", i.e. several simultaneous medium risks
                 should also be taken seriously even if none is individually
                 extreme)

These two derived features are then fuzzified (Low/Medium/High) and combined
with a compact, fully-justifiable 3x3 = 9 rule Mamdani base to produce the
Overall Risk output. This is a standard dimensionality-reduction technique
for fuzzy sensor-fusion problems with many raw inputs.
"""

import numpy as np
import skfuzzy as fuzz


######## INPUT / OUTPUT RANGES ########

feature_range = np.arange(0, 101, 1)   # shared axis for max_score and avg_score

fusion_risk = np.arange(0, 101, 1)     # overall risk output axis


######## DERIVED FEATURE MEMBERSHIP (Max / Avg) ########

feature_low = fuzz.trapmf(feature_range, [0, 0, 20, 40])

feature_medium = fuzz.trapmf(feature_range, [30, 40, 60, 70])

feature_high = fuzz.trapmf(feature_range, [60, 80, 100, 100])


######## OVERALL RISK OUTPUT MEMBERSHIP ########

fusion_safe = fuzz.trapmf(fusion_risk, [0, 0, 20, 40])

fusion_warning = fuzz.trapmf(fusion_risk, [30, 40, 60, 70])

fusion_danger = fuzz.trapmf(fusion_risk, [60, 80, 100, 100])


######## FUZZIFICATION ########


def fuzzification(max_score, avg_score):

    max_dom = {
        "Low": fuzz.interp_membership(feature_range, feature_low, max_score),
        "Medium": fuzz.interp_membership(feature_range, feature_medium, max_score),
        "High": fuzz.interp_membership(feature_range, feature_high, max_score),
    }

    avg_dom = {
        "Low": fuzz.interp_membership(feature_range, feature_low, avg_score),
        "Medium": fuzz.interp_membership(feature_range, feature_medium, avg_score),
        "High": fuzz.interp_membership(feature_range, feature_high, avg_score),
    }

    return max_dom, avg_dom


######## FUSION FUZZY INFERENCE SYSTEM ########


def fusion_fis(fire_score, gas_score, flood_score, intrusion_score):

    scores = [fire_score, gas_score, flood_score, intrusion_score]

    max_score = float(max(scores))

    avg_score = float(sum(scores) / len(scores))

    max_dom, avg_dom = fuzzification(max_score, avg_score)

    MX_L, MX_M, MX_H = max_dom["Low"], max_dom["Medium"], max_dom["High"]
    AV_L, AV_M, AV_H = avg_dom["Low"], avg_dom["Medium"], avg_dom["High"]

    ######## RULE BASE (9 RULES) ########

    rules = {
        "F1": (min(MX_L, AV_L), "Safe",    "Max Low AND Avg Low"),
        "F2": (min(MX_L, AV_M), "Safe",    "Max Low AND Avg Medium"),
        "F3": (min(MX_L, AV_H), "Warning", "Max Low AND Avg High"),

        "F4": (min(MX_M, AV_L), "Safe",    "Max Medium AND Avg Low"),
        "F5": (min(MX_M, AV_M), "Warning", "Max Medium AND Avg Medium"),
        "F6": (min(MX_M, AV_H), "Warning", "Max Medium AND Avg High"),

        "F7": (min(MX_H, AV_L), "Warning", "Max High AND Avg Low"),
        "F8": (min(MX_H, AV_M), "Danger",  "Max High AND Avg Medium"),
        "F9": (min(MX_H, AV_H), "Danger",  "Max High AND Avg High"),
    }

    ######## FIRED RULES ########

    fired_rules = []

    for rule_id, (strength, label, description) in rules.items():

        if strength > 0:

            fired_rules.append(
                f"{rule_id}: {description} -> {label} | Strength {round(float(strength), 2)}"
            )

    ######## AGGREGATE STRENGTH PER OUTPUT SET ########

    safe_strength = max(
        (strength for strength, label, _ in rules.values() if label == "Safe"),
        default=0.0,
    )

    warning_strength = max(
        (strength for strength, label, _ in rules.values() if label == "Warning"),
        default=0.0,
    )

    danger_strength = max(
        (strength for strength, label, _ in rules.values() if label == "Danger"),
        default=0.0,
    )

    ######## MAMDANI IMPLICATION ########

    safe_output = np.fmin(safe_strength, fusion_safe)

    warning_output = np.fmin(warning_strength, fusion_warning)

    danger_output = np.fmin(danger_strength, fusion_danger)

    ######## AGGREGATION ########

    output = np.fmax(safe_output, np.fmax(warning_output, danger_output))

    ######## DEFUZZIFICATION ########

    if np.sum(output) == 0:

        score = 0

    else:

        score = fuzz.defuzz(fusion_risk, output, "centroid")

    score = round(float(score), 2)

    ######## DECISION ########

    if score <= 30:

        decision = "SAFE - Home Secure"

    elif score <= 65:

        decision = "WARNING - Elevated Home Risk"

    else:

        decision = "DANGER - Critical Home Risk"

    features = {"max_score": round(max_score, 2), "avg_score": round(avg_score, 2)}

    return score, decision, fired_rules, output, features


######## EXPORT FOR APP.PY ########

__all__ = [
    "fusion_fis",
    "fusion_risk",
    "fusion_safe",
    "fusion_warning",
    "fusion_danger",
]
