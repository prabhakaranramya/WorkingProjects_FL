import numpy as np
import skfuzzy as fuzz

#Input ranges for gas and smoke sensors

gas_range = np.arange(0, 50.1, 0.1)

smoke_range = np.arange(0, 10.1, 0.1)

gas_risk = np.arange(0, 101, 1)

# GAS MEMBERSHIP - Trapezoidal

gas_low = fuzz.trapmf(gas_range, [0, 0, 10, 20])

gas_medium = fuzz.trapmf(gas_range, [10, 20, 30, 40])

gas_high = fuzz.trapmf(gas_range, [30, 40, 50, 50])

# Smoke MEMBERSHIP - Trapezoidal

smoke_low = fuzz.trapmf(smoke_range, [0, 0, 2, 4])

smoke_medium = fuzz.trapmf(smoke_range, [2, 4, 6, 8])

smoke_high = fuzz.trapmf(smoke_range, [6, 8, 10, 10])

# Gas Risk Aggregated

gas_safe = fuzz.trapmf(gas_risk, [0, 0, 20, 40])

gas_warning = fuzz.trapmf(gas_risk, [30, 40, 60, 70])

gas_danger = fuzz.trapmf(gas_risk, [60, 80, 100, 100])


# Fuzzified values for gas and smoke


def fuzzification(gas_value, smoke_value):

    gas_dom = {
        "Low": fuzz.interp_membership(gas_range, gas_low, gas_value),
        "Medium": fuzz.interp_membership(gas_range, gas_medium, gas_value),
        "High": fuzz.interp_membership(gas_range, gas_high, gas_value),
    }

    smoke_dom = {
        "Low": fuzz.interp_membership(smoke_range, smoke_low, smoke_value),
        "Medium": fuzz.interp_membership(smoke_range, smoke_medium, smoke_value),
        "High": fuzz.interp_membership(smoke_range, smoke_high, smoke_value),
    }

    return gas_dom, smoke_dom


#GAS FIS

    # Rules for Gas FIS (Mamdani, 3x3 rule base = 9 rules)
    # Each rule combines Gas level AND Smoke level using fuzzy AND (min operator)
    # to produce a firing strength, which maps to a hazard output category.
    #
    #              Smoke Low   Smoke Medium   Smoke High
    # Gas Low    |   Safe    |   Warning    |   Warning   |
    # Gas Medium |  Warning  |   Danger     |   Danger    |
    # Gas High   |  Danger   |   Danger     |   Danger    |
    #
    # min(G_x, S_y) = firing strength of rule (fuzzy AND / T-norm)

def gas_fis(gas_value, smoke_value):

    gas_dom, smoke_dom = fuzzification(gas_value, smoke_value)

    G_L, G_M, G_H = gas_dom["Low"], gas_dom["Medium"], gas_dom["High"]
    S_L, S_M, S_H = smoke_dom["Low"], smoke_dom["Medium"], smoke_dom["High"]

    #Rules for Gas FIS

    rules = {
        "G1": (min(G_L, S_L), "Safe",    "Gas Low AND Smoke Low"),
        "G2": (min(G_L, S_M), "Warning", "Gas Low AND Smoke Medium"),
        "G3": (min(G_L, S_H), "Warning", "Gas Low AND Smoke High"),
        "G4": (min(G_M, S_L), "Warning", "Gas Medium AND Smoke Low"),
        "G5": (min(G_M, S_M), "Danger",  "Gas Medium AND Smoke Medium"),
        "G6": (min(G_M, S_H), "Danger",  "Gas Medium AND Smoke High"),
        "G7": (min(G_H, S_L), "Danger",  "Gas High AND Smoke Low"),
        "G8": (min(G_H, S_M), "Danger",  "Gas High AND Smoke Medium"),
        "G9": (min(G_H, S_H), "Danger",  "Gas High AND Smoke High"),
    }

    #Rules Triggered (Fired Rules) - Only rules with firing strength > 0 are considered

    fired_rules = []

    for rule_id, (strength, label, description) in rules.items():

        if strength > 0:

            fired_rules.append(
                f"{rule_id}: {description} -> {label} | Strength {round(float(strength), 2)}"
            )

    # Rule outputs for each hazard category (Safe, Warning, Danger)

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

    #Output Membership Functions for each hazard category (Safe, Warning, Danger)

    safe_output = np.fmin(safe_strength, gas_safe)

    warning_output = np.fmin(warning_strength, gas_warning)

    danger_output = np.fmin(danger_strength, gas_danger)

    # Aggregation

    output = np.fmax(safe_output, np.fmax(warning_output, danger_output))

    #Defuzzification - Centroid Method

    if np.sum(output) == 0:

        score = 0

    else:

        score = fuzz.defuzz(gas_risk, output, "centroid") 
        """Centroid method calculates the center of area under the curve of the aggregated output membership function to produce a crisp score."""

    score = round(float(score), 2)

    #Final Decision based on thresholds

    if score <= 30:

        decision = "SAFE - No Gas Leak Detected"

    elif score <= 65:

        decision = "WARNING - Possible Gas Leak"

    else:

        decision = "DANGER - Gas Leak Alert"

    return score, decision, fired_rules, output


#Final app.py

__all__ = [
    "gas_fis",
    "gas_risk",
    "gas_safe",
    "gas_warning",
    "gas_danger",
]
