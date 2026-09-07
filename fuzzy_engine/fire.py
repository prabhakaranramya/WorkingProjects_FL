import numpy as np
import skfuzzy as fuzz  

# Input Ranges

smoke_range = np.arange(0, 10.1, 0.1)

photo_range = np.arange(0, 10.1, 0.1)

gas_range = np.arange(0, 50.1, 0.1)

fire_risk = np.arange(0, 101, 1)

# Smoke Membership

smoke_low = fuzz.trapmf(smoke_range, [0, 0, 2, 4])

smoke_medium = fuzz.trapmf(smoke_range, [2, 4, 6, 8])

smoke_high = fuzz.trapmf(smoke_range, [6, 8, 10, 10])

# Photoelectric Membership 

photo_low = fuzz.trapmf(photo_range, [0, 0, 2, 4])

photo_medium = fuzz.trapmf(photo_range, [2, 4, 6, 8])

photo_high = fuzz.trapmf(photo_range, [6, 8, 10, 10])

# Gas Membership 

gas_low = fuzz.trapmf(gas_range, [0, 0, 10, 20])

gas_medium = fuzz.trapmf(gas_range, [10, 20, 30, 40])

gas_high = fuzz.trapmf(gas_range, [30, 40, 50, 50])

# Fire Risk Output Membership

safe = fuzz.trapmf(fire_risk, [0, 0, 20, 40])

warning = fuzz.trapmf(fire_risk, [30, 40, 60, 70])

danger = fuzz.trapmf(fire_risk, [60, 80, 100, 100])

# Fuzzification

def fuzzification(smoke_value, photo_value, gas_value):

    smoke_dom = {
        "Low": fuzz.interp_membership(smoke_range, smoke_low, smoke_value),
        "Medium": fuzz.interp_membership(smoke_range, smoke_medium, smoke_value),
        "High": fuzz.interp_membership(smoke_range, smoke_high, smoke_value),
    }

    photo_dom = {
        "Low": fuzz.interp_membership(photo_range, photo_low, photo_value),
        "Medium": fuzz.interp_membership(photo_range, photo_medium, photo_value),
        "High": fuzz.interp_membership(photo_range, photo_high, photo_value),
    }

    gas_dom = {
        "Low": fuzz.interp_membership(gas_range, gas_low, gas_value),
        "Medium": fuzz.interp_membership(gas_range, gas_medium, gas_value),
        "High": fuzz.interp_membership(gas_range, gas_high, gas_value),
    }

    return smoke_dom, photo_dom, gas_dom

# Fire Fuzzy Inference System 

def fire_fis(smoke_value, photo_value, gas_value):

    smoke_dom, photo_dom, gas_dom = fuzzification(smoke_value, photo_value, gas_value)

    S_L, S_M, S_H = smoke_dom["Low"], smoke_dom["Medium"], smoke_dom["High"]
    P_L, P_M, P_H = photo_dom["Low"], photo_dom["Medium"], photo_dom["High"]
    G_L, G_M, G_H = gas_dom["Low"], gas_dom["Medium"], gas_dom["High"]


    rules = {
        "R1":  (min(S_L, P_L, G_L), "Safe",    "Smoke Low AND Photo Low AND Gas Low"),
        "R2":  (min(S_L, P_L, G_M), "Safe",    "Smoke Low AND Photo Low AND Gas Medium"),
        "R3":  (min(S_L, P_M, G_L), "Safe",    "Smoke Low AND Photo Medium AND Gas Low"),
        "R4":  (min(S_M, P_L, G_L), "Safe",    "Smoke Medium AND Photo Low AND Gas Low"),

        "R5":  (min(S_L, P_H, G_L), "Warning", "Smoke Low AND Photo High AND Gas Low"),
        "R6":  (min(S_H, P_L, G_L), "Warning", "Smoke High AND Photo Low AND Gas Low"),
        "R7":  (min(S_L, P_L, G_H), "Warning", "Smoke Low AND Photo Low AND Gas High"),
        "R8":  (min(S_M, P_M, G_L), "Warning", "Smoke Medium AND Photo Medium AND Gas Low"),
        "R9":  (min(S_M, P_L, G_M), "Warning", "Smoke Medium AND Photo Low AND Gas Medium"),
        "R10": (min(S_L, P_M, G_M), "Warning", "Smoke Low AND Photo Medium AND Gas Medium"),

        "R11": (min(S_M, P_M, G_M), "Warning", "Smoke Medium AND Photo Medium AND Gas Medium"),
        "R12": (min(S_H, P_H, G_L), "Danger",  "Smoke High AND Photo High AND Gas Low"),
        "R13": (min(S_H, P_L, G_H), "Danger",  "Smoke High AND Photo Low AND Gas High"),
        "R14": (min(S_L, P_H, G_H), "Danger",  "Smoke Low AND Photo High AND Gas High"),
        "R15": (min(S_H, P_M, G_M), "Danger",  "Smoke High AND Photo Medium AND Gas Medium"),
        "R16": (min(S_M, P_H, G_M), "Danger",  "Smoke Medium AND Photo High AND Gas Medium"),
        "R17": (min(S_M, P_M, G_H), "Danger",  "Smoke Medium AND Photo Medium AND Gas High"),
        "R18": (min(S_H, P_H, G_H), "Danger",  "Smoke High AND Photo High AND Gas High"),

        "R19": (min(S_L, P_M, G_H), "Warning", "Smoke Low AND Photo Medium AND Gas High"),
        "R20": (min(S_L, P_H, G_M), "Warning", "Smoke Low AND Photo High AND Gas Medium"),
        "R21": (min(S_M, P_L, G_H), "Warning", "Smoke Medium AND Photo Low AND Gas High"),
        "R22": (min(S_M, P_H, G_L), "Warning", "Smoke Medium AND Photo High AND Gas Low"),
        "R23": (min(S_H, P_L, G_M), "Warning", "Smoke High AND Photo Low AND Gas Medium"),
        "R24": (min(S_H, P_M, G_L), "Warning", "Smoke High AND Photo Medium AND Gas Low"),

        "R25": (min(S_M, P_H, G_H), "Danger",  "Smoke Medium AND Photo High AND Gas High"),
        "R26": (min(S_H, P_M, G_H), "Danger",  "Smoke High AND Photo Medium AND Gas High"),
        "R27": (min(S_H, P_H, G_M), "Danger",  "Smoke High AND Photo High AND Gas Medium"),
    }

    # Fire Rules 

    fired_rules = []

    for rule_id, (strength, label, description) in rules.items():

        if strength > 0:

            fired_rules.append(
                f"{rule_id}: {description} -> {label} | Strength {round(float(strength), 2)}"
            )

    # Aggregate Strength Per Output Set

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

    # Mamdani Implication 

    safe_output = np.fmin(safe_strength, safe)

    warning_output = np.fmin(warning_strength, warning)

    danger_output = np.fmin(danger_strength, danger)

    # Aggregation 

    output = np.fmax(safe_output, np.fmax(warning_output, danger_output))

    # Defuzzification

    if np.sum(output) == 0:

        score = 0

    else:

        score = fuzz.defuzz(fire_risk, output, "centroid")

    score = round(float(score), 2)

    # Decision

    if score <= 30:

        decision = "SAFE - Normal Condition"

    elif score <= 65:

        decision = "WARNING - Possible Fire Hazard"

    else:

        decision = "DANGER - Fire Alert"

    return score, decision, fired_rules, output

# Export to app.py

__all__ = [
    "fire_fis",
    "fire_risk",
    "safe",
    "warning",
    "danger",
]