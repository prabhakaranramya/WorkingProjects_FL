import numpy as np
import skfuzzy as fuzz

# Input ranges

displacement_range = np.arange(0, 100.1, 0.1)

flood_risk = np.arange(0, 101, 1)


######## DISPLACEMENT MEMBERSHIP (TRIANGULAR) ########

displacement_low = fuzz.trimf(displacement_range, [0, 0, 40])

displacement_medium = fuzz.trimf(displacement_range, [20, 50, 80])

displacement_high = fuzz.trimf(displacement_range, [60, 100, 100])


######## FLOOD OUTPUT MEMBERSHIP ########

flood_safe = fuzz.trapmf(flood_risk, [0, 0, 20, 40])

flood_warning = fuzz.trapmf(flood_risk, [30, 40, 60, 70])

flood_danger = fuzz.trapmf(flood_risk, [60, 80, 100, 100])

# Fuzzification

def fuzzification(displacement_value):

    displacement_dom = {
        "Low": fuzz.interp_membership(displacement_range, displacement_low, displacement_value),
        "Medium": fuzz.interp_membership(displacement_range, displacement_medium, displacement_value),
        "High": fuzz.interp_membership(displacement_range, displacement_high, displacement_value),
    }

    return displacement_dom

# Flood Fuzzy Inference System

def flood_fis(displacement_value):

    displacement_dom = fuzzification(displacement_value)

    FL1 = displacement_dom["Low"]

    FL2 = displacement_dom["Medium"]

    FL3 = displacement_dom["High"]

    rules = {
        "FL1": FL1,
        "FL2": FL2,
        "FL3": FL3,
    }

    rule_names = {
        "FL1": "Displacement Low -> Safe",
        "FL2": "Displacement Medium -> Warning",
        "FL3": "Displacement High -> Danger",
    }

    ######## FIRED RULES ########

    fired_rules = []

    for rule, strength in rules.items():

        if strength > 0:

            fired_rules.append(
                f"{rule}: {rule_names[rule]} | Strength {round(float(strength), 2)}"
            )

    # Mamdani Output 

    safe_output = np.fmin(FL1, flood_safe)

    warning_output = np.fmin(FL2, flood_warning)

    danger_output = np.fmin(FL3, flood_danger)

    output = np.fmax(safe_output, np.fmax(warning_output, danger_output))

    # Defuzzification

    if np.sum(output) == 0:

        score = 0

    else:

        score = fuzz.defuzz(flood_risk, output, "centroid")

    score = round(float(score), 2)

    # Decision

    if score <= 30:

        decision = "SAFE - Normal Water Level"

    elif score <= 65:

        decision = "WARNING - Monitor Flood Level"

    else:

        decision = "DANGER - Flood Alert"

    return score, decision, fired_rules, output

# Export to app.py

__all__ = [
    "flood_fis",
    "flood_risk",
    "flood_safe",
    "flood_warning",
    "flood_danger",
]