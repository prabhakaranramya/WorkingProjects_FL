import numpy as np
import skfuzzy as fuzz


######## INPUT RANGES ########

motion_range = np.arange(0, 10.1, 0.1)

voice_range = np.arange(0, 10.1, 0.1)

displacement_range = np.arange(0, 100.1, 0.1)

intrusion_risk = np.arange(0, 101, 1)


######## MOTION MEMBERSHIP ########

motion_low = fuzz.trapmf(motion_range, [0, 0, 2, 4])

motion_medium = fuzz.trapmf(motion_range, [2, 4, 6, 8])

motion_high = fuzz.trapmf(motion_range, [6, 8, 10, 10])


######## VOICE MEMBERSHIP ########

voice_normal = fuzz.trapmf(voice_range, [0, 0, 3, 5])

voice_loud = fuzz.trapmf(voice_range, [4, 6, 7, 8])

voice_scream = fuzz.trapmf(voice_range, [7, 8, 10, 10])


######## DISPLACEMENT MEMBERSHIP ########

displacement_low = fuzz.trapmf(displacement_range, [0, 0, 20, 40])

displacement_medium = fuzz.trapmf(displacement_range, [30, 50, 60, 80])

displacement_high = fuzz.trapmf(displacement_range, [70, 85, 100, 100])


######## INTRUSION RISK OUTPUT MEMBERSHIP ########

intrusion_safe = fuzz.trapmf(intrusion_risk, [0, 0, 20, 40])

intrusion_warning = fuzz.trapmf(intrusion_risk, [30, 40, 60, 70])

intrusion_danger = fuzz.trapmf(intrusion_risk, [60, 80, 100, 100])


######## FUZZIFICATION ########


def fuzzification(motion_value, voice_value, displacement_value):

    motion_dom = {
        "Low": fuzz.interp_membership(motion_range, motion_low, motion_value),
        "Medium": fuzz.interp_membership(motion_range, motion_medium, motion_value),
        "High": fuzz.interp_membership(motion_range, motion_high, motion_value),
    }

    voice_dom = {
        "Normal": fuzz.interp_membership(voice_range, voice_normal, voice_value),
        "Loud": fuzz.interp_membership(voice_range, voice_loud, voice_value),
        "Scream": fuzz.interp_membership(voice_range, voice_scream, voice_value),
    }

    displacement_dom = {
        "Low": fuzz.interp_membership(displacement_range, displacement_low, displacement_value),
        "Medium": fuzz.interp_membership(displacement_range, displacement_medium, displacement_value),
        "High": fuzz.interp_membership(displacement_range, displacement_high, displacement_value),
    }

    return motion_dom, voice_dom, displacement_dom


######## INTRUSION FUZZY INFERENCE SYSTEM ########


def intrusion_fis(motion_value, voice_value, displacement_value):

    motion_dom, voice_dom, displacement_dom = fuzzification(
        motion_value, voice_value, displacement_value
    )

    M_L, M_M, M_H = motion_dom["Low"], motion_dom["Medium"], motion_dom["High"]
    V_N, V_L, V_S = voice_dom["Normal"], voice_dom["Loud"], voice_dom["Scream"]
    D_L, D_M, D_H = displacement_dom["Low"], displacement_dom["Medium"], displacement_dom["High"]

    ######## RULE BASE (10 RULES) ########
    #
    # I1-I9 cover Motion-alone, Voice-alone, and Displacement-alone triggers
    # plus a few explicit AND-combinations. That left one gap: whenever
    # Motion is Low, Voice is Normal, and Displacement sits in its Medium
    # band (not Low, not High), none of I1-I9 fired at all, the aggregated
    # output was entirely zero, and the system silently defaulted to
    # SAFE - even though a moderate, otherwise-unexplained displacement
    # reading is exactly the kind of ambiguous signal an intrusion system
    # should flag rather than ignore. I10 closes that gap the same way I6
    # already treats Displacement High as a standalone Danger trigger -
    # Displacement Medium alone is treated as a standalone Warning trigger.

    rules = {
        "I1":  (min(M_L, V_N, D_L), "Safe",    "Motion Low AND Voice Normal AND Displacement Low"),
        "I2":  (min(M_M, V_N),      "Warning", "Motion Medium AND Voice Normal"),
        "I3":  (M_H,                "Danger",  "Motion High"),
        "I4":  (V_L,                "Warning", "Voice Loud"),
        "I5":  (V_S,                "Danger",  "Voice Scream"),
        "I6":  (D_H,                "Danger",  "Displacement High"),
        "I7":  (min(M_H, V_L),      "Danger",  "Motion High AND Voice Loud"),
        "I10": (D_M,                "Warning", "Displacement Medium"),
        "I8": (min(M_H, D_H),      "Danger",  "Motion High AND Displacement High"),
        "I9": (min(M_H, V_S, D_H), "Danger",  "Motion High AND Voice Scream AND Displacement High"),
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

    safe_output = np.fmin(safe_strength, intrusion_safe)

    warning_output = np.fmin(warning_strength, intrusion_warning)

    danger_output = np.fmin(danger_strength, intrusion_danger)

    ######## AGGREGATION ########

    output = np.fmax(safe_output, np.fmax(warning_output, danger_output))

    ######## DEFUZZIFICATION ########

    if np.sum(output) == 0:

        score = 0

    else:

        score = fuzz.defuzz(intrusion_risk, output, "centroid")

    score = round(float(score), 2)

    ######## DECISION ########

    if score <= 30:

        decision = "SAFE - No Intrusion Detected"

    elif score <= 65:

        decision = "WARNING - Possible Intrusion"

    else:

        decision = "DANGER - Intrusion Alert"

    return score, decision, fired_rules, output


######## EXPORT FOR APP.PY ########

__all__ = [
    "intrusion_fis",
    "intrusion_risk",
    "intrusion_safe",
    "intrusion_warning",
    "intrusion_danger",
]
