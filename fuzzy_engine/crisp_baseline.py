"""Non-fuzzy baseline for the Fuzzy Engine."""

def _tier_to_score(tier):
    """Convert a tier to a score."""
    if tier == "Safe":
        return 0
    elif tier == "Warning":
        return 50
    elif tier == "Danger":
        return 100
    else:
        raise ValueError(f"Unknown tier: {tier}")


"""Fire"""
def fire_crisp(smoke_value, photo_value, gas_value):
    if smoke_value >= 6 or photo_value >= 6 or gas_value >= 30:

        tier = "Danger"

    elif smoke_value >= 2 or photo_value >= 2 or gas_value >= 10:

        tier = "Warning"

    else:

        tier = "Safe"

    return _tier_to_score(tier), tier

# GAS (crisp) 


def gas_crisp(gas_value, smoke_value):

    if gas_value >= 30 or smoke_value >= 6:

        tier = "Danger"

    elif gas_value >= 10 or smoke_value >= 2:

        tier = "Warning"

    else:

        tier = "Safe"

    return _tier_to_score(tier), tier


#FLOOD (crisp)


def flood_crisp(displacement_value):

    if displacement_value >= 65:

        tier = "Danger"

    elif displacement_value >= 30:

        tier = "Warning"

    else:

        tier = "Safe"

    return _tier_to_score(tier), tier


#INTRUSION (crisp)


def intrusion_crisp(motion_value, voice_value, displacement_value):

    if motion_value >= 6 or voice_value >= 7 or displacement_value >= 70:

        tier = "Danger"

    elif motion_value >= 2 or voice_value >= 4 or displacement_value >= 30:

        tier = "Warning"

    else:

        tier = "Safe"

    return _tier_to_score(tier), tier


#ANALYSIS PAGE 

__all__ = [
    "fire_crisp",
    "gas_crisp",
    "flood_crisp",
    "intrusion_crisp",
]
