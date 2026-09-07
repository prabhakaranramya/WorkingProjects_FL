import itertools

import pytest

from fuzzy_engine.fire import fire_fis
from fuzzy_engine.gas import gas_fis
from fuzzy_engine.flood import flood_fis
from fuzzy_engine.intrusion import intrusion_fis
from fuzzy_engine.fusion import fusion_fis

from fuzzy_engine.crisp_baseline import (
    fire_crisp,
    gas_crisp,
    flood_crisp,
    intrusion_crisp,
)


VALID_TIERS = ("SAFE", "WARNING", "DANGER")


def _assert_valid_output(score, decision, rules):

    assert 0 <= score <= 100

    assert any(decision.startswith(tier) for tier in VALID_TIERS)

    assert isinstance(rules, list)

    for rule in rules:

        assert isinstance(rule, str)

        assert "Strength" in rule


#Fire, Gas, Flood, Intrusion system configurations for the Fuzzy vs Crisp Analysis page

FIRE_BOUNDARY_INPUTS = [
    (0.0, 0.0, 0.0),      # all minimum
    (5.0, 5.0, 25.0),     # all midpoint
    (10.0, 10.0, 50.0),   # all maximum
    (10.0, 0.0, 0.0),     # single extreme sensor
]


@pytest.mark.parametrize("smoke,photo,gas", FIRE_BOUNDARY_INPUTS)
def test_fire_fis_boundaries(smoke, photo, gas):

    score, decision, rules, output = fire_fis(smoke, photo, gas)

    _assert_valid_output(score, decision, rules)


def test_fire_fis_all_zero_is_safe():

    score, decision, rules, output = fire_fis(0.0, 0.0, 0.0)

    assert decision.startswith("SAFE")

    assert score <= 30


def test_fire_fis_all_max_is_danger():

    score, decision, rules, output = fire_fis(10.0, 10.0, 50.0)

    assert decision.startswith("DANGER")

    assert score > 65


def test_fire_fis_monotonic_smoke_increase():

    low_score, _, _, _ = fire_fis(0.0, 0.0, 0.0)

    high_score, _, _, _ = fire_fis(10.0, 0.0, 0.0)

    assert high_score >= low_score


#Rule base

FIRE_CATEGORY_POINTS = {
    "smoke": [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0],
    "photo": [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0],
    "gas": [0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 50.0],
}


@pytest.mark.parametrize(
    "smoke,photo,gas",
    list(itertools.product(
        FIRE_CATEGORY_POINTS["smoke"],
        FIRE_CATEGORY_POINTS["photo"],
        FIRE_CATEGORY_POINTS["gas"],
    )),
)
def test_fire_fis_no_silent_rule_gaps(smoke, photo, gas):

    score, decision, rules, output = fire_fis(smoke, photo, gas)

    if smoke <= 1.0 and photo <= 1.0 and gas <= 5.0:

        return  # genuine near-zero "nothing happening" case, skip

    assert len(rules) > 0, (
        f"No rule fired for (Smoke={smoke}, Photo={photo}, Gas={gas}); "
        "this input combination is falling through the rule base and "
        "silently defaulting to score=0 / SAFE."
    )


#Gas Boundary Inputs for testing

GAS_BOUNDARY_INPUTS = [
    (0.0, 0.0),
    (25.0, 5.0),
    (50.0, 10.0),
]


@pytest.mark.parametrize("gas_val,smoke_val", GAS_BOUNDARY_INPUTS)
def test_gas_fis_boundaries(gas_val, smoke_val):

    score, decision, rules, output = gas_fis(gas_val, smoke_val)

    _assert_valid_output(score, decision, rules)


def test_gas_fis_all_zero_is_safe():

    score, decision, rules, output = gas_fis(0.0, 0.0)

    assert decision.startswith("SAFE")


def test_gas_fis_all_max_is_danger():

    score, decision, rules, output = gas_fis(50.0, 10.0)

    assert decision.startswith("DANGER")


#Flood Boundary Inputs for testing

FLOOD_BOUNDARY_INPUTS = [0.0, 25.0, 50.0, 75.0, 100.0]


@pytest.mark.parametrize("displacement", FLOOD_BOUNDARY_INPUTS)
def test_flood_fis_boundaries(displacement):

    score, decision, rules, output = flood_fis(displacement)

    _assert_valid_output(score, decision, rules)


def test_flood_fis_zero_is_safe():

    score, decision, rules, output = flood_fis(0.0)

    assert decision.startswith("SAFE")


def test_flood_fis_max_is_danger():

    score, decision, rules, output = flood_fis(100.0)

    assert decision.startswith("DANGER")


def test_flood_fis_monotonic():

    scores = [flood_fis(d)[0] for d in [0, 20, 40, 60, 80, 100]]

    # Overall trend across the sweep should be non-decreasing at the
    # endpoints even though individual midpoints may plateau.
    assert scores[0] <= scores[-1]


#Intrusion Boundary Inputs for testing

INTRUSION_BOUNDARY_INPUTS = [
    (0.0, 0.0, 0.0),
    (5.0, 5.0, 50.0),
    (10.0, 10.0, 100.0),
]


@pytest.mark.parametrize("motion,voice,displacement", INTRUSION_BOUNDARY_INPUTS)
def test_intrusion_fis_boundaries(motion, voice, displacement):

    score, decision, rules, output = intrusion_fis(motion, voice, displacement)

    _assert_valid_output(score, decision, rules)


def test_intrusion_fis_all_zero_is_safe():

    score, decision, rules, output = intrusion_fis(0.0, 0.0, 0.0)

    assert decision.startswith("SAFE")


def test_intrusion_fis_all_max_is_danger():

    score, decision, rules, output = intrusion_fis(10.0, 10.0, 100.0)

    assert decision.startswith("DANGER")


#Intrsion rule base

INTRUSION_CATEGORY_POINTS = {
    "motion": [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0],
    "voice": [0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0],
    "displacement": [0.0, 10.0, 30.0, 50.0, 70.0, 90.0, 100.0],
}


@pytest.mark.parametrize(
    "motion,voice,displacement",
    list(itertools.product(
        INTRUSION_CATEGORY_POINTS["motion"],
        INTRUSION_CATEGORY_POINTS["voice"],
        INTRUSION_CATEGORY_POINTS["displacement"],
    )),
)
def test_intrusion_fis_no_silent_rule_gaps(motion, voice, displacement):

    score, decision, rules, output = intrusion_fis(motion, voice, displacement)

    if motion <= 1.0 and voice <= 1.0 and displacement <= 10.0:

        return  # genuine near-zero "nothing happening" case, skip

    assert len(rules) > 0, (
        f"No rule fired for (Motion={motion}, Voice={voice}, "
        f"Displacement={displacement}); this input combination is falling "
        "through the rule base and silently defaulting to score=0 / SAFE."
    )


#Fusion(all systems) system tests


def test_fusion_all_safe_scores_is_safe():

    score, decision, rules, output, features = fusion_fis(5, 5, 5, 5)

    assert decision.startswith("SAFE")

    assert features["max_score"] == 5

    assert features["avg_score"] == 5


def test_fusion_all_danger_scores_is_danger():

    score, decision, rules, output, features = fusion_fis(90, 90, 90, 90)

    assert decision.startswith("DANGER")


def test_fusion_single_extreme_subsystem_dominates():
    # One subsystem at maximum danger, the rest safe: overall risk should
    # still be pulled up meaningfully by the "max_score" feature, not diluted
    # away to Safe by averaging.

    score, decision, rules, output, features = fusion_fis(95, 0, 0, 0)

    assert features["max_score"] == 95

    assert not decision.startswith("SAFE")


def test_fusion_output_contract():

    score, decision, rules, output, features = fusion_fis(40, 60, 20, 80)

    _assert_valid_output(score, decision, rules)

    assert "max_score" in features and "avg_score" in features


#Crisp Baseline system tests


def test_fire_crisp_matches_expected_tiers():

    assert fire_crisp(0.0, 0.0, 0.0)[1] == "Safe"

    assert fire_crisp(10.0, 10.0, 50.0)[1] == "Danger"


def test_gas_crisp_matches_expected_tiers():

    assert gas_crisp(0.0, 0.0)[1] == "Safe"

    assert gas_crisp(50.0, 10.0)[1] == "Danger"


def test_flood_crisp_matches_expected_tiers():

    assert flood_crisp(0.0)[1] == "Safe"

    assert flood_crisp(100.0)[1] == "Danger"


def test_intrusion_crisp_matches_expected_tiers():

    assert intrusion_crisp(0.0, 0.0, 0.0)[1] == "Safe"

    assert intrusion_crisp(10.0, 10.0, 100.0)[1] == "Danger"


@pytest.mark.parametrize(
    "crisp_fn,args",
    [
        (fire_crisp, (5.0, 5.0, 25.0)),
        (gas_crisp, (25.0, 5.0)),
        (flood_crisp, (50.0,)),
        (intrusion_crisp, (5.0, 5.0, 50.0)),
    ],
)
def test_crisp_scores_are_step_valued(crisp_fn, args):

    score, tier = crisp_fn(*args)

    assert score in (0, 50, 100)

    assert tier in ("Safe", "Warning", "Danger")
