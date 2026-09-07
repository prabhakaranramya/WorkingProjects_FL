import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from fuzzy_engine.fire import (
    fire_fis,
    fire_risk,
    safe as fire_safe,
    warning as fire_warning,
    danger as fire_danger
)

from fuzzy_engine.gas import (
    gas_fis,
    gas_risk,
    gas_safe,
    gas_warning,
    gas_danger
)

from fuzzy_engine.flood import (
    flood_fis,
    flood_risk,
    flood_safe,
    flood_warning,
    flood_danger
)

from fuzzy_engine.intrusion import (
    intrusion_fis,
    intrusion_risk,
    intrusion_safe,
    intrusion_warning,
    intrusion_danger
)

from fuzzy_engine.fusion import (
    fusion_fis,
    fusion_risk,
    fusion_safe,
    fusion_warning,
    fusion_danger
)


######## PAGE CONFIGURATION ########

st.set_page_config(
    page_title="SHIELD Smart Home System",
    page_icon="🛡️",
    layout="wide"
)


#Css File for styling the page

def load_css():

    with open("home.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css()


#System Active Flag.
# Streamlit re-executes this script top-to-bottom on every interaction, so
# "infinitely running until the user decides to stop" is represented here
# with a persistent session_state flag: the dashboard keeps accepting runs
# indefinitely until the user clicks "Stop SHIELD System", at which point
# the script halts (st.stop()) before rendering the rest of the page. A
# "Restart" button flips the flag back and resumes normal operation.

if "system_active" not in st.session_state:

    st.session_state.system_active = True

with st.sidebar:

    st.subheader(" System Control")

    if st.session_state.system_active:

        st.caption(
            "SHIELD is running. It will keep accepting new readings "
            "until you stop it below."
        )

        if st.button(" Stop SHIELD System", use_container_width=True):

            st.session_state.system_active = False

            st.rerun()

    else:

        st.warning("SHIELD has been stopped.")

        if st.button(" Restart SHIELD System", use_container_width=True):

            st.session_state.system_active = True

            st.rerun()

if not st.session_state.system_active:

    st.error(" SHIELD System has been stopped by the user.")

    st.info("Click **Restart SHIELD System** in the sidebar to resume monitoring.")

    st.stop()


#Title and description

st.markdown(
    "<h1>🛡️ SHIELD Smart Home Safety System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center; font-size:18px;'>"
    "Mamdani Fuzzy Inference System for real-time Fire, Gas, Flood &amp; Intrusion risk detection"
    "</p>",
    unsafe_allow_html=True
)

st.write("")


#Shared constants for all four systems: valid tiers, output validation, and plotting utilities.


SMALL_FIGSIZE = (3.2, 2.4)

TEAL = "#17a2b8"
NAVY = "#0b1f3a"

PLOT_COLORS = ["#1e9e6b", "#c98a12", "#d64545"]


def _style_small_axes(ax, title, xlabel):

    ax.set_title(title, fontsize=9, color=NAVY, fontweight="bold")

    ax.set_xlabel(xlabel, fontsize=8)

    ax.set_ylabel("Membership", fontsize=8)

    ax.set_ylim(0, 1.1)

    ax.tick_params(axis="both", labelsize=7)

    ax.grid(linewidth=0.4, alpha=0.5)

    ax.legend(fontsize=6, loc="upper right")

    for spine in ["top", "right"]:

        ax.spines[spine].set_visible(False)


def plot_membership(sensor, value, x_max, shape="trapezoid"):

    x = np.arange(0, x_max + 0.1, 0.1)

    if shape == "triangle":

        low = fuzz.trimf(
            x,
            [0, 0, x.max() * 0.4]
        )

        medium = fuzz.trimf(
            x,
            [x.max() * 0.2, x.max() * 0.5, x.max() * 0.8]
        )

        high = fuzz.trimf(
            x,
            [x.max() * 0.6, x.max(), x.max()]
        )

    else:

        low = fuzz.trapmf(
            x,
            [0, 0, x.max() * 0.2, x.max() * 0.4]
        )

        medium = fuzz.trapmf(
            x,
            [x.max() * 0.2, x.max() * 0.4, x.max() * 0.6, x.max() * 0.8]
        )

        high = fuzz.trapmf(
            x,
            [x.max() * 0.6, x.max() * 0.8, x.max(), x.max()]
        )

    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)

    ax.plot(x, low, label="Low", linewidth=1.4, color=PLOT_COLORS[0])

    ax.plot(x, medium, label="Medium", linewidth=1.4, color=PLOT_COLORS[1])

    ax.plot(x, high, label="High", linewidth=1.4, color=PLOT_COLORS[2])

    ax.axvline(value, linestyle="--", color=NAVY, linewidth=1.1, label=f"Input {value}")

    _style_small_axes(ax, sensor, sensor)

    fig.tight_layout()

    return fig


def plot_output(risk_axis, safe_curve, warning_curve, danger_curve, score, title):

    fig, ax = plt.subplots(figsize=SMALL_FIGSIZE)

    ax.plot(risk_axis, safe_curve, label="Safe", linewidth=1.4, color=PLOT_COLORS[0])

    ax.plot(risk_axis, warning_curve, label="Warning", linewidth=1.4, color=PLOT_COLORS[1])

    ax.plot(risk_axis, danger_curve, label="Danger", linewidth=1.4, color=PLOT_COLORS[2])

    ax.axvline(score, linestyle="--", color=NAVY, linewidth=1.1, label=f"Score {score}")

    _style_small_axes(ax, title, "Risk Score")

    fig.tight_layout()

    return fig


def render_decision(decision):

    if "SAFE" in decision:

        st.success(f" {decision}")

    elif "WARNING" in decision:

        st.warning(f" {decision}")

    else:

        st.error(f" {decision}")


def render_rules(rules, empty_message="No rules activated at these input levels."):

    with st.expander(f" Activated Rules ({len(rules)})", expanded=False):

        if rules:

            for rule in rules:

                st.caption(rule)

        else:

            st.caption(empty_message)


def validate_float_input(raw_text, label, min_value, max_value):


    text = "" if raw_text is None else str(raw_text).strip()

    if text == "":

        return None, f"**{label}** is empty. Enter a number between {min_value} and {max_value}."

    try:

        value = float(text)

    except ValueError:

        return None, f"**{label}**: '{text}' is not a valid number."

    if value < min_value or value > max_value:

        return None, (
            f"**{label}**: {value} is out of range. "
            f"Valid range is {min_value} to {max_value}."
        )

    return value, None


#SENSOR INPUTS - ALL SYSTEMS ON ONE SCREEN. 


st.subheader(" Sensor Inputs")

input_mode = st.radio(
    "Input Mode",
    ["Slider Mode", "Manual Entry Mode"],
    horizontal=True,
    key="input_mode",
    help=(
        "Manual Entry Mode accepts typed values and demonstrates input "
        "validation: empty fields, non-numeric text, and out-of-range "
        "values are all caught and reported so you can correct them."
    ),
)

manual_mode = input_mode == "Manual Entry Mode"

input_col1, input_col2, input_col3, input_col4 = st.columns(4)

with input_col1:

    with st.container(border=True):

        st.markdown(" **Fire Detection**")

        if manual_mode:

            smoke_raw = st.text_input("Smoke Level (0.0-10.0)", value="0.0", key="smoke_txt")

            photo_raw = st.text_input("Photoelectric Level (0.0-10.0)", value="0.0", key="photo_txt")

            gas_val_raw = st.text_input("Gas Level % LEL (0.0-50.0)", value="0.0", key="gas_fire_txt")

        else:

            smoke = st.slider("Smoke Level", 0.0, 10.0, 0.0, 0.1, key="smoke")

            photo = st.slider("Photoelectric Level", 0.0, 10.0, 0.0, 0.1, key="photo")

            gas_val = st.slider("Gas Level (% LEL)", 0.0, 50.0, 0.0, 0.5, key="gas_fire")

with input_col2:

    with st.container(border=True):

        st.markdown(" **Gas Leakage**")

        if manual_mode:

            gas_val2_raw = st.text_input("Gas Level % LEL (0.0-50.0)", value="0.0", key="gas_leak_txt")

            smoke2_raw = st.text_input("Smoke Level (0.0-10.0)", value="0.0", key="smoke_gas_txt")

        else:

            gas_val2 = st.slider("Gas Level (% LEL)", 0.0, 50.0, 0.0, 0.5, key="gas_leak")

            smoke2 = st.slider("Smoke Level", 0.0, 10.0, 0.0, 0.1, key="smoke_gas")

with input_col3:

    with st.container(border=True):

        st.markdown(" **Flood Detection**")

        if manual_mode:

            displacement_raw = st.text_input(
                "Water Displacement Level (0.0-100.0)", value="0.0", key="displacement_flood_txt"
            )

        else:

            displacement = st.slider("Water Displacement Level", 0.0, 100.0, 0.0, 0.5, key="displacement_flood")

with input_col4:

    with st.container(border=True):

        st.markdown(" **Intrusion Detection**")

        if manual_mode:

            motion_raw = st.text_input("Motion Level (0.0-10.0)", value="0.0", key="motion_txt")

            voice_raw = st.text_input("Voice Level (0.0-10.0)", value="0.0", key="voice_txt")

            displacement_i_raw = st.text_input(
                "Displacement Level (0.0-100.0)", value="0.0", key="displacement_intrusion_txt"
            )

        else:

            motion = st.slider("Motion Level", 0.0, 10.0, 0.0, 0.1, key="motion")

            voice = st.slider("Voice Level", 0.0, 10.0, 0.0, 0.1, key="voice")

            displacement_i = st.slider("Displacement Level", 0.0, 100.0, 0.0, 0.5, key="displacement_intrusion")


st.write("")

_, button_col, _ = st.columns([1, 1, 1])

with button_col:

    run_clicked = st.button(" Run SHIELD System", use_container_width=True)


# Final all systems.


proceed = run_clicked

if run_clicked and manual_mode:

    # Validate every manually-typed field before doing any fuzzy inference.
    # If anything is invalid, report every problem at once (rather than
    # stopping at the first one) so the user can fix them all in one pass,
    # then click "Run SHIELD System" again.

    fields_to_validate = [
        ("smoke_raw", "Smoke Level (Fire)", 0.0, 10.0),
        ("photo_raw", "Photoelectric Level", 0.0, 10.0),
        ("gas_val_raw", "Gas Level - Fire (% LEL)", 0.0, 50.0),
        ("gas_val2_raw", "Gas Level - Gas Leakage (% LEL)", 0.0, 50.0),
        ("smoke2_raw", "Smoke Level - Gas Leakage", 0.0, 10.0),
        ("displacement_raw", "Water Displacement - Flood", 0.0, 100.0),
        ("motion_raw", "Motion Level", 0.0, 10.0),
        ("voice_raw", "Voice Level", 0.0, 10.0),
        ("displacement_i_raw", "Displacement - Intrusion", 0.0, 100.0),
    ]

    validation_errors = []

    validated = {}

    for var_name, label, lo, hi in fields_to_validate:

        value, error = validate_float_input(locals()[var_name], label, lo, hi)

        if error:

            validation_errors.append(error)

        else:

            validated[var_name] = value

    if validation_errors:

        st.error(
            f" {len(validation_errors)} invalid input(s) detected. "
            "Please correct the fields below and click **Run SHIELD System** again."
        )

        for message in validation_errors:

            st.write(f"- {message}")

        proceed = False

    else:

        # All fields validated successfully - bind the plain variable names
        # used by the rest of the script to the parsed float values.
        smoke = validated["smoke_raw"]
        photo = validated["photo_raw"]
        gas_val = validated["gas_val_raw"]
        gas_val2 = validated["gas_val2_raw"]
        smoke2 = validated["smoke2_raw"]
        displacement = validated["displacement_raw"]
        motion = validated["motion_raw"]
        voice = validated["voice_raw"]
        displacement_i = validated["displacement_i_raw"]

        st.success(" All manual inputs validated successfully.")


fis_error = None

if proceed:

    # All five FIS calls are wrapped together: if any subsystem raises an
    # unexpected error (malformed input slipping past validation, a bug in
    # a rule base, etc.) the user sees a clear message and can adjust their
    # inputs and try again, instead of the app crashing with a raw
    # traceback.

    try:

        fire_score, fire_decision, fire_rules, fire_output = fire_fis(smoke, photo, gas_val)

        gas_score, gas_decision, gas_rules, gas_output = gas_fis(gas_val2, smoke2)

        flood_score, flood_decision, flood_rules, flood_output = flood_fis(displacement)

        intrusion_score, intrusion_decision, intrusion_rules, intrusion_output = intrusion_fis(
            motion, voice, displacement_i
        )

        ######## HIERARCHICAL FUSION LAYER ########
        # Second-tier Mamdani FIS: fuses the four subsystem scores into one
        # Overall Home Risk score. See fuzzy_engine/fusion.py for the design
        # rationale (max_score / avg_score feature reduction + 9-rule base).

        fusion_score, fusion_decision, fusion_rules, fusion_output, fusion_features = fusion_fis(
            fire_score, gas_score, flood_score, intrusion_score
        )

    except Exception as exc:

        fis_error = exc

if fis_error is not None:

    st.error(f" An unexpected error occurred while evaluating the fuzzy inference systems: {fis_error}")

    st.info("Please check your sensor values and click **Run SHIELD System** again.")

elif proceed:

    ######## OVERALL RISK HERO CARD ########

    st.divider()

    st.subheader(" Overall Home Risk ")

    hero_col1, hero_col2 = st.columns([1, 2])

    with hero_col1:

        st.metric("Overall Risk Score", f"{fusion_score}/100")

        render_decision(fusion_decision)

        st.caption(
            f"Derived from worst subsystem (max = {fusion_features['max_score']}) "
            f"and combined load (avg = {fusion_features['avg_score']})"
        )

    with hero_col2:

        st.pyplot(
            plot_output(fusion_risk, fusion_safe, fusion_warning, fusion_danger, fusion_score, "Overall Risk Output"),
            use_container_width=False
        )

        render_rules(fusion_rules)

    ######## SUMMARY ROW ########

    st.divider()

    st.subheader(" SHIELD Status Overview")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:

        st.markdown(" **Fire Risk**")

        st.metric("Risk Score", f"{fire_score}/100")

        render_decision(fire_decision)

    with summary_col2:

        st.markdown(" **Gas Leakage Risk**")

        st.metric("Risk Score", f"{gas_score}/100")

        render_decision(gas_decision)

    with summary_col3:

        st.markdown(" **Flood Risk**")

        st.metric("Risk Score", f"{flood_score}/100")

        render_decision(flood_decision)

    with summary_col4:

        st.markdown(" **Intrusion Risk**")

        st.metric("Risk Score", f"{intrusion_score}/100")

        render_decision(intrusion_decision)

    st.divider()

    ######## FIRE DETAIL ########

    st.markdown("####  Fire Detection")

    fire_text_col, fire_graph_col = st.columns([1, 3])

    with fire_text_col:

        st.write({"Smoke": smoke, "Photoelectric": photo, "Gas": gas_val})

        render_rules(fire_rules)

    with fire_graph_col:

        g1, g2, g3, g4 = st.columns(4)

        with g1:

            st.pyplot(plot_membership("Smoke", smoke, 10.0), use_container_width=False)

        with g2:

            st.pyplot(plot_membership("Photoelectric", photo, 10.0), use_container_width=False)

        with g3:

            st.pyplot(plot_membership("Gas", gas_val, 50.0), use_container_width=False)

        with g4:

            st.pyplot(
                plot_output(fire_risk, fire_safe, fire_warning, fire_danger, fire_score, "Fire Output"),
                use_container_width=False
            )

    st.divider()

    ######## GAS DETAIL ########

    st.markdown("####  Gas Leakage")

    gas_text_col, gas_graph_col = st.columns([1, 3])

    with gas_text_col:

        st.write({"Gas": gas_val2, "Smoke": smoke2})

        render_rules(gas_rules)

    with gas_graph_col:

        g1, g2, g3 = st.columns(3)

        with g1:

            st.pyplot(plot_membership("Gas", gas_val2, 50.0), use_container_width=False)

        with g2:

            st.pyplot(plot_membership("Smoke", smoke2, 10.0), use_container_width=False)

        with g3:

            st.pyplot(
                plot_output(gas_risk, gas_safe, gas_warning, gas_danger, gas_score, "Gas Output"),
                use_container_width=False
            )

    st.divider()

    #Flood Detail

    st.markdown("####  Flood Detection")

    flood_text_col, flood_graph_col = st.columns([1, 3])

    with flood_text_col:

        st.write({"Displacement": displacement})

        render_rules(flood_rules)

    with flood_graph_col:

        g1, g2 = st.columns(2)

        with g1:

            st.pyplot(
                plot_membership("Displacement", displacement, 100.0, shape="triangle"),
                use_container_width=False
            )

        with g2:

            st.pyplot(
                plot_output(flood_risk, flood_safe, flood_warning, flood_danger, flood_score, "Flood Output"),
                use_container_width=False
            )

    st.divider()

    #Intrusion Detail

    st.markdown("####  Intrusion Detection")

    intrusion_text_col, intrusion_graph_col = st.columns([1, 3])

    with intrusion_text_col:

        st.write({"Motion": motion, "Voice": voice, "Displacement": displacement_i})

        render_rules(intrusion_rules)

    with intrusion_graph_col:

        g1, g2, g3, g4 = st.columns(4)

        with g1:

            st.pyplot(plot_membership("Motion", motion, 10.0), use_container_width=False)

        with g2:

            st.pyplot(plot_membership("Voice", voice, 10.0), use_container_width=False)

        with g3:

            st.pyplot(plot_membership("Displacement", displacement_i, 100.0), use_container_width=False)

        with g4:

            st.pyplot(
                plot_output(
                    intrusion_risk, intrusion_safe, intrusion_warning, intrusion_danger,
                    intrusion_score, "Intrusion Output"
                ),
                use_container_width=False
            )

    st.divider()

    st.markdown(
        "<p style='text-align:center; color:#8a97ad; font-size:13px;'>"
        "SHIELD Smart Home Safety System &middot; Mamdani Fuzzy Inference Engine"
        "</p>",
        unsafe_allow_html=True
    )

else:

    st.info(" Set your sensor values above, then click **Run SHIELD System** to evaluate all four systems.")
