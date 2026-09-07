import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

from fuzzy_engine.fire import fire_fis
from fuzzy_engine.gas import gas_fis
from fuzzy_engine.flood import flood_fis
from fuzzy_engine.intrusion import intrusion_fis

from fuzzy_engine.crisp_baseline import (
    fire_crisp,
    gas_crisp,
    flood_crisp,
    intrusion_crisp,
)


#Page configuration

st.set_page_config(
    page_title="SHIELD - Fuzzy vs Crisp Analysis",
    layout="wide"
)


def load_css():

    with open("home.css") as f:

        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


st.markdown("<h1> Fuzzy vs Crisp Comparison</h1>", unsafe_allow_html=True)

st.markdown(
    "<p style='text-align:center; font-size:17px;'>"
    "Sweeping one sensor input at a time to compare the smooth Mamdani fuzzy "
    "risk score against a plain if/else threshold classifier, holding all "
    "other inputs fixed."
    "</p>",
    unsafe_allow_html=True
)

st.write("")


######## SYSTEM CONFIGURATION ########
# Each system defines: which inputs it needs, the ones the user can fix,
# which one gets swept, its axis range, and the fuzzy/crisp callables.

SYSTEMS = {
    " Fire Detection": {
        "inputs": ["Smoke", "Photoelectric", "Gas (% LEL)"],
        "ranges": [(0.0, 10.0), (0.0, 10.0), (0.0, 50.0)],
        "defaults": [0.0, 0.0, 0.0],
        "fuzzy_fn": lambda vals: fire_fis(vals[0], vals[1], vals[2])[0],
        "crisp_fn": lambda vals: fire_crisp(vals[0], vals[1], vals[2])[0],
    },
    " Gas Leakage": {
        "inputs": ["Gas (% LEL)", "Smoke"],
        "ranges": [(0.0, 50.0), (0.0, 10.0)],
        "defaults": [0.0, 0.0],
        "fuzzy_fn": lambda vals: gas_fis(vals[0], vals[1])[0],
        "crisp_fn": lambda vals: gas_crisp(vals[0], vals[1])[0],
    },
    "  Flood Detection": {
        "inputs": ["Water Displacement"],
        "ranges": [(0.0, 100.0)],
        "defaults": [0.0],
        "fuzzy_fn": lambda vals: flood_fis(vals[0])[0],
        "crisp_fn": lambda vals: flood_crisp(vals[0])[0],
    },
    " Intrusion Detection": {
        "inputs": ["Motion", "Voice", "Displacement"],
        "ranges": [(0.0, 10.0), (0.0, 10.0), (0.0, 100.0)],
        "defaults": [0.0, 0.0, 0.0],
        "fuzzy_fn": lambda vals: intrusion_fis(vals[0], vals[1], vals[2])[0],
        "crisp_fn": lambda vals: intrusion_crisp(vals[0], vals[1], vals[2])[0],
    },
}


######## CONTROLS ########

st.subheader(" Configuration")

config_col1, config_col2 = st.columns([1, 1])

with config_col1:

    system_name = st.selectbox("System to analyse", list(SYSTEMS.keys()))

config = SYSTEMS[system_name]

with config_col2:

    sweep_input_name = st.selectbox("Input to sweep", config["inputs"])

sweep_index = config["inputs"].index(sweep_input_name)

st.markdown("**Fix the remaining inputs:**")

fixed_cols = st.columns(max(len(config["inputs"]) - 1, 1))

fixed_values = list(config["defaults"])

col_pointer = 0

for i, name in enumerate(config["inputs"]):

    if i == sweep_index:

        continue

    lo, hi = config["ranges"][i]

    with fixed_cols[col_pointer]:

        fixed_values[i] = st.slider(name, lo, hi, config["defaults"][i], key=f"fixed_{system_name}_{name}")

    col_pointer += 1


######## SWEEP + PLOT ########

sweep_lo, sweep_hi = config["ranges"][sweep_index]

sweep_x = np.linspace(sweep_lo, sweep_hi, 200)

fuzzy_scores = []

crisp_scores = []

for x in sweep_x:

    vals = list(fixed_values)

    vals[sweep_index] = x

    fuzzy_scores.append(config["fuzzy_fn"](vals))

    crisp_scores.append(config["crisp_fn"](vals))

fig, ax = plt.subplots(figsize=(9, 4.5))

ax.plot(sweep_x, fuzzy_scores, label="Fuzzy (Mamdani) Score", color="#17a2b8", linewidth=2.2)

ax.step(sweep_x, crisp_scores, label="Crisp Threshold Score", color="#d64545", linewidth=2, where="post", linestyle="--")

plot_title = system_name.split(" ", 1)[1] if " " in system_name else system_name

ax.set_title(f"{plot_title} - Risk Score vs {sweep_input_name}", fontsize=12, fontweight="bold", color="#0b1f3a")

ax.set_xlabel(sweep_input_name, fontsize=10)

ax.set_ylabel("Risk Score (0-100)", fontsize=10)

ax.set_ylim(-5, 105)

ax.grid(linewidth=0.4, alpha=0.5)

ax.legend(fontsize=10)

for spine in ["top", "right"]:

    ax.spines[spine].set_visible(False)

fig.tight_layout()

st.pyplot(fig, use_container_width=True)


######## DISCUSSION ########

mismatch = np.mean(np.abs(np.array(fuzzy_scores) - np.array(crisp_scores)))

st.metric("Mean Absolute Difference (Fuzzy vs Crisp)", f"{mismatch:.2f} points")
