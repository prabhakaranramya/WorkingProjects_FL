# 🛡️ SHIELD — Smart Home Safety System

**A Mamdani Fuzzy Inference System for real-time Fire, Gas, Flood & Intrusion risk detection.**

SHIELD simulates a smart home safety controller that takes live sensor readings and produces a continuous, human-interpretable risk score (0–100) for four hazard categories, instead of a blunt on/off alarm. It also includes a built-in analysis tool that compares the fuzzy logic approach against a traditional crisp (if/else threshold) classifier, so you can see exactly where fuzzy logic adds value.

---

## ✨ Features

- **Real-time risk scoring** across four independent detection systems:
  - 🔥 **Fire Detection** — Smoke Level, Photoelectric Level, Gas Level (% LEL)
  - 🧪 **Gas Leakage** — Gas Level (% LEL), Smoke Level
  - 💧 **Flood Detection** — Water Displacement Level
  - 🚨 **Intrusion Detection** — Motion Level, Voice Level, Displacement Level
- **Two input modes:**
  - **Slider Mode** — adjust sensor values interactively with live-updating gauges
  - **Manual Entry Mode** — type exact sensor values for repeatable test cases
- **Fuzzy vs. Crisp Analysis** page — sweeps one sensor input at a time (holding the others fixed) and plots the Mamdani fuzzy risk score against a plain threshold-based crisp score, visualizing the smooth transition fuzzy logic provides versus the sharp cutoff of traditional rule-based systems.
- **System Control panel** to start/stop the SHIELD engine from the sidebar.
- **Deploy-ready** Streamlit app.

---

## 🧠 How It Works

SHIELD uses a **Mamdani-style Fuzzy Inference System (FIS)** for each detection category:

1. **Fuzzification** — raw sensor readings (e.g., smoke level, gas %LEL) are mapped to fuzzy membership degrees (e.g., "Low", "Medium", "High").
2. **Rule Evaluation** — a set of expert-defined IF-THEN rules combine the fuzzified inputs (e.g., *IF Smoke is High AND Gas is Medium THEN Fire Risk is High*).
3. **Aggregation** — the outputs of all fired rules are combined into a single fuzzy output set.
4. **Defuzzification** — the aggregated fuzzy set is converted into a single crisp risk score (0–100) using a method such as centroid (center of gravity).

This produces a smooth, continuous risk score that better reflects real-world uncertainty than a simple threshold check, which can only ever output a hard 0 or 1.

---

## 📁 Project Structure

```
.
├── app.py                    # Main Streamlit entry point
├── conftest.py                # Pytest fixtures/configuration
├── test_fuzzy_systems.py      # Unit tests for the fuzzy inference engines
├── home.css                   # Custom styling for the dashboard
├── fuzzy_engine/               # Core fuzzy logic implementation
│   └── ...                     # Membership functions, rule bases, inference logic
├── Pages/                      # Additional Streamlit pages
│   └── Fuzzy_vs_Crisp_Analysis.py   # Fuzzy vs. crisp comparison tool
└── .vscode/                    # Editor configuration
```

> Update this tree if your actual file layout differs — this reflects the structure visible in the repository.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

> If a `requirements.txt` isn't present yet, generate one with `pip freeze > requirements.txt` after installing your dependencies (likely `streamlit`, `scikit-fuzzy` or a custom fuzzy engine, `numpy`, `matplotlib`/`plotly`, and `pytest`).

### Running the App

```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`.

### Running Tests

```bash
pytest test_fuzzy_systems.py
```

---

## 🖥️ Usage

1. Launch the app and select **Slider Mode** or **Manual Entry Mode** in the sidebar.
2. Adjust sensor values under **Fire Detection**, **Gas Leakage**, **Flood Detection**, and **Intrusion Detection**.
3. Click **Stop SHIELD System** to halt live inference, or **Deploy** to publish the app.
4. Visit the **Fuzzy vs Crisp Analysis** page to:
   - Choose a system to analyze (e.g., Fire Detection)
   - Choose an input to sweep (e.g., Smoke)
   - Fix the remaining inputs using the sliders
   - Compare the fuzzy (Mamdani) score curve against the crisp threshold score curve

---

## 🛠️ Tech Stack

- **Streamlit** — interactive web dashboard
- **Python** — core application and fuzzy inference logic
- **Pytest** — unit testing framework

---

## 📌 Notes

- Risk scores range from **0–100**, where higher values indicate greater hazard risk.
- The crisp threshold model exists purely as a benchmark for comparison and is not intended to drive real alerts.

---

## 📄 License

Add your license here (e.g., MIT, Apache 2.0).
