
# 🩺 Diabetic Health Recommendation System Using Neuro‑Fuzzy Logic

An intelligent recommendation system built with Neuro‑Fuzzy Logic designed to provide personalized health guidance for diabetic patients based on key health indicators.

---

## 📌 Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Prerequisites](#prerequisites)  
5. [Installation & Setup](#installation--setup)  
6. [Usage](#usage)  
7. [Neuro‑Fuzzy Logic](#neuro‑fuzzy-logic)  
8. [Results & Visualization](#results--visualization)  
9. [Contributing](#contributing)  
10. [Support](#support)

---

## Overview

This project uses a Neuro‑Fuzzy inference system to analyze diabetic patient data—such as blood sugar, BMI, age, and blood pressure—and output tailored health recommendations (e.g., diet adjustments, exercise tips, medication reminders). It combines the interpretability of fuzzy logic with the adaptability of neural networks for nuanced decision-making.

---

## Features

- ✅ Ingestion of patient data (e.g. blood glucose levels, BMI, age, blood pressure)  
- ✅ Neuro‑Fuzzy inference engine to produce personalized recommendations  
- ✅ Easy-to-customize fuzzy membership functions and rule base  
- ✅ Visual outputs for understanding input-to-output mappings  
- ✅ Modular design to facilitate extensions (e.g., add more health variables or refine fuzzy rules)

---

## Project Structure

```text
├── data/                            # Sample input data (CSV format)
├── neuro_fuzzy_system.py           # Defines Neuro‑Fuzzy architecture & engine
├── train.py                        # Optional training/adaptation scripts
├── recommend.py                    # Main inference module for generating outputs
├── visualize.py                    # Plotting / visualization of inference results
├── requirements.txt                # Python dependencies
├── main.py                         # End‑to‑end pipeline execution script
└── README.md                       # This file
```

---

## Prerequisites

Ensure you have the following installed:

- Python 3.8+  
- pip (Python package manager)  
- Recommended libraries: `numpy`, `scikit-fuzzy`, `scikit-learn`, `matplotlib`, `pandas`

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Tanishq123467658/Diabetic-Person-Health-Recommendation-System-Using-Neuro-Fuzzy-Logic.git
   cd Diabetic-Person-Health-Recommendation-System-Using-Neuro-Fuzzy-Logic
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

- **Run the full pipeline**  
  ```bash
  python main.py
  ```
  This will:
  1. Load sample data from `data/`
  2. Feed inputs into the Neuro‑Fuzzy system
  3. Output personalized recommendations
  4. Generate visual plots in `results/`

- **Customize membership functions or fuzzy rules**  
  Modify `neuro_fuzzy_system.py` to adjust fuzzy sets and rule base for blood sugar, BMI, and other inputs.

- **Generate visualizations separately**  
  ```bash
  python visualize.py
  ```

---

## Neuro‑Fuzzy Logic

- **Fuzzy Logic** offers human-like reasoning via linguistic rules (e.g., “IF blood sugar is high AND BMI is overweight THEN diet recommendation is strict”).
- **Neural adaptation** allows continuous tuning of membership functions based on data patterns.
- **Combined system** yields both transparency and flexibility—a powerful choice for health-related inference.

---

## Results & Visualization

Upon execution:
- **Embedded charts** display input values vs. output recommendations.
- **System outputs**, e.g.:
  ```
  Patient 1 → Blood Sugar: 180 mg/dL, BMI: 28 → Recommendation: “Strict diet + 45 min cardio daily”
  ```
- **Visual plots** saved in `results/` for reference and further analysis.

---

## Contributing

Contributions and improvements are welcome! Some ideas:
- Extend dataset with real-world patient data 
- Add more fuzzy variables (e.g., exercise habits, medication compliance)
- Integrate machine learning modules or mobile/web interfaces
- Enhance visualizations with dashboards (e.g., Plotly, Dash)

To contribute:
1. Fork the repository  
2. Create your feature branch  
3. Make changes and test  
4. Submit a pull request!

---

## Support

For questions, bugs, or feedback, please open an issue or contact **Tanishq** via GitHub.

---
