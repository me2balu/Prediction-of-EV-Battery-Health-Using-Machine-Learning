# 🔋 Prediction of EV-Battery Health Using Machine Learning

## 📌 Project Overview

This project uses **Machine Learning** to estimate the **State of Health (SoH)** of lithium-ion batteries from discharge-cycle sensor measurements.

The aim is to predict remaining battery capacity accurately and make the predictions understandable through **Explainable AI (SHAP)**. The project combines Savitzky–Golay signal filtering, a two-level stacked ensemble, and an interactive **Gradio dashboard**.

The implementation focuses on NASA battery aging data and tests both **later cycles of the same battery** and **previously unseen battery cells**.

## 🎯 Objectives

- Analyze lithium-ion battery degradation using discharge-cycle data.
- Reduce noise in voltage, current, and temperature measurements.
- Extract informative features from each discharge cycle.
- Compare individual regression models with a **stacked ensemble**.
- Predict battery **capacity (Ah)** and calculate **SoH (%)**.
- Evaluate performance using **MAE**, **RMSE**, and **R²**.
- Explain feature contributions using **SHAP**.
- Demonstrate predictions through a manually controlled **Gradio** interface.

## 📂 Dataset

**Source:** NASA Ames Prognostics Center of Excellence — Lithium-Ion Battery Aging Dataset.

The notebook uses **B0005** for its main chronological train/test experiment and **B0005, B0006, B0007, and B0018** for cross-battery testing.

| Data / feature | Description |
|---|---|
| Mean voltage | Average discharge voltage (V) |
| Minimum voltage | Lowest measured voltage during discharge (V) |
| Mean current | Average measured discharge current (A) |
| Maximum temperature | Highest measured battery temperature (°C) |
| Discharge duration | Elapsed discharge time (seconds) |
| Capacity | Measured discharge capacity (Ah); prediction target |

**SoH calculation:**

```text
State of Health (%) = Predicted Capacity (Ah) / Nominal Capacity (Ah) × 100
```

The demonstrated notebook uses **2.0 Ah** as nominal capacity. Because it is a fixed reference, some cells can return SoH above 100% if their measured capacity exceeds 2.0 Ah.

## 🛠️ Technologies & Libraries

- Python / Google Colab / Jupyter Notebook
- Pandas & NumPy
- SciPy (Savitzky–Golay filtering)
- Scikit-learn (SVR, Random Forest, Ridge Regression, model validation, metrics)
- XGBoost
- SHAP (Explainable AI)
- Matplotlib
- Gradio (interactive dashboard)

## ⚙️ Project Workflow

### 1. Data Loading
- Mount Google Drive in Colab.
- Read `metadata.csv` and extract the battery cycle CSVs from `data.zip`.
- Select discharge cycles and retrieve capacity measurements.

### 2. Preprocessing and Feature Extraction
- Apply the **Savitzky–Golay filter** to within-cycle voltage, current, and temperature signals (window length **15**, polynomial order **2**).
- Calculate the five discharge-cycle features listed above.
- Apply **Min-Max scaling**; fit the scaler only on training data to reduce data leakage.
- Order B0005 cycles by file number rather than lexicographic `start_time`, which gave an incorrect cycle order in the source run.

### 3. Chronological Train/Test Split
- **134 early B0005 cycles (80%)** for training.
- **34 later B0005 cycles (20%)** for testing.
- Do **not** shuffle the cycles: evaluation is intended to predict future degradation.

### 4. Model Training and Comparison

The notebook trains and compares:

1. **Support Vector Regression (SVR)** — RBF-kernel regression.
2. **Random Forest** — ensemble of decision trees.
3. **XGBoost** — gradient-boosted trees.
4. **Stacked Ensemble** — combines base-model predictions with a **Ridge Regression** meta-learner.

It also reports comparison baselines and ablation experiments. Hyperparameters are selected using forward-chaining `TimeSeriesSplit` within the training partition.

### 5. Model Evaluation

Models are compared using:

- **MAE:** Mean Absolute Error (Ah)
- **RMSE:** Root Mean Squared Error (Ah)
- **R²:** Coefficient of Determination
- **Predicted vs. measured capacity** over held-out discharge cycles

### 6. Cross-Battery Validation

A **leave-one-battery-out** evaluation trains on three cells and tests on the remaining cell, rotating across **B0005, B0006, B0007, and B0018**. This provides a separate check of generalization beyond the B0005 time split.

### 7. Explainable AI (SHAP)

- Apply SHAP to the **full stacked model's predictions**.
- Generate a global SHAP summary plot and local feature-attribution plots.
- Inspect which discharge features increase or decrease the predicted capacity.

### 8. Gradio Dashboard

A demonstration interface allows manual input of:

- Mean Voltage (V)
- Minimum Voltage (V)
- Mean Current (A)
- Maximum Temperature (°C)
- Discharge Duration (s)

It displays the predicted capacity, estimated SoH, battery-health status, and a local SHAP contribution chart. **This is a manually entered telemetry demo, not a connected real-time vehicle BMS.**

## 📊 Key Findings

### B0005 — Chronological Test Set

| Metric | Stacked ensemble (Ridge meta-learner) |
|---|---:|
| **R²** | **0.9841** |
| **RMSE** | **0.0028 Ah** |
| Training cycles | 134 |
| Test cycles | 34 |

The predicted-capacity curve closely follows the measured degradation trend in the held-out B0005 cycles.

### Cross-Battery Evaluation

The accompanying project paper reports the following **average leave-one-battery-out** results for the stacked ensemble:

| Metric | Reported mean |
|---|---:|
| **R²** | **0.9976** |
| **RMSE** | **0.0082 Ah** |

These are **reported results from the supplied project material**, not scores independently reproduced here. Cross-cell performance depends on battery conditions and experimental setup.

### SHAP Findings

- **Discharge duration** and **mean current** were the strongest contributors in the reported analysis.
- SHAP plots show individual features' contributions to a particular capacity prediction.
- These relationships are **model explanations**, not proof of physical causation.

### Example Dashboard Output

For one example entered in the notebook, the dashboard reported approximately:

```text
Predicted capacity: 1.7673 Ah
Estimated SoH:      88.36%
Status:             HEALTHY: normal operating conditions
```

Results change with the input feature values. The status thresholds are demonstration logic rather than a validated safety or diagnostic standard.

## 📁 Project Structure

```text
EV-Battery-Health-ML/
├── README.md
├── requirements.txt
└── EV_battery_camera_ready_experiments.ipynb
```

> Upload the **actual `.ipynb` notebook** from Colab. The PDF screenshots alone cannot be executed. The NASA dataset is downloaded separately; it is not included in this structure.

## 🚀 Installation & Execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/EV-Battery-Health-ML.git
cd EV-Battery-Health-ML
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install numpy pandas scipy matplotlib scikit-learn xgboost shap gradio notebook
```

You can also save those packages in `requirements.txt` and run:

```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset in Google Drive

Download the NASA lithium-ion battery aging dataset. Match the folder and filenames expected by the notebook:

```text
MyDrive/
└── NASA_Battery_Data/
    ├── metadata.csv
    └── data.zip
```

The notebook loads files from:

```python
zip_path = '/content/drive/MyDrive/NASA_Battery_Data/data.zip'
metadata_path = '/content/drive/MyDrive/NASA_Battery_Data/metadata.csv'
```

Make sure `data.zip` contains the battery cycle CSVs in the folder layout expected by the notebook.

### 4. Run in Google Colab (Recommended)

1. Open `EV_battery_camera_ready_experiments.ipynb` in **Google Colab**.
2. If needed, install the dependencies in a code cell:

   ```python
   %pip install numpy pandas scipy matplotlib scikit-learn xgboost shap gradio
   ```

3. Select **Runtime → Run all**.
4. Authorize the Google Drive mount when prompted.
5. Review the model comparison tables, capacity-prediction plots, and SHAP figures.
6. Run the Gradio dashboard cell and open the temporary link printed by Colab.

### 5. Run Locally (Optional)

```bash
jupyter notebook EV_battery_camera_ready_experiments.ipynb
```

**Important:** The supplied notebook uses **Colab-specific Google Drive paths and mounting code**. To run it locally, first replace these with local dataset paths and remove/adjust `google.colab` imports. The dashboard can then be launched from its final cell.

**No compilation step is required:** Python executes the notebook cells directly.

## 🖥️ Expected Outputs

After a successful run, the notebook produces:

- A check of correct battery cycle ordering.
- Per-cell feature tables and battery capacity summaries.
- A table comparing regression models and ablations (MAE, RMSE, R²).
- A measured-vs-predicted **B0005 discharge capacity** graph.
- Leave-one-battery-out cross-cell comparison tables.
- Global SHAP feature-importance visualizations.
- Local SHAP explanations for selected predictions.
- A **Gradio dashboard** for manual battery telemetry input.

## ⚠️ Limitations

- Results are based on NASA **laboratory aging data**, not live EV telemetry.
- The model uses features calculated from a **complete discharge cycle**, limiting use during partial real-world driving cycles.
- Dataset protocols and cell-specific discharge cutoffs can affect cross-cell generalization.
- Current and duration encode information closely related to capacity; high prediction accuracy should not be interpreted as independent long-term prognostics.
- The dashboard is a **software demonstrator**, not a hardware-integrated or safety-certified battery management system.

## 🔮 Future Improvements

- Evaluate additional battery chemistries and environmental conditions.
- Test with partial-cycle data and variable load profiles.
- Integrate real sensor streams with a battery management system.
- Assess embedded-device inference cost and latency.
- Add uncertainty estimates and calibrated battery-health alerts.

## 👨‍💻 Project Team

- Kandagatla Anoop
- Nallamothu Venkata Charan
- Thogiti Manideep
- Kotakonda Jai Raj Maharishi
- Yedula Vijay Eshwar

**Project Guide:** Dr. G. Bala Krishna, Assistant Professor  
**Institution:** Anurag University, Department of Computer Science and Engineering

---

*Academic and research demonstration of lithium-ion battery health estimation using machine learning and explainable AI.*
