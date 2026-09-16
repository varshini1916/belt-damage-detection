import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "predictive_maintenance/ai4i2020.csv"
MODEL_PATH = "models/failure_risk_xgboost.pkl"

OUTPUT_DIR = "outputs/shap"
SUMMARY_PLOT = os.path.join(
    OUTPUT_DIR,
    "failure_risk_shap_summary.png"
)

BAR_PLOT = os.path.join(
    OUTPUT_DIR,
    "failure_risk_shap_bar.png"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading predictive maintenance dataset...")

df = pd.read_csv(DATA_PATH)


# ============================================================
# FEATURES
# ============================================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

display_names = [
    "Air Temperature",
    "Process Temperature",
    "Rotational Speed",
    "Torque",
    "Tool Wear"
]

X = df[features].copy()

X.columns = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading XGBoost model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP explainer...")

explainer = shap.TreeExplainer(model)

# Use a sample for faster visualization
X_sample = X.sample(
    n=min(2000, len(X)),
    random_state=42
)

shap_values = explainer.shap_values(X_sample)

print("SHAP values calculated successfully.")


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=display_names,
    show=False
)

plt.title(
    "SHAP Feature Impact on Machine Failure Risk"
)

plt.tight_layout()

plt.savefig(
    SUMMARY_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Summary plot saved to:")
print(SUMMARY_PLOT)


# ============================================================
# SHAP BAR PLOT
# ============================================================

print("\nGenerating SHAP feature importance plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    X_sample,
    feature_names=display_names,
    plot_type="bar",
    show=False
)

plt.title(
    "Average SHAP Feature Importance"
)

plt.tight_layout()

plt.savefig(
    BAR_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Bar plot saved to:")
print(BAR_PLOT)


# ============================================================
# MEAN ABSOLUTE SHAP VALUES
# ============================================================

mean_abs_shap = pd.DataFrame({
    "Feature": display_names,
    "Mean Absolute SHAP": abs(shap_values).mean(axis=0)
})

mean_abs_shap = mean_abs_shap.sort_values(
    "Mean Absolute SHAP",
    ascending=False
)

print("\n" + "=" * 60)
print("SHAP FEATURE IMPORTANCE")
print("=" * 60)

for _, row in mean_abs_shap.iterrows():

    print(
        f"{row['Feature']:<25} : "
        f"{row['Mean Absolute SHAP']:.6f}"
    )


# ============================================================
# SAVE SHAP IMPORTANCE
# ============================================================

importance_path = os.path.join(
    OUTPUT_DIR,
    "shap_feature_importance.csv"
)

mean_abs_shap.to_csv(
    importance_path,
    index=False
)

print("\nSHAP importance saved to:")
print(importance_path)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)

print(f"Summary plot : {SUMMARY_PLOT}")
print(f"Bar plot     : {BAR_PLOT}")
print(f"CSV results  : {importance_path}")