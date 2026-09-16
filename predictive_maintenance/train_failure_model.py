import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from xgboost import XGBClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "predictive_maintenance/ai4i2020.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "failure_risk_xgboost.pkl"
)

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading predictive maintenance dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")


# ============================================================
# SELECT FEATURES
# ============================================================

features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

X = df[features].copy()
y = df[target].copy()


# ============================================================
# CLEAN FEATURE NAMES FOR XGBOOST
# ============================================================

# XGBoost does not allow characters such as
# [, ] and < in feature names.

X.columns = [
    "air_temperature",
    "process_temperature",
    "rotational_speed",
    "torque",
    "tool_wear"
]


# ============================================================
# DISPLAY DATA INFORMATION
# ============================================================

print("\nSelected features:")
for feature in X.columns:
    print(f"  - {feature}")

print(f"\nTarget: {target}")

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ============================================================
# HANDLE CLASS IMBALANCE
# ============================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_pos_weight = negative / positive

print(f"\nNormal samples : {negative}")
print(f"Failure samples: {positive}")
print(f"Scale pos weight: {scale_pos_weight:.2f}")


# ============================================================
# CREATE XGBOOST MODEL
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining XGBoost failure-risk model...")

model.fit(
    X_train,
    y_train
)

print("Training completed successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("FAILURE-RISK MODEL RESULTS")
print("=" * 60)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)


print(f"\nROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\nFeature Importance:")

importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(
    ascending=False
)

for feature, value in importance.items():
    print(
        f"  {feature:<25} : {value:.4f}"
    )


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print("\n" + "=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"Model path: {MODEL_PATH}")
print("\nXGBoost failure-risk model is ready.")