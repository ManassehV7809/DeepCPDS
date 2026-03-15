"""
rq3_baselines.py
Standalone discriminative baselines for RQ3 (Point 6 — Ritesh feedback).
Fits Logistic Regression and Random Forest on the MIMIC-IV discretised cohort
and evaluates at the sepsis node using the same train/test split as rq3.py.
Outputs: rq3_baseline_results.csv and a printed comparison table.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
import warnings
warnings.filterwarnings("ignore")

# ── Config (must match rq3.py exactly) ───────────────────────────────────────
CSV_PATH      = "rq3_clinical_extended_discretised.csv"
TARGET_NODE   = "sepsis"
TARGET_STATE  = "yes"
TEST_SIZE     = 0.20
RANDOM_STATE  = 42
OUTPUT_CSV    = "rq3_baseline_results.csv"

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"[info] Loaded {len(df)} rows, {len(df.columns)} columns.")
print(f"[info] Sepsis distribution:\n{df[TARGET_NODE].value_counts()}\n")

# ── Encode all columns with OrdinalEncoder ────────────────────────────────────
# OrdinalEncoder maps each string category to an integer.
# We encode ALL columns (including target) so splits are consistent.
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
df_enc = pd.DataFrame(
    enc.fit_transform(df),
    columns=df.columns
)

# Find the integer code for target_state "yes"
target_col_idx = list(df.columns).index(TARGET_NODE)
target_categories = enc.categories_[target_col_idx]
target_state_int = int(np.where(target_categories == TARGET_STATE)[0][0])
print(f"[info] '{TARGET_STATE}' encoded as integer: {target_state_int}")

# ── Features and target ───────────────────────────────────────────────────────
feature_cols = [c for c in df.columns if c != TARGET_NODE]
X = df_enc[feature_cols].values
y = (df_enc[TARGET_NODE].values == target_state_int).astype(int)
print(f"[info] Class balance — sepsis=1: {y.sum()} | sepsis=0: {(1-y).sum()}\n")

# ── Train / test split (identical to rq3.py) ─────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"[info] Train: {len(X_train)} | Test: {len(X_test)}\n")

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
        C=1.0,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=10,
    ),
}

# ── Evaluate ──────────────────────────────────────────────────────────────────
results = []

for name, model in models.items():
    print(f"[info] Fitting {name}...")
    model.fit(X_train, y_train)

    y_prob  = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    auroc   = roc_auc_score(y_test, y_prob)
    pr_auc  = average_precision_score(y_test, y_prob)
    f1      = f1_score(y_test, y_pred, zero_division=0)
    prec    = precision_score(y_test, y_pred, zero_division=0)
    rec     = recall_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model":     name,
        "AUROC":     round(auroc, 4),
        "PR-AUC":    round(pr_auc, 4),
        "F1":        round(f1, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
    })

    print(f"  AUROC={auroc:.4f}  PR-AUC={pr_auc:.4f}  F1={f1:.4f}\n")

# ── Output ────────────────────────────────────────────────────────────────────
results_df = pd.DataFrame(results)

# Add your BN results from rq3.py here for direct comparison
# (fill these in once rq3.sh finishes)
bn_rows = pd.DataFrame([
    {
        "Model":     "BN + Tabular (BIC)",
        "AUROC":     None,
        "PR-AUC":    None,
        "F1":        None,
        "Precision": None,
        "Recall":    None,
    },
    {
        "Model":     "BN + DeepCPD (mBIC)",
        "AUROC":     None,
        "PR-AUC":    None,
        "F1":        None,
        "Precision": None,
        "Recall":    None,
    },
])

final_df = pd.concat([results_df, bn_rows], ignore_index=True)
final_df.to_csv(OUTPUT_CSV, index=False)

print("=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))
print(f"\n[info] Full results saved to {OUTPUT_CSV}")
print("[info] Fill in BN rows manually once rq3.sh completes.")
