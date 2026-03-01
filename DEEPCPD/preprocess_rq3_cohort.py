"""
preprocess_rq3_cohort.py
=========================
Takes mimic_sepsis_rq3_cohort.csv and produces a fully discretised,
BN-ready dataset for RQ3 structure learning and CPD experiments.

Design decisions:
  - Lactate: kept + informative missingness flag (lactate_measured)
  - All continuous variables binned into clinically meaningful states
  - Categorical variables label-encoded
  - Train/val/test split: 70/10/20 stratified on sepsis

Output files:
  rq3_full_discretised.csv      — full dataset, all rows
  rq3_train.csv
  rq3_val.csv
  rq3_test.csv
  rq3_metadata.json             — bin edges, category maps (for reproducibility)

Bin rationale (documented for thesis methodology section):
  heart_rate:  bradycardia <60, normal 60-100, tachycardia >100
               (SIRS criterion: >90 bpm)
  temperature: hypothermia <36, normal 36-38, fever >38
               (SIRS criterion: <36 or >38 °C)
  resp_rate:   low <12, normal 12-20, tachypnoea >20
               (SIRS criterion: >20 breaths/min)
  map:         shock <65, low-normal 65-70, normal 70-100, high >100
               (Sepsis-3: MAP <65 mmHg defines haemodynamic instability)
  spo2:        severe hypoxia <90, mild hypoxia 90-94, normal >=95
  wbc:         leucopaenia <4, normal 4-12, leucocytosis >12
               (SIRS criterion: <4 or >12 x10^9/L)
  lactate:     normal <2.0, elevated 2.0-4.0, high >4.0
               (Sepsis-3: lactate >2 mmol/L indicates tissue hypoperfusion)
  age:         young <45, middle 45-65, older >65
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────

IN_FILE  = "mimic_sepsis_rq3_cohort.csv"
OUT_DIR  = "."   # write outputs here; change if needed

RANDOM_SEED = 42

# ── Bin definitions ───────────────────────────────────────────────────────────
# (edges, labels) — edges are the cut points between bins
# Using -inf/inf as outer bounds so no value falls outside

BINS = {
    "heart_rate": {
        "edges":  [-np.inf, 60, 100, np.inf],
        "labels": ["bradycardia", "normal", "tachycardia"],
        "rationale": "SIRS criterion >90 bpm; clinical thresholds 60/100"
    },
    "temperature": {
        "edges":  [-np.inf, 36.0, 38.0, np.inf],
        "labels": ["hypothermia", "normal", "fever"],
        "rationale": "SIRS criterion <36°C or >38°C"
    },
    "resp_rate": {
        "edges":  [-np.inf, 12, 20, np.inf],
        "labels": ["low", "normal", "tachypnoea"],
        "rationale": "SIRS criterion >20 breaths/min"
    },
    "map": {
        "edges":  [-np.inf, 65, 70, 100, np.inf],
        "labels": ["shock", "low_normal", "normal", "high"],
        "rationale": "Sepsis-3: MAP <65 mmHg = haemodynamic instability"
    },
    "spo2": {
        "edges":  [-np.inf, 90, 95, np.inf],
        "labels": ["severe_hypoxia", "mild_hypoxia", "normal"],
        "rationale": "Standard clinical oxygen saturation thresholds"
    },
    "wbc": {
        "edges":  [-np.inf, 4, 12, np.inf],
        "labels": ["leucopaenia", "normal", "leucocytosis"],
        "rationale": "SIRS criterion <4 or >12 x10^9/L"
    },
    "lactate": {
        "edges":  [-np.inf, 2.0, 4.0, np.inf],
        "labels": ["normal", "elevated", "high"],
        "rationale": "Sepsis-3: >2 mmol/L = tissue hypoperfusion risk"
    },
    "age": {
        "edges":  [-np.inf, 45, 65, np.inf],
        "labels": ["young", "middle", "older"],
        "rationale": "Standard age stratification for ICU risk"
    },
}

# ── Categorical mappings ──────────────────────────────────────────────────────

GENDER_MAP = {"M": "male", "F": "female"}

# Collapse admission_type to 3 meaningful groups
def map_admission_type(val):
    if pd.isna(val):
        return "other"
    v = str(val).upper()
    if "EMER" in v:
        return "emergency"
    if "URGENT" in v:
        return "urgent"
    return "elective_other"

# Collapse first_careunit to 5 groups
def map_careunit(val):
    if pd.isna(val):
        return "other"
    v = str(val).upper()
    if "MICU" in v and "SICU" in v:
        return "MICU_SICU"
    if "MICU" in v:
        return "MICU"
    if "SICU" in v:
        return "SICU"
    if "CCU" in v or "CORONARY" in v:
        return "CCU"
    if "CSRU" in v or "CARDIAC" in v:
        return "CSRU"
    if "NICU" in v or "NEURO" in v:
        return "NICU"
    return "other"

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading cohort...")
    df = pd.read_csv(IN_FILE)
    print(f"  Shape: {df.shape}")

    # ── Lactate: informative missingness flag ─────────────────────────────────
    df["lactate_measured"] = df["lactate"].notna().astype(int).map(
        {1: "yes", 0: "no"}
    )
    print(f"\nLactate measured in {(df['lactate_measured']=='yes').sum():,} "
          f"/ {len(df):,} stays ({100*(df['lactate_measured']=='yes').mean():.1f}%)")

    # ── Discretise continuous variables ───────────────────────────────────────
    print("\nDiscretising continuous variables...")
    metadata = {"bins": {}, "categoricals": {}}

    for col, spec in BINS.items():
        if col not in df.columns:
            print(f"  SKIP {col} — not in dataframe")
            continue

        binned_col = col + "_bin"

        df[binned_col] = pd.cut(
            df[col],
            bins=spec["edges"],
            labels=spec["labels"],
            right=True,
        ).astype(str)

        # Where original was NaN, mark explicitly
        df.loc[df[col].isna(), binned_col] = "missing"

        n_missing = (df[binned_col] == "missing").sum()
        dist = df[binned_col].value_counts().to_dict()
        print(f"  {col:<14} → {binned_col}  |  {dist}  |  missing={n_missing}")

        metadata["bins"][col] = {
            "edges":    [str(e) for e in spec["edges"]],
            "labels":   spec["labels"],
            "rationale": spec["rationale"],
        }

    # ── Encode categoricals ───────────────────────────────────────────────────
    print("\nEncoding categorical variables...")

    df["gender_enc"]         = df["gender"].map(GENDER_MAP).fillna("unknown")
    df["admission_type_enc"] = df["admission_type"].apply(map_admission_type)
    df["first_careunit_enc"] = df["first_careunit"].apply(map_careunit)

    for col in ["gender_enc", "admission_type_enc", "first_careunit_enc"]:
        dist = df[col].value_counts().to_dict()
        print(f"  {col}: {dist}")
        metadata["categoricals"][col] = list(df[col].unique())

    # ── Build final BN-ready dataframe ────────────────────────────────────────
    BN_COLS = {
        # Final column name      : source column
        "age":                   "age_bin",
        "gender":                "gender_enc",
        "admission_type":        "admission_type_enc",
        "first_careunit":        "first_careunit_enc",
        "heart_rate":            "heart_rate_bin",
        "temperature":           "temperature_bin",
        "resp_rate":             "resp_rate_bin",
        "map":                   "map_bin",
        "spo2":                  "spo2_bin",
        "wbc":                   "wbc_bin",
        "lactate_measured":      "lactate_measured",
        "lactate":               "lactate_bin",
        "sepsis":                "sepsis",
    }

    bn = pd.DataFrame()
    for out_col, src_col in BN_COLS.items():
        if src_col in df.columns:
            bn[out_col] = df[src_col]
        else:
            print(f"  WARNING: {src_col} not found — skipping {out_col}")

    # Sepsis as string for consistency
    bn["sepsis"] = bn["sepsis"].map({1: "yes", 0: "no"})

    # ── Drop rows with any 'missing' in core SIRS variables ──────────────────
    # (these are the <3% missing — safe to drop)
    core_cols = ["heart_rate", "temperature", "resp_rate", "map", "spo2", "wbc"]
    before = len(bn)
    for col in core_cols:
        if col in bn.columns:
            bn = bn[bn[col] != "missing"]
    after = len(bn)
    print(f"\nDropped {before - after} rows with missing core vitals "
          f"({before} → {after})")

    # For lactate: if not measured, assign a special 'not_measured' bin
    # This preserves informative missingness in the lactate node itself
    if "lactate" in bn.columns:
        bn.loc[bn["lactate"] == "missing", "lactate"] = "not_measured"

    print(f"\nFinal BN dataset shape: {bn.shape}")
    print(f"Columns: {list(bn.columns)}")
    print(f"\nSepsis distribution:\n{bn['sepsis'].value_counts()}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    # No split here — splitting is done in the experiment script at fit time.
    # This keeps preprocessing and experimental design cleanly separated.
    bn.to_csv(os.path.join(OUT_DIR, "rq3_full_discretised.csv"), index=False)
    metadata["node_states"] = {}
    for col in bn.columns:
        metadata["node_states"][col] = sorted(bn[col].dropna().unique().tolist())

    with open(os.path.join(OUT_DIR, "rq3_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\nSaved:")
    print("  rq3_full_discretised.csv")
    print("  rq3_metadata.json")
    print("  (train/val/test split is handled in the experiment script)")

    # ── Print thesis-ready variable summary table ─────────────────────────────
    print("\n" + "="*70)
    print("THESIS TABLE: Variable Discretisation Summary")
    print("="*70)
    print(f"{'Variable':<18} {'States':<50} {'Rationale'}")
    print("-"*70)
    for col, spec in BINS.items():
        if col in bn.columns:
            states = ", ".join(spec["labels"])
            print(f"{col:<18} {states:<50} {spec['rationale'][:40]}")
    for col in ["gender", "admission_type", "first_careunit",
                "lactate_measured", "sepsis"]:
        if col in bn.columns:
            states = ", ".join(sorted(bn[col].unique().tolist()))
            print(f"{col:<18} {states}")
    print("="*70)


if __name__ == "__main__":
    main()
