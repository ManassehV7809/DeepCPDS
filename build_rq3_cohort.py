"""
build_rq3_cohort.py
====================
Builds a clean sepsis cohort from MIMIC-IV 3.1 for RQ3.

Sources used:
  icu/icustays.csv          → ICU stay info (stay_id, intime, outtime, careunit)
  icu/chartevents.csv       → Vitals (HR, Temp, RR, MAP, SpO2)
  hosp/admissions.csv       → admission_type, hadm_id
  hosp/patients.csv         → age (via anchor_age), gender
  hosp/labevents.csv        → WBC, lactate
  hosp/diagnoses_icd.csv    → Sepsis label via ICD-10 codes

Output columns:
  subject_id, hadm_id, stay_id,
  age, gender, admission_type, first_careunit,
  heart_rate, temperature, resp_rate, map, spo2,
  wbc, lactate,
  sepsis  (1 = sepsis diagnosis present, 0 = not)

Usage:
  python build_rq3_cohort.py

Adjust MIMIC_DIR below if needed.
"""

import os
import pandas as pd
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

MIMIC_DIR   = "/datasets/vradzilani/physionet.org/files/mimiciv/3.1"
ICU_DIR     = os.path.join(MIMIC_DIR, "icu")
HOSP_DIR    = os.path.join(MIMIC_DIR, "hosp")
OUT_FILE    = "mimic_sepsis_rq3_cohort.csv"

# Restrict to first N hours of ICU stay (24h is standard in sepsis literature)
FIRST_N_HOURS = 24

# ── Item IDs (verified against MIMIC-IV MetaVision / mimic-code repo) ────────

CHART_ITEMS = {
    # Vital                itemids
    "heart_rate":  [220045],
    "resp_rate":   [220210, 224690],
    "spo2":        [220277],
    # Temperature: 223761=Fahrenheit, 223762=Celsius
    "temp_f":      [223761],
    "temp_c":      [223762],
    # Blood pressure — invasive and non-invasive MAP combined
    "map":         [220052,   # Arterial BP Mean (invasive)
                    220181,   # Non-invasive BP Mean
                    225312],  # ART BP Mean (alternative label)
    # SBP + DBP kept so we can derive MAP if direct readings sparse
    "sbp":         [220050, 220179, 225309],
    "dbp":         [220051, 220180, 225310],
}

LAB_ITEMS = {
    "wbc":     [51301],   # White Blood Cells (K/uL)
    "lactate": [50813],   # Lactate (mmol/L)
}

# ICD-10 codes for sepsis (A41.x = Other sepsis, A40.x = Streptococcal sepsis)
# R65.20/R65.21 = Severe sepsis without/with septic shock
SEPSIS_ICD10 = {
    "A4101", "A4102", "A411",  "A412",  "A413",  "A414",
    "A4150", "A4151", "A4152", "A4153", "A4159",
    "A4181", "A4189", "A419",
    "A400",  "A401",  "A403",  "A408",  "A409",
    "R6520", "R6521",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def rpath(subdir, fname):
    """Return path, preferring plain .csv over .csv.gz."""
    plain = os.path.join(subdir, fname)
    gz    = plain + ".gz"
    if os.path.exists(plain):
        return plain
    if os.path.exists(gz):
        return gz
    raise FileNotFoundError(f"Cannot find {plain} or {gz}")


def read(subdir, fname, **kwargs):
    path = rpath(subdir, fname)
    print(f"  Reading {os.path.basename(path)} ...", end=" ", flush=True)
    df = pd.read_csv(path, **kwargs)
    print(f"{len(df):,} rows")
    return df


# ── Step 1: ICU stays ─────────────────────────────────────────────────────────

print("\n[1/6] Loading ICU stays")
icustays = read(ICU_DIR, "icustays.csv",
                usecols=["subject_id", "hadm_id", "stay_id",
                         "first_careunit", "intime", "outtime"],
                parse_dates=["intime", "outtime"])

# ── Step 2: Patient demographics ──────────────────────────────────────────────

print("\n[2/6] Loading patients")
patients = read(HOSP_DIR, "patients.csv",
                usecols=["subject_id", "gender", "anchor_age"])
patients = patients.rename(columns={"anchor_age": "age"})

# ── Step 3: Admissions ────────────────────────────────────────────────────────

print("\n[3/6] Loading admissions")
admissions = read(HOSP_DIR, "admissions.csv",
                  usecols=["hadm_id", "admission_type"])

# ── Step 4: Sepsis labels from ICD-10 diagnoses ───────────────────────────────

print("\n[4/6] Building sepsis labels from ICD-10 codes")
diag = read(HOSP_DIR, "diagnoses_icd.csv",
            usecols=["hadm_id", "icd_code", "icd_version"])

sepsis_hadm = (
    diag[
        (diag["icd_version"] == 10) &
        (diag["icd_code"].isin(SEPSIS_ICD10))
    ]["hadm_id"]
    .unique()
)
print(f"  Admissions with sepsis ICD-10 code: {len(sepsis_hadm):,}")

# ── Step 5: Chartevents — vitals ──────────────────────────────────────────────

print("\n[5/6] Extracting vitals from chartevents (large file — be patient)")

all_item_ids = set()
for ids in CHART_ITEMS.values():
    all_item_ids.update(ids)

stay_id_set = set(icustays["stay_id"])

# Build a lookup: itemid → feature name
itemid_to_feat = {}
for feat, ids in CHART_ITEMS.items():
    for iid in ids:
        itemid_to_feat[iid] = feat

chunks = []
reader = pd.read_csv(
    rpath(ICU_DIR, "chartevents.csv"),
    usecols=["stay_id", "itemid", "valuenum", "charttime"],
    parse_dates=["charttime"],
    chunksize=2_000_000,
    low_memory=True,
)

for i, chunk in enumerate(reader):
    sub = chunk[
        chunk["stay_id"].isin(stay_id_set) &
        chunk["itemid"].isin(all_item_ids) &
        chunk["valuenum"].notna()
    ].copy()
    if len(sub):
        chunks.append(sub)
    if (i + 1) % 5 == 0:
        print(f"    ...{(i+1)*2}M rows scanned", flush=True)

print(f"  Done scanning chartevents.")
chart = pd.concat(chunks, ignore_index=True)
print(f"  Matching rows: {len(chart):,}")

# Merge intime so we can restrict to first 24h
chart = chart.merge(icustays[["stay_id", "intime"]], on="stay_id", how="left")
chart["hours_in"] = (chart["charttime"] - chart["intime"]).dt.total_seconds() / 3600
chart = chart[chart["hours_in"].between(0, FIRST_N_HOURS)]
print(f"  After restricting to first {FIRST_N_HOURS}h: {len(chart):,} rows")

# Map feature names
chart["feature"] = chart["itemid"].map(itemid_to_feat)

# Temperature: convert Fahrenheit → Celsius, then merge into single "temperature"
chart_f = chart[chart["feature"] == "temp_f"].copy()
chart_c = chart[chart["feature"] == "temp_c"].copy()

# Plausibility filter before conversion
chart_f = chart_f[(chart_f["valuenum"] > 86) & (chart_f["valuenum"] < 110)]
chart_f["valuenum"] = (chart_f["valuenum"] - 32) / 1.8
chart_f["feature"] = "temperature"

chart_c = chart_c[(chart_c["valuenum"] > 30) & (chart_c["valuenum"] < 43)]
chart_c["feature"] = "temperature"

# Replace temp_f and temp_c rows
chart = chart[~chart["feature"].isin(["temp_f", "temp_c"])]
chart = pd.concat([chart, chart_f, chart_c], ignore_index=True)

# Plausibility filters for remaining vitals
FILTERS = {
    "heart_rate": (20,  300),
    "resp_rate":  (4,   70),
    "spo2":       (50,  100),
    "map":        (20,  200),
    "sbp":        (50,  300),
    "dbp":        (20,  200),
    "temperature":(30,  43),
}
for feat, (lo, hi) in FILTERS.items():
    mask = chart["feature"] == feat
    chart = chart[~(mask & ((chart["valuenum"] < lo) | (chart["valuenum"] > hi)))]

# Aggregate to stay level (mean over first 24h)
vitals = (
    chart.groupby(["stay_id", "feature"])["valuenum"]
    .mean()
    .unstack("feature")
    .reset_index()
)

# Derive MAP from SBP/DBP where direct MAP is missing
if "map" not in vitals.columns:
    vitals["map"] = np.nan
if "sbp" in vitals.columns and "dbp" in vitals.columns:
    needs = vitals["map"].isna()
    vitals.loc[needs, "map"] = (
        vitals.loc[needs, "dbp"] + (vitals.loc[needs, "sbp"] - vitals.loc[needs, "dbp"]) / 3
    )

# Drop raw SBP/DBP — not needed as final features
vitals = vitals.drop(columns=[c for c in ["sbp", "dbp"] if c in vitals.columns])

# ── Step 6: Labevents — WBC and lactate ───────────────────────────────────────

print("\n[6/6] Extracting labs from labevents (large file)")

lab_item_ids = set()
for ids in LAB_ITEMS.values():
    lab_item_ids.update(ids)
lab_itemid_to_feat = {iid: feat for feat, ids in LAB_ITEMS.items() for iid in ids}

hadm_id_set = set(icustays["hadm_id"])
lab_chunks = []

reader = pd.read_csv(
    rpath(HOSP_DIR, "labevents.csv"),
    usecols=["hadm_id", "itemid", "valuenum"],
    chunksize=1_000_000,
    low_memory=True,
)
for i, chunk in enumerate(reader):
    sub = chunk[
        chunk["hadm_id"].isin(hadm_id_set) &
        chunk["itemid"].isin(lab_item_ids) &
        chunk["valuenum"].notna()
    ]
    if len(sub):
        lab_chunks.append(sub)

labs_long = pd.concat(lab_chunks, ignore_index=True)
labs_long["feature"] = labs_long["itemid"].map(lab_itemid_to_feat)

labs = (
    labs_long.groupby(["hadm_id", "feature"])["valuenum"]
    .mean()
    .unstack("feature")
    .reset_index()
)
print(f"  Labs extracted: {len(labs):,} admissions")

# ── Assemble final cohort ─────────────────────────────────────────────────────

print("\nAssembling cohort...")

cohort = icustays[["subject_id", "hadm_id", "stay_id", "first_careunit"]].copy()
cohort = cohort.merge(patients[["subject_id", "age", "gender"]], on="subject_id", how="left")
cohort = cohort.merge(admissions[["hadm_id", "admission_type"]],  on="hadm_id",   how="left")
cohort = cohort.merge(vitals,                                      on="stay_id",   how="left")
cohort = cohort.merge(labs,                                        on="hadm_id",   how="left")

# Sepsis label
cohort["sepsis"] = cohort["hadm_id"].isin(sepsis_hadm).astype(int)

# Final column order
FINAL_COLS = [
    "subject_id", "hadm_id", "stay_id",
    "age", "gender",
    "admission_type", "first_careunit",
    "heart_rate", "temperature", "resp_rate", "map", "spo2",
    "wbc", "lactate",
    "sepsis",
]
FINAL_COLS = [c for c in FINAL_COLS if c in cohort.columns]
cohort = cohort[FINAL_COLS]

# ── Missingness report ────────────────────────────────────────────────────────

print("\n── Missingness Report ──────────────────────────────────────────────────")
for col in FINAL_COLS:
    n   = cohort[col].isna().sum()
    pct = 100 * n / len(cohort)
    flag = "  ⚠  HIGH MISSING — consider dropping or imputing" if pct > 30 else ""
    print(f"  {col:<20} {n:>7,} missing  ({pct:5.1f}%){flag}")

# ── Sepsis prevalence ─────────────────────────────────────────────────────────

n_sepsis = cohort["sepsis"].sum()
print(f"\n── Sepsis Prevalence ───────────────────────────────────────────────────")
print(f"  Total ICU stays : {len(cohort):,}")
print(f"  Sepsis = 1      : {n_sepsis:,}  ({100*n_sepsis/len(cohort):.1f}%)")
print(f"  Sepsis = 0      : {len(cohort)-n_sepsis:,}  ({100*(1-n_sepsis/len(cohort)):.1f}%)")

# ── Save ──────────────────────────────────────────────────────────────────────

cohort.to_csv(OUT_FILE, index=False)
print(f"\nSaved → {OUT_FILE}")
print(f"Shape  : {cohort.shape}")
print(f"Columns: {list(cohort.columns)}")
