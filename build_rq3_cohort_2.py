import os
import pandas as pd
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────
MIMIC_DIR   = "/datasets/vradzilani/physionet.org/files/mimiciv/3.1"
ICU_DIR     = os.path.join(MIMIC_DIR, "icu")
HOSP_DIR    = os.path.join(MIMIC_DIR, "hosp")
OUT_FILE    = "mimic_sepsis_rq3_extended.csv"
FIRST_N_HOURS = 24

# ── Item IDs ──────────────────────────────────────────────────────────────────
CHART_ITEMS = {
    "heart_rate":  [220045],
    "resp_rate":   [220210, 224690],
    "spo2":        [220277],
    "temp_f":      [223761],
    "temp_c":      [223762],
    "map":         [220052, 220181, 225312],
    "sbp":         [220050, 220179, 225309],
    "dbp":         [220051, 220180, 225310],
}

LAB_ITEMS = {
    "wbc":         [51301],
    "lactate":     [50813],
    "creatinine":  [50912],
    "bilirubin":   [50885],
    "platelets":   [51265],
    "hemoglobin":  [50811, 51222],
    "bun":         [51006],
    "potassium":   [50971, 50822],
    "sodium":      [50983, 50824],
    "glucose":     [50931, 50809],
    "bicarbonate": [50882],
    "chloride":    [50902, 50806],
}

SEPSIS_ICD10 = {
    "A4101", "A4102", "A411",  "A412",  "A413",  "A414",
    "A4150", "A4151", "A4152", "A4153", "A4159",
    "A4181", "A4189", "A419",
    "A400",  "A401",  "A403",  "A408",  "A409",
    "R6520", "R6521",
}

def rpath(subdir, fname):
    plain = os.path.join(subdir, fname)
    gz    = plain + ".gz"
    if os.path.exists(plain):
        return plain
    if os.path.exists(gz):
        return gz
    raise FileNotFoundError(f"Cannot find {plain} or {gz}")

def read(subdir, fname, **kwargs):
    return pd.read_csv(rpath(subdir, fname), **kwargs)

print("[1/6] Loading ICU stays, Patients, Admissions")
icustays = read(
    ICU_DIR,
    "icustays.csv",
    usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime"],
    parse_dates=["intime"],
)
patients = read(HOSP_DIR, "patients.csv", usecols=["subject_id", "gender", "anchor_age"]).rename(columns={"anchor_age": "age"})
admissions = read(HOSP_DIR, "admissions.csv", usecols=["hadm_id", "admission_type"])

print("[2/6] Building Labels & Comorbidities from ICD-10")
diag = read(HOSP_DIR, "diagnoses_icd.csv", usecols=["hadm_id", "icd_code", "icd_version"])
diag10 = diag[diag["icd_version"] == 10].copy()

sepsis_hadm = diag10[diag10["icd_code"].isin(SEPSIS_ICD10)]["hadm_id"].unique()
htn_hadm    = diag10[diag10["icd_code"].str.startswith(("I10", "I11", "I12", "I13", "I14", "I15"), na=False)]["hadm_id"].unique()
diab_hadm   = diag10[diag10["icd_code"].str.startswith(("E08", "E09", "E10", "E11", "E13"), na=False)]["hadm_id"].unique()
ckd_hadm    = diag10[diag10["icd_code"].str.startswith(("N17", "N18", "N19"), na=False)]["hadm_id"].unique()
heart_hadm  = diag10[diag10["icd_code"].str.startswith(("I20", "I21", "I22", "I23", "I24", "I25", "I50"), na=False)]["hadm_id"].unique()

print("[3/6] Extracting Vitals (chartevents)")
all_chart_ids = {iid for ids in CHART_ITEMS.values() for iid in ids}
itemid_to_feat = {iid: feat for feat, ids in CHART_ITEMS.items() for iid in ids}

stay_id_set = set(icustays["stay_id"])
chunks = []
reader = pd.read_csv(
    rpath(ICU_DIR, "chartevents.csv"),
    usecols=["stay_id", "itemid", "valuenum", "charttime"],
    parse_dates=["charttime"],
    chunksize=2_000_000,
    low_memory=True,
)

for chunk in reader:
    sub = chunk[
        chunk["stay_id"].isin(stay_id_set)
        & chunk["itemid"].isin(all_chart_ids)
        & chunk["valuenum"].notna()
    ]
    if len(sub):
        chunks.append(sub)

chart = pd.concat(chunks, ignore_index=True)
chart = chart.merge(icustays[["stay_id", "intime"]], on="stay_id", how="left")
chart["charttime"] = pd.to_datetime(chart["charttime"], errors="coerce")
chart["intime"] = pd.to_datetime(chart["intime"], errors="coerce")
chart = chart.dropna(subset=["charttime", "intime"])
chart["hours_in"] = (chart["charttime"] - chart["intime"]).dt.total_seconds() / 3600
chart = chart[chart["hours_in"].between(0, FIRST_N_HOURS)]
chart["feature"] = chart["itemid"].map(itemid_to_feat)

chart_f = chart[chart["feature"] == "temp_f"].copy()
chart_c = chart[chart["feature"] == "temp_c"].copy()
chart_f = chart_f[(chart_f["valuenum"] > 86) & (chart_f["valuenum"] < 110)]
chart_f["valuenum"] = (chart_f["valuenum"] - 32) / 1.8
chart_f["feature"] = "temperature"
chart_c = chart_c[(chart_c["valuenum"] > 30) & (chart_c["valuenum"] < 43)]
chart_c["feature"] = "temperature"
chart = chart[~chart["feature"].isin(["temp_f", "temp_c"])]
chart = pd.concat([chart, chart_f, chart_c], ignore_index=True)

vitals = chart.groupby(["stay_id", "feature"])["valuenum"].mean().unstack("feature").reset_index()
if "sbp" in vitals.columns and "dbp" in vitals.columns:
    if "map" not in vitals.columns:
        vitals["map"] = np.nan
    needs = vitals["map"].isna()
    vitals.loc[needs, "map"] = vitals.loc[needs, "dbp"] + (vitals.loc[needs, "sbp"] - vitals.loc[needs, "dbp"]) / 3
vitals = vitals.drop(columns=[c for c in ["sbp", "dbp"] if c in vitals.columns])

print("[4/6] Extracting Labs (labevents)")
all_lab_ids = {iid for ids in LAB_ITEMS.values() for iid in ids}
lab_itemid_to_feat = {iid: feat for feat, ids in LAB_ITEMS.items() for iid in ids}

hadm_id_set = set(icustays["hadm_id"])
lab_chunks = []
reader = pd.read_csv(rpath(HOSP_DIR, "labevents.csv"), usecols=["hadm_id", "itemid", "valuenum"], chunksize=2_000_000, low_memory=True)

for chunk in reader:
    sub = chunk[chunk["hadm_id"].isin(hadm_id_set) & chunk["itemid"].isin(all_lab_ids) & chunk["valuenum"].notna()]
    if len(sub):
        lab_chunks.append(sub)

labs_long = pd.concat(lab_chunks, ignore_index=True)
labs_long["feature"] = labs_long["itemid"].map(lab_itemid_to_feat)
labs = labs_long.groupby(["hadm_id", "feature"])["valuenum"].mean().unstack("feature").reset_index()

print("[5/6] Assembling final extended cohort")
cohort = icustays[["subject_id", "hadm_id", "stay_id", "first_careunit"]].copy()
cohort = cohort.merge(patients, on="subject_id", how="left")
cohort = cohort.merge(admissions, on="hadm_id", how="left")
cohort = cohort.merge(vitals, on="stay_id", how="left")
cohort = cohort.merge(labs, on="hadm_id", how="left")

cohort["sepsis"] = cohort["hadm_id"].isin(sepsis_hadm).astype(int)
cohort["comorb_htn"] = cohort["hadm_id"].isin(htn_hadm).astype(int)
cohort["comorb_diabetes"] = cohort["hadm_id"].isin(diab_hadm).astype(int)
cohort["comorb_ckd"] = cohort["hadm_id"].isin(ckd_hadm).astype(int)
cohort["comorb_heart"] = cohort["hadm_id"].isin(heart_hadm).astype(int)

cohort = cohort.drop(columns=["subject_id", "hadm_id", "stay_id"])

cohort.to_csv(OUT_FILE, index=False)
print(f"Saved extended dataset with {len(cohort.columns)} features to {OUT_FILE}")
