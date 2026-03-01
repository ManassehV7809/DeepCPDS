#!/usr/bin/env python
"""
build_mimic_sepsis_bn_cohort.py

Constructs a simple, BN-ready sepsis cohort from MIMIC-IV v3.1.

- Adult ICU patients (>=18 years).
- One row per ICU stay.
- Static features: age, gender, admission_type, first_careunit.
- Physiological features: mean heart rate, mean temperature, mean WBC, mean lactate.
- Label: sepsis (binary) from sepsis-related ICD codes.

This is a lightweight, static cohort inspired by Sepsis-3 criteria and
MIMIC-Sepsis style work, but does NOT reproduce their full temporal pipeline.
"""

import os
import pandas as pd


def main():
    # ---- 1. Configure paths ----
    # Base path where you moved the MIMIC-IV files
    base = "/datasets/vradzilani/physionet.org/files/mimiciv/3.1"

    hosp_dir = os.path.join(base, "hosp")
    icu_dir = os.path.join(base, "icu")

    out_path = "/datasets/vradzilani/mimic_sepsis_bn_cohort.csv"

    print(f"[info] Using MIMIC-IV base directory: {base}")
    print(f"[info] Output will be written to: {out_path}")

    # ---- 2. Load core tables ----
    # patients: subject_id, gender, anchor_age
    print("[info] Loading patients.csv ...")
    patients = pd.read_csv(
        os.path.join(hosp_dir, "patients.csv"),
        usecols=["subject_id", "gender", "anchor_age"],
    )

    # admissions: hadm_id, admission_type
    print("[info] Loading admissions.csv ...")
    admissions = pd.read_csv(
        os.path.join(hosp_dir, "admissions.csv"),
        usecols=["hadm_id", "admission_type"],
    )

    # icustays: subject_id, hadm_id, stay_id, intime, first_careunit
    print("[info] Loading icustays.csv ...")
    icustays = pd.read_csv(
        os.path.join(icu_dir, "icustays.csv"),
        usecols=["subject_id", "hadm_id", "stay_id", "intime", "first_careunit"],
    )

    # diagnoses_icd: hadm_id, icd_code (for sepsis flag)
    print("[info] Loading diagnoses_icd.csv ...")
    diagnoses = pd.read_csv(
        os.path.join(hosp_dir, "diagnoses_icd.csv"),
        usecols=["hadm_id", "icd_code"],
    )

    # ---- 3. Define a simple ICD-based sepsis flag ----
    # Sepsis-related ICD codes (example: A41*, 99591, 99592; you can refine)
    print("[info] Creating ICD-based sepsis flag ...")
    sepsis_icd_prefixes = ("A41", "99591", "99592")

    diagnoses["icd_code"] = diagnoses["icd_code"].astype(str)
    diagnoses["sepsis_flag"] = diagnoses["icd_code"].str.startswith(sepsis_icd_prefixes).astype(int)
    sepsis_by_hadm = (
        diagnoses.groupby("hadm_id")["sepsis_flag"]
        .max()
        .reset_index()
    )

    # ---- 4. Load vitals from chartevents ----
    # We only pull the columns we need for memory reasons.
    # Heart rate: itemid 220045
    # Temperature: itemids 223761 (Fahrenheit), 223762 (Celsius) – we will convert both to Celsius.
    print("[info] Loading chartevents.csv (this may take a while) ...")
    chartevents = pd.read_csv(
        os.path.join(icu_dir, "chartevents.csv"),
        usecols=["stay_id", "itemid", "valuenum", "valueuom"],
    )
    chartevents = chartevents.dropna(subset=["valuenum"])

    # Heart rate
    hr_ids = [220045]
    hr = chartevents[chartevents["itemid"].isin(hr_ids)].copy()
    hr = hr.rename(columns={"valuenum": "heart_rate"})
    hr = hr.groupby("stay_id", as_index=False)["heart_rate"].mean()

    # Temperature: convert all to Celsius
    temp_ids = [223761, 223762]
    temp = chartevents[chartevents["itemid"].isin(temp_ids)].copy()
    # Normalize units
    temp["valueuom"] = temp["valueuom"].astype(str).str.lower()
    # If Fahrenheit, convert to Celsius
    is_f = temp["valueuom"].str.contains("f")
    temp.loc[is_f, "valuenum"] = (temp.loc[is_f, "valuenum"] - 32.0) * (5.0 / 9.0)
    temp = temp.rename(columns={"valuenum": "temperature"})
    temp = temp.groupby("stay_id", as_index=False)["temperature"].mean()

    # ---- 5. Load labs from labevents ----
    print("[info] Loading labevents.csv (this may also take a while) ...")
    labevents = pd.read_csv(
        os.path.join(hosp_dir, "labevents.csv"),
        usecols=["hadm_id", "itemid", "valuenum", "valueuom"],
    )
    labevents = labevents.dropna(subset=["valuenum"])

    # WBC: example itemid 51300 (you can add more if needed)
    wbc_ids = [51300]
    wbc = labevents[labevents["itemid"].isin(wbc_ids)].copy()
    wbc = wbc.rename(columns={"valuenum": "wbc"})
    wbc = wbc.groupby("hadm_id", as_index=False)["wbc"].mean()

    # Lactate: example itemid 50813 (you can refine as needed)
    lactate_ids = [50813]
    lactate = labevents[labevents["itemid"].isin(lactate_ids)].copy()
    lactate = lactate.rename(columns={"valuenum": "lactate"})
    lactate = lactate.groupby("hadm_id", as_index=False)["lactate"].mean()

    # ---- 6. Merge into a single cohort ----
    print("[info] Merging tables to form cohort ...")

    # Start from ICU stays
    cohort = icustays.merge(patients, on="subject_id", how="left")
    cohort = cohort.merge(admissions, on="hadm_id", how="left")
    cohort = cohort.merge(sepsis_by_hadm, on="hadm_id", how="left")

    # Attach vitals (by stay_id)
    cohort = cohort.merge(hr, on="stay_id", how="left")
    cohort = cohort.merge(temp, on="stay_id", how="left")

    # Attach labs (by hadm_id)
    cohort = cohort.merge(wbc, on="hadm_id", how="left")
    cohort = cohort.merge(lactate, on="hadm_id", how="left")

    # ---- 7. Basic filtering ----
    print("[info] Applying filters (adults, non-missing key vars) ...")

    # Adults only
    cohort = cohort[cohort["anchor_age"] >= 18]

    # Convert missing sepsis_flag to 0
    cohort["sepsis_flag"] = cohort["sepsis_flag"].fillna(0).astype(int)

    # Drop rows with missing key physiologic values
    key_cols = ["heart_rate", "temperature", "wbc", "lactate"]
    cohort = cohort.dropna(subset=key_cols)

    # Optionally downsample to keep cohort size manageable for MSc experiments
    if len(cohort) > 50000:
        print(f"[info] Cohort has {len(cohort)} rows; downsampling to 50,000 for practicality ...")
        cohort = cohort.sample(n=50000, random_state=42)

    # ---- 8. Rename columns to match BN variable naming ----
    cohort = cohort.rename(columns={
        "anchor_age": "age",
        "gender": "gender",
        "admission_type": "admission_type",
        "first_careunit": "first_careunit",
        "sepsis_flag": "sepsis",
    })

    # Keep only columns we want for BN
    final_cols = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "age",
        "gender",
        "admission_type",
        "first_careunit",
        "heart_rate",
        "temperature",
        "wbc",
        "lactate",
        "sepsis",
    ]
    cohort_bn = cohort[final_cols].copy()

    print(f"[info] Final cohort shape: {cohort_bn.shape[0]} rows, {cohort_bn.shape[1]} columns")

    # ---- 9. Save to CSV ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cohort_bn.to_csv(out_path, index=False)
    print(f"[info] Wrote BN-ready sepsis cohort to: {out_path}")


if __name__ == "__main__":
    main()
