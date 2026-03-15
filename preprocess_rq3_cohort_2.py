import pandas as pd
import numpy as np

# Load the extended cohort
df = pd.read_csv("mimic_sepsis_rq3_extended.csv")

# 1. Missingness Indicators (Before Imputation)
# Clinicians care if a lab was ordered at all. If a doctor didn't order a bilirubin test, 
# that implies they didn't suspect liver failure.
for col in ["lactate", "creatinine", "bilirubin", "platelets"]:
    if col in df.columns:
        df[f"{col}_measured"] = np.where(df[col].isna(), "no", "yes")

# 2. Imputation
# Impute continuous with median
continuous_vars = ["age", "heart_rate", "temperature", "resp_rate", "map", "spo2", 
                   "wbc", "lactate", "creatinine", "bilirubin", "platelets", 
                   "hemoglobin", "bun", "potassium", "sodium", "glucose", 
                   "bicarbonate", "chloride"]

for var in continuous_vars:
    if var in df.columns:
        df[var] = df[var].fillna(df[var].median())

# Fill categorical missingness
cat_vars = ["gender", "admission_type", "first_careunit"]
for var in cat_vars:
    df[var] = df[var].fillna("UNKNOWN")

# Binary conversions
binary_vars = ["sepsis", "comorb_htn", "comorb_diabetes", "comorb_ckd", "comorb_heart"]
for var in binary_vars:
    df[var] = np.where(df[var] == 1, "yes", "no")

# 3. Clinical Discretisation (Based on MEWS/SOFA/Normal ranges)

def clinical_cut(series, bins, labels):
    # bins must be strictly increasing, start with -inf, end with inf
    return pd.cut(series, bins=bins, labels=labels, right=False)

# Vitals
df["heart_rate"] = clinical_cut(df["heart_rate"], 
                                bins=[-np.inf, 50, 100, 130, np.inf], 
                                labels=["brady", "normal", "tachy", "severe_tachy"])

df["resp_rate"] = clinical_cut(df["resp_rate"], 
                               bins=[-np.inf, 12, 20, 30, np.inf], 
                               labels=["brady", "normal", "tachy", "severe_tachy"])

df["map"] = clinical_cut(df["map"], 
                         bins=[-np.inf, 65, 70, 110, np.inf], 
                         labels=["severe_hypo", "hypo", "normal", "hyper"])

df["temperature"] = clinical_cut(df["temperature"], 
                                 bins=[-np.inf, 36.0, 38.0, 39.0, np.inf], 
                                 labels=["hypo", "normal", "fever", "severe_fever"])

df["spo2"] = clinical_cut(df["spo2"], 
                          bins=[-np.inf, 90, 95, np.inf], 
                          labels=["severe_hypoxia", "hypoxia", "normal"])

# Demographics
df["age"] = clinical_cut(df["age"], 
                         bins=[-np.inf, 40, 65, 80, np.inf], 
                         labels=["young", "middle", "senior", "elderly"])

# Labs (SOFA criteria and standard lab ranges)
df["lactate"] = clinical_cut(df["lactate"], 
                             bins=[-np.inf, 2.0, 4.0, np.inf], 
                             labels=["normal", "elevated", "severe"])

df["creatinine"] = clinical_cut(df["creatinine"], 
                                bins=[-np.inf, 1.2, 2.0, 3.5, np.inf], 
                                labels=["normal", "mild", "mod", "severe"])

df["bilirubin"] = clinical_cut(df["bilirubin"], 
                               bins=[-np.inf, 1.2, 2.0, 6.0, np.inf], 
                               labels=["normal", "mild", "mod", "severe"])

df["platelets"] = clinical_cut(df["platelets"], 
                               bins=[-np.inf, 50, 100, 150, np.inf], 
                               labels=["severe_low", "mod_low", "mild_low", "normal"])

df["wbc"] = clinical_cut(df["wbc"], 
                         bins=[-np.inf, 4.0, 11.0, 20.0, np.inf], 
                         labels=["leukopenia", "normal", "mild_elev", "severe_elev"])

df["hemoglobin"] = clinical_cut(df["hemoglobin"], 
                                bins=[-np.inf, 7.0, 10.0, 13.0, np.inf], 
                                labels=["severe_anemia", "mod_anemia", "mild_anemia", "normal"])

df["glucose"] = clinical_cut(df["glucose"], 
                             bins=[-np.inf, 70, 140, 200, np.inf], 
                             labels=["hypo", "normal", "mild_hyper", "severe_hyper"])

# Fallback for remaining continuous variables (BUN, potassium, sodium, bicarb, chloride)
# We will use 3 equal-frequency bins (low, normal, high) for these to keep it simple but moderate.
remaining = ["bun", "potassium", "sodium", "bicarbonate", "chloride"]
for var in remaining:
    if var in df.columns:
        df[var] = pd.qcut(df[var], q=3, labels=["low", "med", "high"], duplicates='drop')

# 4. Save and Report
OUT_FILE = "rq3_clinical_extended_discretised.csv"
df.to_csv(OUT_FILE, index=False)

print(f"Dataset saved to {OUT_FILE} with {df.shape[1]} features.")
print("\nCardinalities:")
for col in df.columns:
    print(f"{col:<20} : {df[col].nunique()}")
