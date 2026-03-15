import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
import json

# 1. Load the exact same discretised dataset used for the BN
df = pd.read_csv("rq3_full_discretised.csv")

# Ensure sepsis is binary (0/1) for scikit-learn
if df['sepsis'].dtype == 'O':
    # Assuming 'yes'/'no' or similar based on your config
    df['sepsis'] = (df['sepsis'] == 'yes').astype(int)

# 2. Recreate the exact same train/test split from your config
# test_size=0.2, val_size=0.1 of train, random_seed=42
X = df.drop(columns=['sepsis'])
y = df['sepsis']

# Convert categorical variables to one-hot encoding for the baseline models
X_encoded = pd.get_dummies(X, columns=X.columns)

# First split: 80% train+val, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: 90% train, 10% val (from the 80% temp)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.1, random_state=42, stratify=y_temp
)

print(f"Training on {len(X_train)} rows, Testing on {len(X_test)} rows")
print(f"Sepsis prevalence in test set: {y_test.mean():.3f}")

# 3. Train Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
lr.fit(X_train, y_train)

lr_probs = lr.predict_proba(X_test)[:, 1]
lr_preds = lr.predict(X_test)

lr_prauc = average_precision_score(y_test, lr_probs)
lr_f1 = f1_score(y_test, lr_preds)
lr_prec = precision_score(y_test, lr_preds)
lr_rec = recall_score(y_test, lr_preds)

print(f"Logistic Regression -> PR-AUC: {lr_prauc:.4f}, F1: {lr_f1:.4f}, Prec: {lr_prec:.4f}, Rec: {lr_rec:.4f}")

# 4. Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)

rf_prauc = average_precision_score(y_test, rf_probs)
rf_f1 = f1_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds)
rf_rec = recall_score(y_test, rf_preds)

print(f"Random Forest -> PR-AUC: {rf_prauc:.4f}, F1: {rf_f1:.4f}, Prec: {rf_prec:.4f}, Rec: {rf_rec:.4f}")

# Compare to your BN results
print("\n--- Summary vs Bayesian Network ---")
print(f"mBIC + Tabular BN : PR-AUC = 0.2163, F1 = 0.3091")
print(f"mBIC + DeepCPD BN : PR-AUC = 0.2144, F1 = 0.3057")
print(f"Logistic Reg      : PR-AUC = {lr_prauc:.4f}, F1 = {lr_f1:.4f}")
print(f"Random Forest     : PR-AUC = {rf_prauc:.4f}, F1 = {rf_f1:.4f}")
