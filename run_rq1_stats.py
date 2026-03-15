import pandas as pd
import numpy as np
import scipy.stats as stats
import os

def run_wilcoxon_tests():
    data_path = "RQ1_RESULTS_BASELINES/rq1_arch_sweep.csv"
    if not os.path.exists(data_path):
        print(f"Error: Could not find raw data at {data_path}")
        return

    print("Loading raw RQ1 data...")
    df = pd.read_csv(data_path)

    # Filter to the High Complexity Regime Boundary (N=1500, Card=6, Indeg=7)
    # Using the baseline architecture (depth=5, relu, adamw)
    subset = df[
        (df['dataset_size'] == 1500) &
        (df['node_cardinality'] == 6) &
        (df['max_indegree'] == 7) &
        (df['arch_depth'] == 5) &
        (df['activation'] == 'relu') &
        (df['optimizer'] == 'adamw')
    ]

    if subset.empty:
        print("Error: No data found for the target configuration (C=6, D=7, N=1500, Depth=5).")
        return

    # Extract paired metric arrays
    kl_nn = subset['kl_nn'].values
    kl_tab = subset['kl_tab'].values
    kl_dt = subset['kl_dt'].values
    kl_lr = subset['kl_lr'].values

    print("\n" + "="*65)
    print(" RQ1: WILCOXON SIGNED-RANK TESTS ON KL DIVERGENCE")
    print(" Regime Boundary: Cardinality=6, Max Indegree=7, N=1500")
    print("="*65)
    print(f"Sample size (paired node evaluations): {len(kl_nn)}")

    # 1. DeepCPD vs Tabular
    stat_tab, p_tab = stats.wilcoxon(kl_nn, kl_tab)
    mean_diff_tab = np.mean(kl_nn - kl_tab)
    print(f"\n[DeepCPD vs Tabular BN]")
    print(f"DeepCPD Mean KL: {np.mean(kl_nn):.4f}")
    print(f"Tabular Mean KL: {np.mean(kl_tab):.4f}")
    print(f"Mean Difference: {mean_diff_tab:.4f}")
    print(f"Wilcoxon p-value: {p_tab:.4e}")

    # 2. DeepCPD vs Decision Tree
    stat_dt, p_dt = stats.wilcoxon(kl_nn, kl_dt)
    mean_diff_dt = np.mean(kl_nn - kl_dt)
    print(f"\n[DeepCPD vs Decision Tree CPD]")
    print(f"DeepCPD Mean KL: {np.mean(kl_nn):.4f}")
    print(f"DT-CPD Mean KL:  {np.mean(kl_dt):.4f}")
    print(f"Mean Difference: {mean_diff_dt:.4f}")
    print(f"Wilcoxon p-value: {p_dt:.4e}")

    # 3. DeepCPD vs Logistic Regression
    stat_lr, p_lr = stats.wilcoxon(kl_nn, kl_lr)
    mean_diff_lr = np.mean(kl_nn - kl_lr)
    print(f"\n[DeepCPD vs Logistic Regression CPD]")
    print(f"DeepCPD Mean KL: {np.mean(kl_nn):.4f}")
    print(f"LR-CPD Mean KL:  {np.mean(kl_lr):.4f}")
    print(f"Mean Difference: {mean_diff_lr:.4f}")
    print(f"Wilcoxon p-value: {p_lr:.4e}")
    print("="*65)

if __name__ == "__main__":
    run_wilcoxon_tests()
