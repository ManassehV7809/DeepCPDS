import pandas as pd
import numpy as np
import scipy.stats as stats
import glob
import os

def get_95_ci(data):
    """Helper to calculate the 95% Confidence Interval"""
    n = len(data)
    if n < 2:
        return np.mean(data), (np.mean(data), np.mean(data))
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
    return mean, ci

print("="*60)
print("RQ1: WILCOXON TEST (High Complexity: C=6, D=7, N=1500)")
print("="*60)
try:
    # Look into your combined RQ1 results folder
    rq1_file = 'RQ1_RESULTS_COMBINED/rq1_per_structure_summary.csv'
    df1 = pd.read_csv(rq1_file)
    
    # Dynamically find column names just in case they differ slightly
    c_col = [c for c in df1.columns if 'card' in c.lower()][0]
    d_col = [c for c in df1.columns if 'indegree' in c.lower() or 'd' == c.lower()][0]
    n_col = [c for c in df1.columns if 'n' == c.lower() or 'size' in c.lower()][0]
    
    # Filter for the hardest complexity grid
    hardest = df1[(df1[c_col] == 6) & (df1[d_col] == 7) & (df1[n_col] == 1500)]
    
    deep_kl_col = [c for c in hardest.columns if 'deep' in c.lower() and 'kl' in c.lower()][0]
    tab_kl_col = [c for c in hardest.columns if 'tab' in c.lower() and 'kl' in c.lower()][0]
    
    deep_kl = hardest[deep_kl_col].dropna().values
    tab_kl = hardest[tab_kl_col].dropna().values
    
    stat, p = stats.wilcoxon(deep_kl, tab_kl)
    d_mean, d_ci = get_95_ci(deep_kl)
    t_mean, t_ci = get_95_ci(tab_kl)
    
    print(f"Wilcoxon p-value : {p:.5e}")
    print(f"DeepCPD KL Mean  : {d_mean:.4f}, 95% CI: [{d_ci[0]:.4f}, {d_ci[1]:.4f}]")
    print(f"Tabular KL Mean  : {t_mean:.4f}, 95% CI: [{t_ci[0]:.4f}, {t_ci[1]:.4f}]")
except Exception as e:
    print(f"Error computing RQ1: {e}")

print("\n" + "="*60)
print("RQ2: WILCOXON TEST (40 Nodes, k=4, max_in=5, N=1000)")
print("="*60)
try:
    # Read across all your RQ2 pilot/results folders to get all the runs
    rq2_files = glob.glob("RQ2_RESULTS*/rq2_summary.csv") + glob.glob("RQ2_RESULTS*/rq2_combined.csv")
    dfs = [pd.read_csv(f) for f in rq2_files]
    df2 = pd.concat(dfs, ignore_index=True)
    
    # Filter for N=1000, 40 nodes, cardinality 4, max indegree 5
    # (Update these string matches if your CSV columns are named differently)
    subset_rq2 = df2[(df2['N'] == 1000) & (df2['n'] == 40)]
    
    # Separate mBIC and BIC
    mbic_nll = subset_rq2[subset_rq2['Method'].str.contains('mBIC', case=False, na=False)]['mean NLL'].values
    bic_nll = subset_rq2[subset_rq2['Method'].str.contains('BIC', case=False, na=False) & 
                         ~subset_rq2['Method'].str.contains('mBIC', case=False, na=False)]['mean NLL'].values
    
    if len(mbic_nll) > 0 and len(bic_nll) > 0:
        # Match lengths for Wilcoxon
        min_len = min(len(mbic_nll), len(bic_nll))
        stat2, p2 = stats.wilcoxon(mbic_nll[:min_len], bic_nll[:min_len])
        m_mean, m_ci = get_95_ci(mbic_nll)
        b_mean, b_ci = get_95_ci(bic_nll)
        
        print(f"Wilcoxon p-value  : {p2:.5e}")
        print(f"DeepCPD mBIC NLL  : {m_mean:.4f}, 95% CI: [{m_ci[0]:.4f}, {m_ci[1]:.4f}]")
        print(f"Tabular BIC NLL   : {b_mean:.4f}, 95% CI: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
    else:
        print("Could not find matching RQ2 data rows for Wilcoxon.")
except Exception as e:
    print(f"Error computing RQ2: {e}")

print("\n" + "="*60)
print("RQ3: PAIRED T-TEST (MIMIC-IV Test Set NLL)")
print("="*60)
try:
    # Usually the raw arrays are kept in rq3_results.csv or rq3_per_node_nll.csv
    # We will try rq3_per_node_nll.csv first
    rq3_file = 'RQ3_RESULTS/rq3_per_node_nll.csv'
    df3 = pd.read_csv(rq3_file)
    
    # Assuming this CSV contains a row per test sample, with columns for Tabular_NLL and DeepCPD_NLL
    deep_col = [c for c in df3.columns if 'deep' in c.lower() and 'nll' in c.lower()][0]
    tab_col = [c for c in df3.columns if 'tab' in c.lower() and 'nll' in c.lower()][0]
    
    deep_nll = df3[deep_col].values
    tab_nll = df3[tab_col].values
    
    stat3, p3 = stats.ttest_rel(tab_nll, deep_nll)
    
    # Delta = Tabular - DeepCPD
    diff = tab_nll - deep_nll
    mean_diff, ci_diff = get_95_ci(diff)
    
    print(f"Paired t-test p-value : {p3:.5e}")
    print(f"Difference (Delta)    : {mean_diff:.6f}")
    print(f"95% CI for Delta      : [{ci_diff[0]:.6f}, {ci_diff[1]:.6f}]")
except Exception as e:
    print(f"Error computing RQ3: {e}")
    print("Note: If RQ3 per-row NLL wasn't saved in 'rq3_per_node_nll.csv', you may need to extract it from 'rq3_mimic_results.csv' or rerun the test loop with NLL arrays saved.")
