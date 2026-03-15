import pandas as pd
import numpy as np
import scipy.stats as stats
import glob
import warnings
warnings.filterwarnings('ignore')

def get_95_ci(data):
    n = len(data)
    if n < 2: return np.mean(data), (np.mean(data), np.mean(data))
    mean = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=se)
    return mean, ci

print("="*60)
print("RQ1: WILCOXON TEST (High Complexity: C=6, D=7, N=1500)")
print("="*60)
try:
    df1 = pd.read_csv('RQ1_RESULTS_COMBINED/rq1_per_structure_summary.csv')
    hardest = df1[(df1['node_cardinality'] == 6) & 
                  (df1['max_indegree'] == 7) & 
                  (df1['dataset_size'] == 1500)]
    
    deep_kl = hardest['mean_kl_nn'].dropna().values
    tab_kl = hardest['mean_kl_tab'].dropna().values
    
    stat, p = stats.wilcoxon(deep_kl, tab_kl)
    d_mean, d_ci = get_95_ci(deep_kl)
    t_mean, t_ci = get_95_ci(tab_kl)
    
    print(f"Wilcoxon p-value : {p:.5e}")
    print(f"DeepCPD KL Mean  : {d_mean:.4f}, 95% CI: [{d_ci[0]:.4f}, {d_ci[1]:.4f}]")
    print(f"Tabular KL Mean  : {t_mean:.4f}, 95% CI: [{t_ci[0]:.4f}, {t_ci[1]:.4f}]")
except Exception as e: print(f"RQ1 Error: {e}")

print("\n" + "="*60)
print("RQ2: WILCOXON TEST (40 Nodes, N=1000)")
print("="*60)
try:
    rq2_files = glob.glob("RQ2_RESULTS*/rq2_summary.csv")
    df2 = pd.concat([pd.read_csv(f) for f in rq2_files], ignore_index=True)
    subset_rq2 = df2[(df2['dataset_size'] == 1000) & (df2['n_nodes'] == 40)]
    
    mbic_nll = subset_rq2[subset_rq2['method'].str.contains('mBIC', case=False, na=False)]['mean_test_NLL'].values
    bic_nll = subset_rq2[subset_rq2['method'].str.contains('BIC', case=False, na=False) & 
                         ~subset_rq2['method'].str.contains('mBIC', case=False, na=False)]['mean_test_NLL'].values
    
    min_len = min(len(mbic_nll), len(bic_nll))
    stat2, p2 = stats.wilcoxon(mbic_nll[:min_len], bic_nll[:min_len])
    m_mean, m_ci = get_95_ci(mbic_nll)
    b_mean, b_ci = get_95_ci(bic_nll)
    
    print(f"Wilcoxon p-value  : {p2:.5e}")
    print(f"DeepCPD mBIC NLL  : {m_mean:.4f}, 95% CI: [{m_ci[0]:.4f}, {m_ci[1]:.4f}]")
    print(f"Tabular BIC NLL   : {b_mean:.4f}, 95% CI: [{b_ci[0]:.4f}, {b_ci[1]:.4f}]")
except Exception as e: print(f"RQ2 Error: {e}")

print("\n" + "="*60)
print("RQ3: PAIRED T-TEST (MIMIC-IV per-node test NLL)")
print("="*60)
try:
    df3 = pd.read_csv('RQ3_RESULTS/rq3_per_node_nll.csv')
    # Filter to mBIC structure to compare tabular vs neural fairly
    df3_mbic = df3[df3['Structure'] == 'mBIC']
    
    deep_nll = df3_mbic[df3_mbic['CPD'].str.contains('Deep', case=False)]['NLL'].values
    tab_nll = df3_mbic[df3_mbic['CPD'].str.contains('Tab', case=False)]['NLL'].values
    
    min_len = min(len(deep_nll), len(tab_nll))
    stat3, p3 = stats.ttest_rel(tab_nll[:min_len], deep_nll[:min_len])
    
    diff = tab_nll[:min_len] - deep_nll[:min_len]
    mean_diff, ci_diff = get_95_ci(diff)
    
    print(f"Paired t-test p-value : {p3:.5e}")
    print(f"Difference (Delta)    : {mean_diff:.6f} (Tabular - DeepCPD)")
    print(f"95% CI for Delta      : [{ci_diff[0]:.6f}, {ci_diff[1]:.6f}]")
except Exception as e: print(f"RQ3 Error: {e}")
