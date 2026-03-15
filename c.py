import pandas as pd
import glob

print("=== RQ1 COLUMNS ===")
try:
    df1 = pd.read_csv('RQ1_RESULTS_COMBINED/rq1_per_structure_summary.csv')
    print(df1.columns.tolist())
except Exception as e: print(e)

print("\n=== RQ2 COLUMNS ===")
try:
    rq2_files = glob.glob("RQ2_RESULTS1/rq2_summary.csv")
    if rq2_files:
        df2 = pd.read_csv(rq2_files[0])
        print(df2.columns.tolist())
    else:
        print("No rq2_summary.csv found in RQ2_RESULTS1")
except Exception as e: print(e)

print("\n=== RQ3 COLUMNS ===")
try:
    df3 = pd.read_csv('RQ3_RESULTS/rq3_mimic_results.csv')
    print("rq3_mimic_results.csv:", df3.columns.tolist())
    df3_node = pd.read_csv('RQ3_RESULTS/rq3_per_node_nll.csv')
    print("rq3_per_node_nll.csv:", df3_node.columns.tolist())
except Exception as e: print(e)
