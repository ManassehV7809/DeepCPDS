import os
import glob
import pandas as pd
import argparse


def combine_rq2_results(input_dir="RQ2_RESULTS", output_path=None):
    """
    Combine all rq2_task_*.csv files in input_dir into a single CSV.

    Assumes each file has the same columns, e.g.:
      task_id, graph_seed, cpd_type, penalty_type, label, shd, edge_count, true_edge_count
    """
    pattern = os.path.join(input_dir, "rq2_task_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    if output_path is None:
        output_path = os.path.join(input_dir, "rq2_results_combined.csv")

    combined.to_csv(output_path, index=False)
    print(f"Combined {len(files)} files into {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="RQ2_RESULTS")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    combine_rq2_results(input_dir=args.input_dir, output_path=args.output_path)

