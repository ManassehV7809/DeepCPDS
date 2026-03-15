import os
import math
import time
import random
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, average_precision_score

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except Exception:
    from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator, HillClimbSearch, BicScore


device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"[info] using device: {device}")


CONFIG = {
    "mimic_csv_path":            "rq3_clinical_extended_discretised.csv",
    "target_node":               "sepsis",
    "target_state":              "yes",
    "test_size":                 0.20,
    "val_size":                  0.10,
    "random_seed":               42,
    "max_indegree":              4,
    "mbic_lambda":               2.0,
    "epochs":                    80,
    "patience":                  15,
    "lr":                        1e-3,
    "weight_decay":              1e-4,
    "dropout":                   0.15,
    "batch_size":                256,
    "sepsis_sample_size":        500,
    "tabular_prior_type":        "BDeu",
    "tabular_equiv_sample_size": 5.0,
}

SUBSAMPLE_SIZES = [500, 1000, 5000]
N_SEEDS         = 3
SWEEP_SEEDS     = [42, 123, 7]

PRIMARY_ARCH = {"layers": [12, 10, 8], "label": "Shallow_RQ1"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


class MBicScore(BicScore):
    def __init__(self, data, lambda_=2.0, **kwargs):
        super().__init__(data, **kwargs)
        self.lambda_ = lambda_

    def local_score(self, variable, parents):
        base    = super().local_score(variable, parents)
        N       = len(self.data)
        k       = self._count_free_params(variable, list(parents))
        penalty = (math.log(N) / 2.0) * k
        ll      = base + penalty
        return ll - self.lambda_ * penalty

    def _count_free_params(self, variable, parents):
        states      = {col: self.data[col].nunique() for col in self.data.columns}
        parent_card = 1
        for p in parents:
            parent_card *= states[p]
        return (states[variable] - 1) * parent_card


class CPDNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, layers, dropout=0.15):
        super().__init__()
        blocks      = []
        in_features = input_dim
        for units in layers:
            blocks += [
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_features = units
        blocks.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


def build_tensors(df, node, parents, cardinalities):
    if parents:
        parts = []
        for p in parents:
            oh   = np.zeros((len(df), cardinalities[p]), dtype=np.float32)
            # clamp parent values to valid range — UNK index would be out of bounds
            pvals = np.clip(df[p].values.astype(int), 0, cardinalities[p] - 1)
            oh[np.arange(len(df)), pvals] = 1.0
            parts.append(oh)
        X = np.concatenate(parts, axis=1)
    else:
        X = np.zeros((len(df), 1), dtype=np.float32)

    # clamp labels to valid range — UNK index would be out of bounds
    y   = np.clip(df[node].values.astype(np.int64), 0, cardinalities[node] - 1)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)
    return X_t, y_t


def train_node(X_tr, y_tr, X_va, y_va, input_dim, output_dim, arch):
    model     = CPDNetwork(input_dim, output_dim,
                           arch["layers"], CONFIG["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_tr, y_tr),
        batch_size=CONFIG["batch_size"], shuffle=True)

    best_val   = float("inf")
    best_state = None
    patience   = 0

    for _ in range(CONFIG["epochs"]):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_va), y_va).item()

        if val_loss < best_val - 1e-8:
            best_val   = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1
            if patience >= CONFIG["patience"]:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


def fit_neural_bn(edges, train_int, val_int, cardinalities, arch):
    dag    = BayesianNetwork(edges)
    models = {}
    for node in dag.nodes():
        parents = list(dag.get_parents(node))
        X_tr, y_tr = build_tensors(train_int, node, parents, cardinalities)
        X_va, y_va = build_tensors(val_int,   node, parents, cardinalities)
        models[node] = train_node(
            X_tr, y_tr, X_va, y_va,
            X_tr.shape[1], cardinalities[node], arch
        )
    return dag, models


def fit_tabular_bn(edges, train_int, state_names_dict):
    bn = BayesianNetwork(edges)
    bn.fit(
        train_int,
        estimator=BayesianEstimator,
        prior_type=CONFIG["tabular_prior_type"],
        equivalent_sample_size=CONFIG["tabular_equiv_sample_size"],
        state_names=state_names_dict,
    )
    return bn


def compute_nll_tabular(tab_bn, test_int):
    nodes     = list(tab_bn.nodes())
    total_nll = 0.0
    N         = len(test_int)

    for node in nodes:
        parents  = list(tab_bn.get_parents(node))
        node_nll = 0.0
        for _, row in test_int.iterrows():
            y         = int(row[node])
            ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist())) if parents else []
            cpd_vals  = tab_bn.get_cpds(node).reduce(ev_tuples, inplace=False).values.flatten()
            # clamp y to valid CPD range
            y_safe    = min(y, len(cpd_vals) - 1)
            p_y       = max(float(cpd_vals[y_safe]), 1e-12)
            node_nll += -math.log(p_y)
        total_nll += node_nll / N

    return total_nll / len(nodes)


def compute_nll_neural(dag, models, test_int, cardinalities):
    nodes     = list(dag.nodes())
    total_nll = 0.0

    for node in nodes:
        parents    = list(dag.get_parents(node))
        X_te, y_te = build_tensors(test_int, node, parents, cardinalities)
        model      = models[node]
        model.eval()
        with torch.no_grad():
            log_p    = torch.log_softmax(model(X_te), dim=-1)
            node_nll = -log_p[torch.arange(len(y_te)), y_te].mean().item()
        total_nll += node_nll

    return total_nll / len(nodes)


def compute_sepsis_metrics_tabular(tab_bn, test_int, val_int, target_node, target_state_int):
    parents = list(tab_bn.get_parents(target_node))
    if not parents:
        return {"sepsis_nll": None, "pr_auc": None, "precision": None, "recall": None, "f1": None}

    sample_test = test_int.sample(
        min(CONFIG["sepsis_sample_size"], len(test_int)), random_state=42
    ).reset_index(drop=True)

    y_true_test = (sample_test[target_node] == target_state_int).astype(int).values
    probs_test  = []
    cpd = tab_bn.get_cpds(target_node)

    for _, row in sample_test.iterrows():
        ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist()))
        vals      = cpd.reduce(ev_tuples, inplace=False).values.flatten()
        idx       = min(target_state_int, len(vals) - 1)
        probs_test.append(max(float(vals[idx]), 1e-12))

    probs_test = np.array(probs_test)
    pr_auc     = average_precision_score(y_true_test, probs_test)

    sample_val = val_int.sample(
        min(CONFIG["sepsis_sample_size"], len(val_int)), random_state=43
    ).reset_index(drop=True)
    y_true_val = (sample_val[target_node] == target_state_int).astype(int).values
    probs_val  = []

    for _, row in sample_val.iterrows():
        ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist()))
        vals      = cpd.reduce(ev_tuples, inplace=False).values.flatten()
        idx       = min(target_state_int, len(vals) - 1)
        probs_val.append(max(float(vals[idx]), 1e-12))

    probs_val = np.array(probs_val)

    thresholds  = np.linspace(0.01, 0.99, 99)
    best_f1     = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (probs_val >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true_val, preds, average='binary', zero_division=0)
        if f > best_f1:
            best_f1     = f
            best_thresh = thresh

    preds_test = (probs_test >= best_thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_test, preds_test, average='binary', zero_division=0)

    nll_sepsis = 0.0
    for _, row in sample_test.iterrows():
        y         = int(row[target_node])
        ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist()))
        vals      = cpd.reduce(ev_tuples, inplace=False).values.flatten()
        y_safe    = min(y, len(vals) - 1)
        p_y       = max(float(vals[y_safe]), 1e-12)
        nll_sepsis += -math.log(p_y)
    nll_sepsis /= len(sample_test)

    return {
        "sepsis_nll": round(nll_sepsis, 4),
        "pr_auc":     round(pr_auc, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
    }


def compute_sepsis_metrics_neural(dag, models, test_int, val_int, cardinalities,
                                   target_node, target_state_int):
    parents = list(dag.get_parents(target_node))
    if not parents:
        return {"sepsis_nll": None, "pr_auc": None, "precision": None, "recall": None, "f1": None}

    sample_test = test_int.sample(
        min(CONFIG["sepsis_sample_size"], len(test_int)), random_state=42
    ).reset_index(drop=True)

    X_te, y_te = build_tensors(sample_test, target_node, parents, cardinalities)
    model      = models[target_node]
    model.eval()

    with torch.no_grad():
        probs_test = torch.softmax(model(X_te), dim=-1)
        # clamp target_state_int to valid output range
        idx        = min(target_state_int, probs_test.shape[1] - 1)
        probs_test = probs_test[:, idx].cpu().numpy()
        log_p      = torch.log_softmax(model(X_te), dim=-1)
        nll_sepsis = -log_p[torch.arange(len(y_te)), y_te].mean().item()

    y_true_test = (sample_test[target_node] == target_state_int).astype(int).values
    pr_auc      = average_precision_score(y_true_test, probs_test)

    sample_val = val_int.sample(
        min(CONFIG["sepsis_sample_size"], len(val_int)), random_state=43
    ).reset_index(drop=True)
    X_va, y_va = build_tensors(sample_val, target_node, parents, cardinalities)

    with torch.no_grad():
        probs_val = torch.softmax(model(X_va), dim=-1)[:, idx].cpu().numpy()

    y_true_val = (sample_val[target_node] == target_state_int).astype(int).values

    thresholds  = np.linspace(0.01, 0.99, 99)
    best_f1     = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (probs_val >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        p, r, f, _ = precision_recall_fscore_support(
            y_true_val, preds, average='binary', zero_division=0)
        if f > best_f1:
            best_f1     = f
            best_thresh = thresh

    preds_test = (probs_test >= best_thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_test, preds_test, average='binary', zero_division=0)

    return {
        "sepsis_nll": round(nll_sepsis, 4),
        "pr_auc":     round(pr_auc, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
    }


def learn_structure(train_str, score_name):
    scoring = (
        BicScore(train_str) if score_name == "BIC"
        else MBicScore(train_str, lambda_=CONFIG["mbic_lambda"])
    )
    hc  = HillClimbSearch(train_str)
    dag = hc.estimate(
        scoring_method=scoring,
        max_indegree=CONFIG["max_indegree"],
        max_iter=int(1e4),
        show_progress=False,
    )
    return list(dag.edges())


def structural_hamming_distance(edges_a, edges_b):
    set_a = set(edges_a)
    set_b = set(edges_b)
    shd   = 0
    for edge in set_a | set_b:
        if edge in set_a and edge in set_b:
            continue
        rev = (edge[1], edge[0])
        if (edge in set_a and rev in set_b) or (edge in set_b and rev in set_a):
            shd += 1
        else:
            shd += 1
    return shd


def fmt(val):
    return f"{val:.4f}" if val is not None else "N/A"


def subsample_stratified(df_int, size, target_node, seed):
    try:
        subset = df_int.groupby(target_node, group_keys=False).apply(
            lambda x: x.sample(
                min(len(x), max(1, int(round(size * len(x) / len(df_int))))),
                random_state=seed,
            ),
            include_groups=True,  # silence pandas deprecation warning
        )
        if len(subset) > size:
            subset = subset.sample(size, random_state=seed)
        elif len(subset) < size:
            extra  = df_int.drop(subset.index).sample(size - len(subset), random_state=seed)
            subset = pd.concat([subset, extra])
    except Exception:
        subset = df_int.sample(size, random_state=seed)
    return subset.reset_index(drop=True)


def run_rq3_scarce(output_dir="RQ3_SCARCE_RESULTS"):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(CONFIG["random_seed"])

    print("\n" + "=" * 65)
    print("RQ3-SCARCE: End-to-end 2x2 factorial at reduced data sizes")
    print("Sizes: 500, 1000, 5000 | BIC + mBIC | Tabular + DeepCPD")
    print("Test set fixed = same split as rq3.py")
    print("=" * 65)

    df_str       = pd.read_csv(CONFIG["mimic_csv_path"])
    target_node  = CONFIG["target_node"]
    target_state = CONFIG["target_state"]

    print(f"[scarce] Shape: {df_str.shape}")
    print(f"[scarce] Sepsis prevalence: "
          f"{(df_str[target_node] == target_state).mean()*100:.1f}%")

    train_val_str, test_str = train_test_split(
        df_str, test_size=CONFIG["test_size"],
        random_state=CONFIG["random_seed"],
        stratify=df_str[target_node],
    )
    val_frac = CONFIG["val_size"] / (1.0 - CONFIG["test_size"])
    train_str, val_str = train_test_split(
        train_val_str, test_size=val_frac,
        random_state=CONFIG["random_seed"],
        stratify=train_val_str[target_node],
    )
    print(f"[scarce] Full train: {len(train_str):,}  "
          f"Val: {len(val_str):,}  Test: {len(test_str):,}")

    # encode using full train vocabulary
    cat_to_int = {}
    unk_id     = {}
    for col in df_str.columns:
        cats            = sorted(train_str[col].dropna().unique().tolist())
        cat_to_int[col] = {c: i for i, c in enumerate(cats)}
        unk_id[col]     = len(cat_to_int[col])   # captured BEFORE __UNK__
        cat_to_int[col]["__UNK__"] = unk_id[col]

    int_to_cat = {
        col: {v: k for k, v in cat_to_int[col].items() if k != "__UNK__"}
        for col in df_str.columns
    }

    def encode(df):
        out = df.copy()
        for col in df.columns:
            out[col] = out[col].map(cat_to_int[col]).fillna(unk_id[col]).astype(int)
        return out

    def decode_to_str(df_int):
        out = df_int.copy()
        for col in df_int.columns:
            out[col] = df_int[col].map(int_to_cat[col])
        return out

    train_int = encode(train_str)
    val_int   = encode(val_str)
    test_int  = encode(test_str)

    # FIXED: use unk_id (cardinality before __UNK__ was added)
    cardinalities    = {col: unk_id[col] for col in df_str.columns}
    target_state_int = cat_to_int[target_node][target_state]
    state_names_dict = {col: list(range(cardinalities[col])) for col in df_str.columns}

    print(f"[scarce] Cardinalities: {cardinalities}")

    # reference structures from full data
    print("\n[scarce] Learning reference structures on full training data...")
    ref_edges = {}
    for score_name in ["BIC", "mBIC"]:
        ref_edges[score_name] = learn_structure(train_str, score_name)
        sep = [(u, v) for u, v in ref_edges[score_name] if target_node in (u, v)]
        print(f"[scarce] Reference {score_name}: {len(ref_edges[score_name])} edges  "
              f"sepsis_edges={sep}")

    ref_shd = structural_hamming_distance(ref_edges["BIC"], ref_edges["mBIC"])
    print(f"[scarce] Reference SHD (BIC vs mBIC, full data): {ref_shd}")

    all_records = []

    for size in SUBSAMPLE_SIZES:
        print(f"\n[scarce] {'='*55}")
        print(f"[scarce] TRAIN SIZE = {size:,}")
        print(f"[scarce] {'='*55}")

        seed_records = {
            score: {cpd: [] for cpd in ["Tabular", "DeepCPD"]}
            for score in ["BIC", "mBIC"]
        }

        for seed in SWEEP_SEEDS:
            print(f"\n[scarce] ── seed={seed} ──────────────────────────────────")

            subset_int = subsample_stratified(train_int, size, target_node, seed)
            subset_str = decode_to_str(subset_int)

            print(f"[scarce]   Subset size: {len(subset_int):,}  "
                  f"Sepsis%: {(subset_int[target_node]==target_state_int).mean()*100:.1f}%")

            structures = {}
            for score_name in ["BIC", "mBIC"]:
                set_seed(seed)
                try:
                    edges = learn_structure(subset_str, score_name)
                except Exception as e:
                    print(f"[scarce]   {score_name} structure failed: {e}")
                    edges = []
                structures[score_name] = edges
                sep        = [(u, v) for u, v in edges if target_node in (u, v)]
                shd_to_ref = structural_hamming_distance(edges, ref_edges[score_name])
                print(f"[scarce]   {score_name}: {len(edges)} edges  "
                      f"SHD_to_ref={shd_to_ref}  sepsis_edges={sep}")

            for score_name in ["BIC", "mBIC"]:
                edges = structures[score_name]
                if not edges:
                    for cpd_name in ["Tabular", "DeepCPD"]:
                        seed_records[score_name][cpd_name].append(
                            {"nll": float("nan"), "sepsis_metrics": {}}
                        )
                    continue

                # Tabular
                set_seed(seed)
                try:
                    tab_bn  = fit_tabular_bn(edges, subset_int, state_names_dict)
                    tab_nll = compute_nll_tabular(tab_bn, test_int)
                    tab_sep = compute_sepsis_metrics_tabular(
                        tab_bn, test_int, val_int, target_node, target_state_int)
                except Exception as e:
                    print(f"[scarce]   {score_name}+Tabular failed: {e}")
                    tab_nll = float("nan")
                    tab_sep = {"sepsis_nll": None, "pr_auc": None,
                               "precision": None, "recall": None, "f1": None}

                print(f"[scarce]   {score_name}+Tabular  NLL={fmt(tab_nll)}  "
                      f"Sepsis_NLL={fmt(tab_sep['sepsis_nll'])}  "
                      f"PR-AUC={fmt(tab_sep['pr_auc'])}")
                seed_records[score_name]["Tabular"].append(
                    {"nll": tab_nll, "sepsis_metrics": tab_sep})

                # DeepCPD
                set_seed(seed)
                try:
                    dag, models = fit_neural_bn(
                        edges, subset_int, val_int, cardinalities, PRIMARY_ARCH)
                    deep_nll = compute_nll_neural(dag, models, test_int, cardinalities)
                    deep_sep = compute_sepsis_metrics_neural(
                        dag, models, test_int, val_int,
                        cardinalities, target_node, target_state_int)
                except Exception as e:
                    print(f"[scarce]   {score_name}+DeepCPD failed: {e}")
                    deep_nll = float("nan")
                    deep_sep = {"sepsis_nll": None, "pr_auc": None,
                                "precision": None, "recall": None, "f1": None}

                print(f"[scarce]   {score_name}+DeepCPD  NLL={fmt(deep_nll)}  "
                      f"Sepsis_NLL={fmt(deep_sep['sepsis_nll'])}  "
                      f"PR-AUC={fmt(deep_sep['pr_auc'])}")
                seed_records[score_name]["DeepCPD"].append(
                    {"nll": deep_nll, "sepsis_metrics": deep_sep})

        # aggregate across seeds
        print(f"\n[scarce] ── Aggregated results (size={size:,}) ────────────")
        for score_name in ["BIC", "mBIC"]:
            for cpd_name in ["Tabular", "DeepCPD"]:
                runs = seed_records[score_name][cpd_name]
                nlls = [r["nll"] for r in runs]

                nll_mean = float(np.nanmean(nlls))
                nll_std  = float(np.nanstd(nlls))

                sep_keys  = ["sepsis_nll", "pr_auc", "precision", "recall", "f1"]
                sep_means = {}
                for k in sep_keys:
                    vals = [r["sepsis_metrics"].get(k) for r in runs
                            if r["sepsis_metrics"].get(k) is not None]
                    sep_means[k] = round(float(np.mean(vals)), 4) if vals else None

                print(f"[scarce]   {score_name}+{cpd_name}:  "
                      f"NLL={nll_mean:.4f}±{nll_std:.4f}  "
                      f"PR-AUC={fmt(sep_means['pr_auc'])}")

                all_records.append({
                    "Train_size":      size,
                    "Structure":       score_name,
                    "CPD":             cpd_name,
                    "Label":           f"{score_name}+{cpd_name}",
                    "NLL_mean":        round(nll_mean, 4),
                    "NLL_std":         round(nll_std,  4),
                    "Sepsis_NLL_mean": sep_means["sepsis_nll"],
                    "PR_AUC_mean":     sep_means["pr_auc"],
                    "Precision_mean":  sep_means["precision"],
                    "Recall_mean":     sep_means["recall"],
                    "F1_mean":         sep_means["f1"],
                    "N_seeds":         N_SEEDS,
                })

    # save CSV
    results_df = pd.DataFrame(all_records)
    csv_path   = os.path.join(output_dir, "rq3_scarce_results.csv")
    results_df.to_csv(csv_path, index=False)

    print("\n" + "=" * 65)
    print("RQ3-SCARCE FINAL RESULTS TABLE")
    print("=" * 65)
    print(results_df.to_string(index=False))

    # plots
    try:
        import matplotlib.pyplot as plt

        for score_name in ["BIC", "mBIC"]:
            subset = results_df[results_df["Structure"] == score_name]
            fig, ax = plt.subplots(figsize=(7, 4))

            for cpd_name, marker in [("Tabular", "o"), ("DeepCPD", "s")]:
                cpd_sub = subset[subset["CPD"] == cpd_name]
                ax.errorbar(
                    cpd_sub["Train_size"],
                    cpd_sub["NLL_mean"],
                    yerr=cpd_sub["NLL_std"],
                    label=cpd_name, marker=marker, capsize=4,
                )

            ax.set_xscale("log")
            ax.set_xlabel("Training set size (log scale)")
            ax.set_ylabel("Mean Test NLL (avg over nodes)")
            ax.set_title(
                f"DeepCPD vs Tabular — Data Scarcity ({score_name}, MIMIC-IV)\n"
                f"End-to-end: structure + CPDs learned on each subset"
            )
            ax.legend()
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            plt.tight_layout()
            png_path = os.path.join(output_dir, f"rq3_scarce_curve_{score_name.lower()}.png")
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"[scarce] Plot saved → {png_path}")

    except Exception as e:
        print(f"[scarce] Plots skipped: {e}")

    # metadata
    json.dump(
        {
            "note":            "Full end-to-end RQ3 experiment at each subsample size. "
                               "Structure AND CPDs learned on subset. Test set fixed.",
            "subsample_sizes": SUBSAMPLE_SIZES,
            "score_names":     ["BIC", "mBIC"],
            "cpd_names":       ["Tabular", "DeepCPD"],
            "n_seeds":         N_SEEDS,
            "sweep_seeds":     SWEEP_SEEDS,
            "cardinalities":   cardinalities,
            "ref_shd_bic_vs_mbic_full_data": ref_shd,
            "split_sizes":     {"train": len(train_str),
                                "val":   len(val_str),
                                "test":  len(test_str)},
            "config":          CONFIG,
        },
        open(os.path.join(output_dir, "rq3_scarce_summary.json"), "w"), indent=2
    )

    print(f"\n[scarce] All results saved to {output_dir}/")
    print("=" * 65)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="RQ3_SCARCE_RESULTS")
    args = parser.parse_args()
    run_rq3_scarce(args.output_dir)
