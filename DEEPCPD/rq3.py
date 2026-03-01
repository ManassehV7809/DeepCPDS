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
    "mimic_csv_path":            "rq3_full_discretised.csv",
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
    "sepsis_sample_size":        3000,
    "tabular_prior_type":        "BDeu",
    "tabular_equiv_sample_size": 5.0,
}


ARCH_SWEEP = [
    {"layers": [12, 10, 8],         "label": "Shallow_RQ1"},
    {"layers": [32, 64, 32],        "label": "Wide"},
    {"layers": [64, 128, 64, 32],   "label": "Wide_Deep"},
    {"layers": [32, 32, 32, 32],    "label": "Uniform_Deep"},
    {"layers": [128, 64, 32, 16],   "label": "Funnel"},
]


PRIMARY_ARCH = ARCH_SWEEP[0]



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


def fmt(val):
    """Safe formatter — returns 'N/A' for None, 4dp string otherwise."""
    return f"{val:.4f}" if val is not None else "N/A"


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
        blocks    = []
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
            oh = np.zeros((len(df), cardinalities[p]), dtype=np.float32)
            oh[np.arange(len(df)), df[p].values.astype(int)] = 1.0
            parts.append(oh)
        X = np.concatenate(parts, axis=1)
    else:
        X = np.zeros((len(df), 1), dtype=np.float32)


    y = df[node].values.astype(np.int64)
    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)
    return X_t, y_t



def train_node(X_tr, y_tr, X_va, y_va, input_dim, output_dim, arch):
    model     = CPDNetwork(input_dim, output_dim,
                           arch["layers"], CONFIG["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])


    loader     = torch.utils.data.DataLoader(
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
        parents   = list(tab_bn.get_parents(node))
        node_nll  = 0.0
        for _, row in test_int.iterrows():
            y         = int(row[node])
            ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist())) if parents else []
            cpd_vals  = tab_bn.get_cpds(node).reduce(ev_tuples, inplace=False).values.flatten()
            p_y       = max(float(cpd_vals[y]), 1e-12)
            node_nll += -math.log(p_y)
        total_nll += node_nll / N


    return total_nll / len(nodes)



def compute_nll_neural(dag, models, test_int, cardinalities):
    nodes     = list(dag.nodes())
    total_nll = 0.0
    N         = len(test_int)


    for node in nodes:
        parents  = list(dag.get_parents(node))
        X_te, y_te = build_tensors(test_int, node, parents, cardinalities)
        model    = models[node]
        model.eval()
        with torch.no_grad():
            log_p    = torch.log_softmax(model(X_te), dim=-1)
            node_nll = -log_p[torch.arange(len(y_te)), y_te].mean().item()
        total_nll += node_nll


    return total_nll / len(nodes)



def compute_per_node_nll_tabular(tab_bn, test_int):
    nodes      = list(tab_bn.nodes())
    node_nlls  = {}
    N          = len(test_int)


    for node in nodes:
        parents   = list(tab_bn.get_parents(node))
        node_nll  = 0.0
        for _, row in test_int.iterrows():
            y         = int(row[node])
            ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist())) if parents else []
            cpd_vals  = tab_bn.get_cpds(node).reduce(ev_tuples, inplace=False).values.flatten()
            p_y       = max(float(cpd_vals[y]), 1e-12)
            node_nll += -math.log(p_y)
        node_nlls[node] = node_nll / N


    return node_nlls



def compute_per_node_nll_neural(dag, models, test_int, cardinalities):
    nodes     = list(dag.nodes())
    node_nlls = {}


    for node in nodes:
        parents  = list(dag.get_parents(node))
        X_te, y_te = build_tensors(test_int, node, parents, cardinalities)
        model    = models[node]
        model.eval()
        with torch.no_grad():
            log_p    = torch.log_softmax(model(X_te), dim=-1)
            node_nll = -log_p[torch.arange(len(y_te)), y_te].mean().item()
        node_nlls[node] = node_nll


    return node_nlls



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
        vals = cpd.reduce(ev_tuples, inplace=False).values.flatten()
        probs_test.append(max(float(vals[target_state_int]), 1e-12))


    probs_test = np.array(probs_test)
    pr_auc = average_precision_score(y_true_test, probs_test)


    sample_val = val_int.sample(
        min(CONFIG["sepsis_sample_size"], len(val_int)), random_state=43
    ).reset_index(drop=True)
    y_true_val = (sample_val[target_node] == target_state_int).astype(int).values
    probs_val  = []


    for _, row in sample_val.iterrows():
        ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist()))
        vals = cpd.reduce(ev_tuples, inplace=False).values.flatten()
        probs_val.append(max(float(vals[target_state_int]), 1e-12))


    probs_val = np.array(probs_val)


    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (probs_val >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        p, r, f, _ = precision_recall_fscore_support(y_true_val, preds, average='binary', zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh


    preds_test = (probs_test >= best_thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_test, preds_test, average='binary', zero_division=0
    )


    nll_sepsis = 0.0
    for _, row in sample_test.iterrows():
        y = int(row[target_node])
        ev_tuples = list(zip(parents, row[parents].values.astype(int).tolist()))
        vals = cpd.reduce(ev_tuples, inplace=False).values.flatten()
        p_y = max(float(vals[y]), 1e-12)
        nll_sepsis += -math.log(p_y)
    nll_sepsis /= len(sample_test)


    return {
        "sepsis_nll": round(nll_sepsis, 4),
        "pr_auc":     round(pr_auc, 4),
        "precision":  round(precision, 4),
        "recall":     round(recall, 4),
        "f1":         round(f1, 4),
    }



def compute_sepsis_metrics_neural(dag, models, test_int, val_int, cardinalities, target_node, target_state_int):
    parents = list(dag.get_parents(target_node))
    if not parents:
        return {"sepsis_nll": None, "pr_auc": None, "precision": None, "recall": None, "f1": None}


    sample_test = test_int.sample(
        min(CONFIG["sepsis_sample_size"], len(test_int)), random_state=42
    ).reset_index(drop=True)


    X_te, y_te = build_tensors(sample_test, target_node, parents, cardinalities)
    model = models[target_node]
    model.eval()


    with torch.no_grad():
        probs_test = torch.softmax(model(X_te), dim=-1)[:, target_state_int].cpu().numpy()
        log_p = torch.log_softmax(model(X_te), dim=-1)
        nll_sepsis = -log_p[torch.arange(len(y_te)), y_te].mean().item()


    y_true_test = (sample_test[target_node] == target_state_int).astype(int).values
    pr_auc = average_precision_score(y_true_test, probs_test)


    sample_val = val_int.sample(
        min(CONFIG["sepsis_sample_size"], len(val_int)), random_state=43
    ).reset_index(drop=True)
    X_va, y_va = build_tensors(sample_val, target_node, parents, cardinalities)


    with torch.no_grad():
        probs_val = torch.softmax(model(X_va), dim=-1)[:, target_state_int].cpu().numpy()


    y_true_val = (sample_val[target_node] == target_state_int).astype(int).values


    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        preds = (probs_val >= thresh).astype(int)
        if preds.sum() == 0:
            continue
        p, r, f, _ = precision_recall_fscore_support(y_true_val, preds, average='binary', zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh


    preds_test = (probs_test >= best_thresh).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_test, preds_test, average='binary', zero_division=0
    )


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



def run_rq3(output_dir="RQ3_RESULTS"):
    os.makedirs(output_dir, exist_ok=True)
    set_seed(CONFIG["random_seed"])


    print("\n" + "=" * 65)
    print("RQ3: MIMIC-IV 2x2 FACTORIAL — Structure x CPD")
    print("=" * 65)


    df_str = pd.read_csv(CONFIG["mimic_csv_path"])
    print(f"[RQ3] Shape: {df_str.shape}")
    print(f"[RQ3] Sepsis prevalence: "
          f"{(df_str[CONFIG['target_node']] == CONFIG['target_state']).mean()*100:.1f}%")


    target_node  = CONFIG["target_node"]
    target_state = CONFIG["target_state"]


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
    print(f"[RQ3] Train: {len(train_str):,}  Val: {len(val_str):,}  "
          f"Test: {len(test_str):,}")


    cat_to_int = {}
    unk_id = {}


    for col in df_str.columns:
        cats = sorted(train_str[col].dropna().unique().tolist())
        cat_to_int[col] = {c: i for i, c in enumerate(cats)}
        unk_id[col] = len(cat_to_int[col])
        cat_to_int[col]["__UNK__"] = unk_id[col]


    def encode(df):
        out = df.copy()
        unk_counts = {}
        for col in df.columns:
            mapped = out[col].map(cat_to_int[col])
            n_unk = mapped.isna().sum()
            unk_counts[col] = n_unk
            out[col] = mapped.fillna(unk_id[col]).astype(int)
        return out, unk_counts


    train_int, unk_tr = encode(train_str)
    val_int, unk_va   = encode(val_str)
    test_int, unk_te  = encode(test_str)


    print(f"[RQ3] Unknown categories: Train={sum(unk_tr.values())}, "
          f"Val={sum(unk_va.values())}, Test={sum(unk_te.values())}")


    cardinalities    = {col: len(cat_to_int[col]) for col in df_str.columns}
    target_state_int = cat_to_int[target_node][target_state]
    state_names_dict = {col: list(range(cardinalities[col])) for col in df_str.columns}


    print(f"[RQ3] Node cardinalities: {cardinalities}")


    print("\n[RQ3] ── Structure Learning ──────────────────────────────────")
    structures = {}
    for score_name in ["BIC", "mBIC"]:
        print(f"[RQ3] Running Hill Climbing with {score_name}...")
        t0    = time.time()
        edges = learn_structure(train_str, score_name)
        structures[score_name] = edges
        sep   = [(u, v) for u, v in edges if target_node in (u, v)]
        print(f"[RQ3] {score_name}: {len(edges)} edges  ({time.time()-t0:.1f}s)")
        print(f"[RQ3] Edges involving '{target_node}': {sep}")


    shd = structural_hamming_distance(structures["BIC"], structures["mBIC"])
    print(f"\n[RQ3] SHD (BIC vs mBIC): {shd}")


    with open(os.path.join(output_dir, "bic_edges.pkl"),  "wb") as f:
        pickle.dump(structures["BIC"],  f)
    with open(os.path.join(output_dir, "mbic_edges.pkl"), "wb") as f:
        pickle.dump(structures["mBIC"], f)


    print("\n[RQ3] ── 2x2 Evaluation ──────────────────────────────────────")
    results = []
    per_node_results = []


    for score_name in ["BIC", "mBIC"]:
        edges = structures[score_name]


        for cpd_name in ["Tabular", "DeepCPD"]:
            label = f"{score_name} + {cpd_name}"
            print(f"\n[RQ3] ── {label} ──")
            set_seed(CONFIG["random_seed"])
            t0 = time.time()


            if cpd_name == "Tabular":
                tab_bn = fit_tabular_bn(edges, train_int, state_names_dict)
                nll    = compute_nll_tabular(tab_bn, test_int)
                sepsis_metrics = compute_sepsis_metrics_tabular(
                    tab_bn, test_int, val_int, target_node, target_state_int
                )
                node_nlls = compute_per_node_nll_tabular(tab_bn, test_int)


            else:
                dag, models = fit_neural_bn(edges, train_int, val_int, cardinalities, PRIMARY_ARCH)
                nll   = compute_nll_neural(dag, models, test_int, cardinalities)
                sepsis_metrics = compute_sepsis_metrics_neural(
                    dag, models, test_int, val_int, cardinalities, target_node, target_state_int
                )
                node_nlls = compute_per_node_nll_neural(dag, models, test_int, cardinalities)


            fit_time = time.time() - t0
            print(f"[RQ3]   Fit time     : {fit_time:.1f}s")
            print(f"[RQ3]   Test NLL     : {nll:.4f}")
            for metric_name, metric_val in sepsis_metrics.items():
                print(f"[RQ3]   {metric_name:<14}: {fmt(metric_val)}")


            results.append({
                "Structure":      score_name,
                "CPD":            cpd_name,
                "Arch":           PRIMARY_ARCH["label"],
                "Label":          label,
                "N_edges":        len(edges),
                "SHD_vs_BIC":     0 if score_name == "BIC" else shd,
                "Test_NLL":       round(nll, 4),
                "Sepsis_NLL":     sepsis_metrics["sepsis_nll"],
                "PR_AUC":         sepsis_metrics["pr_auc"],
                "Precision":      sepsis_metrics["precision"],
                "Recall":         sepsis_metrics["recall"],
                "F1":             sepsis_metrics["f1"],
                "Fit_time_sec":   round(fit_time, 1),
            })


            for node, node_nll in node_nlls.items():
                n_parents = len(edges) if cpd_name == "Tabular" else len(list(dag.get_parents(node)))
                per_node_results.append({
                    "Structure":    score_name,
                    "CPD":          cpd_name,
                    "Node":         node,
                    "Cardinality":  cardinalities[node],
                    "N_parents":    n_parents if cpd_name == "DeepCPD" else len(list(tab_bn.get_parents(node))),
                    "NLL":          round(node_nll, 4),
                })


    res_df = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("RQ3 RESULTS TABLE")
    print("=" * 65)
    print(res_df.to_string(index=False))
    res_df.to_csv(os.path.join(output_dir, "rq3_results.csv"), index=False)


    per_node_df = pd.DataFrame(per_node_results)
    per_node_df.to_csv(os.path.join(output_dir, "rq3_per_node_nll.csv"), index=False)


    print("\n[RQ3] ── Per-Node NLL Breakdown by Cardinality ──────────────")
    for score_name in ["BIC", "mBIC"]:
        print(f"\n[{score_name}]")
        subset = per_node_df[per_node_df["Structure"] == score_name]
        for cpd_name in ["Tabular", "DeepCPD"]:
            print(f"  {cpd_name}:")
            cpd_subset = subset[subset["CPD"] == cpd_name]
            for card in sorted(cpd_subset["Cardinality"].unique()):
                card_data = cpd_subset[cpd_subset["Cardinality"] == card]
                mean_nll = card_data["NLL"].mean()
                std_nll  = card_data["NLL"].std()
                print(f"    Card={card}: {mean_nll:.4f} ± {std_nll:.4f}")


    print("\n[RQ3] ── Per-Node NLL Breakdown by In-degree ─────────────────")
    for score_name in ["BIC", "mBIC"]:
        print(f"\n[{score_name}]")
        subset = per_node_df[per_node_df["Structure"] == score_name]
        for cpd_name in ["Tabular", "DeepCPD"]:
            print(f"  {cpd_name}:")
            cpd_subset = subset[subset["CPD"] == cpd_name]
            for n_parents in sorted(cpd_subset["N_parents"].unique()):
                parent_data = cpd_subset[cpd_subset["N_parents"] == n_parents]
                mean_nll = parent_data["NLL"].mean()
                std_nll  = parent_data["NLL"].std()
                print(f"    Parents={n_parents}: {mean_nll:.4f} ± {std_nll:.4f}")


    print("\n" + "=" * 65)
    print("RQ3 ARCHITECTURE SWEEP")
    print("=" * 65)


    sweep_records = []
    for score_name in ["BIC", "mBIC"]:
        edges = structures[score_name]
        for arch in ARCH_SWEEP:
            label = f"{score_name} + DeepCPD({arch['label']})"
            print(f"\n[sweep] {label}")
            set_seed(CONFIG["random_seed"])
            t0 = time.time()


            dag, models = fit_neural_bn(edges, train_int, val_int, cardinalities, arch)
            nll   = compute_nll_neural(dag, models, test_int, cardinalities)
            sepsis_metrics = compute_sepsis_metrics_neural(
                dag, models, test_int, val_int, cardinalities, target_node, target_state_int
            )


            fit_time = time.time() - t0
            print(f"[sweep]   NLL={nll:.4f}  "
                  f"Sepsis_NLL={fmt(sepsis_metrics['sepsis_nll'])}  "
                  f"PR-AUC={fmt(sepsis_metrics['pr_auc'])}  "
                  f"time={fit_time:.1f}s")


            sweep_records.append({
                "Structure":      score_name,
                "Arch":           arch["label"],
                "Layers":         str(arch["layers"]),
                "Label":          label,
                "N_edges":        len(edges),
                "SHD_vs_BIC":     0 if score_name == "BIC" else shd,
                "Test_NLL":       round(nll, 4),
                "Sepsis_NLL":     sepsis_metrics["sepsis_nll"],
                "PR_AUC":         sepsis_metrics["pr_auc"],
                "Precision":      sepsis_metrics["precision"],
                "Recall":         sepsis_metrics["recall"],
                "F1":             sepsis_metrics["f1"],
                "Fit_time_sec":   round(fit_time, 1),
            })


    sweep_df = pd.DataFrame(sweep_records)
    print("\n" + "=" * 65)
    print("SWEEP RESULTS TABLE")
    print("=" * 65)
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(os.path.join(output_dir, "rq3_arch_sweep.csv"), index=False)


    json.dump(
        {
            "structures":      {k: [list(e) for e in v] for k, v in structures.items()},
            "shd_bic_vs_mbic": shd,
            "cardinalities":   cardinalities,
            "split_sizes":     {"train": len(train_str),
                                "val":   len(val_str),
                                "test":  len(test_str)},
            "config":          CONFIG,
        },
        open(os.path.join(output_dir, "rq3_summary.json"), "w"), indent=2
    )


    print(f"\n[RQ3] Results saved to {output_dir}/")
    print("=" * 65)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="RQ3_RESULTS")
    args = parser.parse_args()
    run_rq3(args.output_dir)
