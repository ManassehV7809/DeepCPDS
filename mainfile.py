# ============================
# Imports
# ============================
import os
import math
import time
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm

# Torch
import torch
import torch.nn as nn
import torch.optim as optim

# SciPy / Sklearn
from scipy.stats import entropy
from sklearn.model_selection import KFold

# pgmpy
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except Exception:
    from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator

# BN Generator
import bayesian_network_generator as bng


# ============================
# Device
# ============================
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# ============================
# Reproducibility
# ============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================
# Config
# ============================
EXPERIMENT_CONFIG = {
    # RQ1: architecture sweep
    "rq1": {
        "topologies": ["dag", "tree", "polytree"],   # generator topologies
        "num_nodes": 8,                               # keep modest for runtime; adjust as needed
        "node_cardinality": 2,                        # your learner assumes binary
        "dataset_sizes": [100, 500, 2000],            # scale as needed
        "epochs": 80,
        "patience": 10,
        "repeats": 3,                                 # per size/topology repeat count
        "kfold_max": 5,
        "arch_depths": [2, 4, 6],                     # your sweep; widths drawn per-layer
        "width_choices": [32, 64, 128],
    },
    # RQ2: unlabeled mixture robustness
    "rq2": {
        "topology": "dag",                            # shared structure across components
        "num_nodes": 10,
        "node_cardinality": 2,
        "components": 3,                              # number of mixture components
        "weights": [0.4, 0.4, 0.2],                   # sums to 1
        "dataset_size_total": 3000,                   # total pooled dataset size
        "epochs": 80,
        "patience": 10,
        "arch_depths": [4],                           # pick the best from RQ1 later; start with 4
        "width_choices": [64],                        # fix width for speed; swap after RQ1
        "kfold_max": 5,
        "repeats": 3,
    },
    # IO
    "output_dir": "./bn_results",
}

os.makedirs(EXPERIMENT_CONFIG["output_dir"], exist_ok=True)

# ============================
# Helper: KFold choices
# ============================
def get_k_folds(dataset_size, max_k=5):
    if dataset_size < 10:
        return 2
    elif dataset_size < 50:
        return min(5, dataset_size // 10)
    else:
        return max_k

def get_dynamic_batch_size(training_size):
    if training_size <= 10:
        return 1
    elif training_size <= 20:
        return 2
    elif training_size <= 50:
        return 4
    elif training_size <= 100:
        return 5
    elif training_size <= 500:
        return 32
    else:
        return 128

# ============================
# Neural CPD model
# ============================
class CPDNetwork(nn.Module):
    def __init__(self, input_size, output_size, layers_config):
        super(CPDNetwork, self).__init__()
        layers = []
        in_features = input_size
        for units in layers_config:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.LayerNorm(units))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_features = units
        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class NeuralBayesianNetwork(BayesianNetwork):
    def __init__(self, ebunch=None):
        super(NeuralBayesianNetwork, self).__init__(ebunch)
        self.models = {}
        self.architecture_log = []

    def fit(self, data, epochs=60, batch_size=5, patience=10, fixed_architecture=None, lr=1e-3, weight_decay=1e-4):
        # Learn a neural CPD per node
        for node in self.nodes():
            parents = list(self.get_parents(node))
            self._add_cpd_with_nn(
                node,
                parents,
                data,
                epochs,
                batch_size,
                patience,
                fixed_architecture,
                lr,
                weight_decay,
            )
        self.check_model()

    def _add_cpd_with_nn(
        self,
        variable,
        evidence,
        data,
        epochs,
        batch_size,
        patience,
        fixed_architecture,
        lr,
        weight_decay,
    ):
        # Binary-only pipeline; assume data columns are ints {0,1}
        if evidence:
            X = data[evidence].values.astype("float32")
        else:
            X = np.zeros((data.shape[0], 1), dtype="float32")

        y = data[variable].values.astype("int64")
        input_size = X.shape[1] if evidence else 1
        output_size = 2  # binary target

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)

        assert torch.all((y_tensor == 0) | (y_tensor == 1)), f"Non-binary target in {variable}"

        # train
        model, architecture_details = self._train_nn(
            X_tensor,
            y_tensor,
            input_size,
            output_size,
            epochs,
            batch_size,
            patience,
            fixed_architecture,
            lr,
            weight_decay,
            variable,
        )
        self.models[variable] = model
        self.architecture_log.append((variable, architecture_details))

        # Build CPD table by enumerating parent states
        if evidence:
            ev_vals = [list(range(2)) for _ in evidence]  # binary parents
            ev_combos = np.array(list(itertools.product(*ev_vals)), dtype="float32")
        else:
            ev_combos = np.array([[0]], dtype="float32")

        evidence_tensor = torch.tensor(ev_combos, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = model(evidence_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_cpd = probabilities.detach().cpu().numpy().T  # shape: (2, #columns)

        if evidence:
            cpd = TabularCPD(
                variable=variable,
                variable_card=2,
                values=predicted_cpd,
                evidence=evidence,
                evidence_card=[2] * len(evidence),
                state_names={variable: [0, 1], **{e: [0, 1] for e in evidence}},
            )
        else:
            cpd = TabularCPD(
                variable=variable,
                variable_card=2,
                values=predicted_cpd,
                state_names={variable: [0, 1]},
            )
        self.add_cpds(cpd)

    def _train_nn(
        self,
        X_train,
        y_train,
        input_size,
        output_size,
        epochs,
        batch_size,
        patience,
        layers_config,
        lr,
        weight_decay,
        variable=None,
    ):
        model = CPDNetwork(input_size, output_size, layers_config).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float("inf")
        best_model_state = None
        early_stop_counter = 0

        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            model.train()
            total = 0.0
            for xb, yb in dataloader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * xb.size(0)
            avg = total / len(dataset)
            if avg < best_loss:
                best_loss = avg
                best_model_state = model.state_dict()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if early_stop_counter >= patience:
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        details = {"layers_config": layers_config, "best_loss": float(best_loss)}
        return model, details

    def sample(self, n):
        sampler = BayesianModelSampling(self)
        return sampler.forward_sample(size=n)

# ============================
# Generator integration
# ============================
def generate_from_bng(
    num_nodes=6,
    node_cardinality=2,
    sample_size=1000,
    topology_type="dag",
    quality_assessment=True,
    seed=None,
):
    # Create generator instance
    generator = bng.NetworkGenerator(random_state=seed)
    params = {
        "num_nodes": num_nodes,
        "node_cardinality": node_cardinality,
        "sample_size": sample_size,
        "topology_type": topology_type,
        "quality_assessment": quality_assessment,
    }
    result = generator.generate_network(**params)
    model = result["model"]
    samples = result["samples"]
    runtime = result["runtime"]
    metrics = result.get("quality_metrics", None)
    # Force integer dtype for consistency
    samples = samples.astype(int)
    return model, samples, runtime, metrics

# ============================
# Mixture creation (RQ2)
# ============================
def average_tabular_cpds(cpds_list, weights):
    # Assumes identical ordering and shapes, binary only
    values = None
    for cpd, w in zip(cpds_list, weights):
        v = cpd.values
        values = v * w if values is None else values + v * w
    return values

def make_mixture_bn(models, weights):
    # Create a BN with same structure; CPDs weighted-average across components
    base = models[0]
    edges = list(base.edges())
    mix_bn = BayesianNetwork(edges)
    for node in base.nodes():
        comp_cpds = [m.get_cpds(node) for m in models]
        # use first CPD meta
        ref = comp_cpds[0]
        values = average_tabular_cpds(comp_cpds, weights)
        if ref.get_evidence():
            cpd = TabularCPD(
                variable=ref.variable,
                variable_card=ref.variable_card,
                values=values,
                evidence=ref.get_evidence(),
                evidence_card=ref.cardinality[1:].tolist(),
                state_names=ref.state_names,
            )
        else:
            cpd = TabularCPD(
                variable=ref.variable,
                variable_card=ref.variable_card,
                values=values,
                state_names=ref.state_names,
            )
        mix_bn.add_cpds(cpd)
    mix_bn.check_model()
    return mix_bn

def sample_mixture(models, weights, total_size, seed=None):
    # Draw floor(weights*total), fill remainder to reach total_size
    if seed is not None:
        set_seed(seed)
    sizes = [int(w * total_size) for w in weights]
    short = total_size - sum(sizes)
    for i in range(short):
        sizes[i % len(sizes)] += 1
    frames = []
    for m, n in zip(models, sizes):
        samp = BayesianModelSampling(m).forward_sample(size=n)
        frames.append(samp.astype(int))
    data = pd.concat(frames, axis=0, ignore_index=True)
    return data

# ============================
# Evaluation
# ============================
def smooth_distribution(distribution, epsilon=1e-8):
    arr = np.clip(np.asarray(distribution, dtype=float) + epsilon, a_min=epsilon, a_max=None)
    return arr / arr.sum()

def node_probs_from_tabular(cpd, evidence_tuples):
    # Reduce CPD to the evidence column and return row vector
    red = cpd.reduce(evidence_tuples, inplace=False)
    return red.values.flatten()

def node_probs_from_neural(model_nn, parents, evidence_config):
    if parents:
        X = torch.tensor(evidence_config.astype("float32")).unsqueeze(0).to(device)
    else:
        X = torch.tensor([[0.0]], dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model_nn(X)
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
    return probs

def enumerate_evidence_configs(parent_names, df, assume_binary=True):
    if not parent_names:
        return np.array([[0]], dtype=int)
    if assume_binary:
        return np.array(list(itertools.product([0, 1], repeat=len(parent_names))), dtype=int)
    # Fallback: unique combos from data (not used here)
    uniq_lists = [sorted(df[p].unique().tolist()) for p in parent_names]
    return np.array(list(itertools.product(*uniq_lists)), dtype=int)

def per_node_kl_and_nll(original_bn, neural_bn, traditional_bn, test_df, node, assume_binary=True):
    parents = list(original_bn.get_parents(node))
    ev_combos = enumerate_evidence_configs(parents, test_df, assume_binary=assume_binary)
    # Average KL across parent configs
    kl_nn_total, kl_tab_total = 0.0, 0.0
    for ev in ev_combos:
        ev_tuples = list(zip(parents, ev)) if parents else []
        p_true = smooth_distribution(node_probs_from_tabular(original_bn.get_cpds(node), ev_tuples))
        p_nn = smooth_distribution(node_probs_from_neural(neural_bn.models[node], parents, ev))
        p_tab = smooth_distribution(node_probs_from_tabular(traditional_bn.get_cpds(node), ev_tuples))
        kl_nn_total += entropy(p_true, p_nn)
        kl_tab_total += entropy(p_true, p_tab)
    avg_kl_nn = kl_nn_total / len(ev_combos)
    avg_kl_tab = kl_tab_total / len(ev_combos)
    # Empirical NLL on test rows for this node
    nll_nn, nll_tab, count = 0.0, 0.0, 0
    for _, row in test_df.iterrows():
        y = int(row[node])
        ev = row[parents].values.astype(int) if parents else np.array([0], dtype=int)
        ev_tuples = list(zip(parents, ev)) if parents else []
        p_nn = node_probs_from_neural(neural_bn.models[node], parents, ev)[y]
        p_tab = node_probs_from_tabular(traditional_bn.get_cpds(node), ev_tuples)[y]
        p_nn = max(p_nn, 1e-12)
        p_tab = max(p_tab, 1e-12)
        nll_nn += -math.log(p_nn)
        nll_tab += -math.log(p_tab)
        count += 1
    return {
        "kl_nn": float(avg_kl_nn),
        "kl_tab": float(avg_kl_tab),
        "nll_nn": float(nll_nn / max(count, 1)),
        "nll_tab": float(nll_tab / max(count, 1)),
    }

# ============================
# RQ1: Architecture sweeps
# ============================
def run_rq1_arch_sweep(cfg):
    out_rows = []
    for topo in cfg["topologies"]:
        for size in cfg["dataset_sizes"]:
            for rep in range(cfg["repeats"]):
                # Generate GT BN + data
                model_gt, samples, gen_time, q_metrics = generate_from_bng(
                    num_nodes=cfg["num_nodes"],
                    node_cardinality=cfg["node_cardinality"],
                    sample_size=size,
                    topology_type=topo,
                    quality_assessment=True,
                    seed=rep + 1234,
                )
                edges = list(model_gt.edges())

                # K-fold
                kfold = get_k_folds(len(samples), max_k=cfg["kfold_max"])
                kf = KFold(n_splits=kfold, shuffle=True, random_state=rep)

                for depth in cfg["arch_depths"]:
                    # Random widths for each layer
                    layers_config = [int(np.random.choice(cfg["width_choices"])) for _ in range(depth)]
                    fold_idx = 0
                    for tr_idx, te_idx in kf.split(samples):
                        fold_idx += 1
                        train_df = samples.iloc[tr_idx].copy()
                        test_df = samples.iloc[te_idx].copy()

                        # Train Neural BN
                        neural_bn = NeuralBayesianNetwork(edges)
                        batch_size = get_dynamic_batch_size(len(train_df))
                        t0 = time.time()
                        neural_bn.fit(
                            train_df,
                            epochs=cfg["epochs"],
                            batch_size=batch_size,
                            patience=cfg["patience"],
                            fixed_architecture=layers_config,
                        )
                        neural_time = time.time() - t0

                        # Train Tabular (pooled ML)
                        tab_bn = BayesianNetwork(edges)
                        tab_bn.fit(
                            train_df,
                            estimator=MaximumLikelihoodEstimator,
                            state_names={col: [0, 1] for col in train_df.columns},
                        )

                        # Evaluate per node
                        for node in tab_bn.nodes():
                            scores = per_node_kl_and_nll(model_gt, neural_bn, tab_bn, test_df, node)
                            out_rows.append({
                                "RQ": "RQ1",
                                "topology": topo,
                                "num_nodes": cfg["num_nodes"],
                                "dataset_size": size,
                                "repeat": rep + 1,
                                "fold": fold_idx,
                                "node": node,
                                "arch_depth": depth,
                                "layers_config": str(layers_config),
                                "neural_time_sec": neural_time,
                                "kl_nn": scores["kl_nn"],
                                "kl_tab": scores["kl_tab"],
                                "nll_nn": scores["nll_nn"],
                                "nll_tab": scores["nll_tab"],
                                "gen_time_sec": gen_time,
                            })
    df = pd.DataFrame(out_rows)
    out_path = os.path.join(EXPERIMENT_CONFIG["output_dir"], "rq1_arch_sweep.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved RQ1 results to {out_path}")
    return df

# ============================
# RQ2: Unlabeled mixture robustness
# ============================
def run_rq2_mixture(cfg):
    out_rows = []
    assert abs(sum(cfg["weights"]) - 1.0) < 1e-6, "Mixture weights must sum to 1"

    for rep in range(cfg["repeats"]):
        # Build K identical-structure BNs with different CPDs
        models = []
        edges = None
        for k in range(cfg["components"]):
            model_k, _, _, _ = generate_from_bng(
                num_nodes=cfg["num_nodes"],
                node_cardinality=cfg["node_cardinality"],
                sample_size=10,                      # tiny sample just to create model; we won't use these rows
                topology_type=cfg["topology"],
                quality_assessment=False,
                seed=1000 + rep*17 + k,
            )
            if edges is None:
                edges = list(model_k.edges())
            models.append(model_k)

        # Mixture BN (ground truth CPDs are weighted averages)
        mix_bn = make_mixture_bn(models, cfg["weights"])

        # Pooled dataset from mixture components
        data = sample_mixture(models, cfg["weights"], cfg["dataset_size_total"], seed=777 + rep)

        # K-fold on pooled data
        kfold = get_k_folds(len(data), max_k=cfg["kfold_max"])
        kf = KFold(n_splits=kfold, shuffle=True, random_state=rep)

        for depth in cfg["arch_depths"]:
            layers_config = [int(np.random.choice(cfg["width_choices"])) for _ in range(depth)]
            fold_idx = 0
            for tr_idx, te_idx in kf.split(data):
                fold_idx += 1
                train_df = data.iloc[tr_idx].copy()
                test_df = data.iloc[te_idx].copy()

                # Neural CPDs (single-backbone, pooled)
                neural_bn = NeuralBayesianNetwork(edges)
                batch_size = get_dynamic_batch_size(len(train_df))
                t0 = time.time()
                neural_bn.fit(
                    train_df,
                    epochs=cfg["epochs"],
                    batch_size=batch_size,
                    patience=cfg["patience"],
                    fixed_architecture=layers_config,
                )
                neural_time = time.time() - t0

                # Pooled tabular baseline
                tab_bn = BayesianNetwork(edges)
                tab_bn.fit(
                    train_df,
                    estimator=MaximumLikelihoodEstimator,
                    state_names={col: [0, 1] for col in train_df.columns},
                )

                # Evaluate against mixture ground-truth CPDs
                for node in tab_bn.nodes():
                    scores = per_node_kl_and_nll(mix_bn, neural_bn, tab_bn, test_df, node)
                    out_rows.append({
                        "RQ": "RQ2",
                        "topology": cfg["topology"],
                        "num_nodes": cfg["num_nodes"],
                        "dataset_size_total": cfg["dataset_size_total"],
                        "components": cfg["components"],
                        "weights": str(cfg["weights"]),
                        "repeat": rep + 1,
                        "fold": fold_idx,
                        "node": node,
                        "arch_depth": depth,
                        "layers_config": str(layers_config),
                        "neural_time_sec": neural_time,
                        "kl_nn": scores["kl_nn"],
                        "kl_tab": scores["kl_tab"],
                        "nll_nn": scores["nll_nn"],
                        "nll_tab": scores["nll_tab"],
                    })
    df = pd.DataFrame(out_rows)
    out_path = os.path.join(EXPERIMENT_CONFIG["output_dir"], "rq2_mixture.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved RQ2 results to {out_path}")
    return df

# ============================
# Run experiments
# ============================
if __name__ == "__main__":
    print("Running RQ1 sweeps...")
    rq1_df = run_rq1_arch_sweep(EXPERIMENT_CONFIG["rq1"])

    print("Running RQ2 mixture...")
    rq2_df = run_rq2_mixture(EXPERIMENT_CONFIG["rq2"])

    print("Done.")
