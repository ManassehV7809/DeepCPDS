import os
import math
import time
import random
import argparse
import glob
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import entropy
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression as LR


# pgmpy
try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except Exception:
    from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
# bn generator
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
print(f"[info] using device: {device}")

# Important: don't force torch to 1 CPU thread if you request multiple CPUs via Slurm.
# You can still cap it if you want determinism, but then request fewer CPUs.
# torch.set_num_threads(1)


# ============================
# Reproducibility helper
# ============================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


# ============================
# Experiment config (defaults)
# ============================
# ============================
# Experiment config (defaults)
# ============================
DEFAULT_CONFIG = {
    "rq1": {
        # Topology / structure
        "topologies": ["dag"],
        "num_nodes": 30,

        # Cardinality:
        #   Option A: sweep over a list via "node_cardinality_list"
        #   Option B: single fixed cardinality via "node_cardinality"
        #
        # If node_cardinality_list is present, run_single_task will use it.
        # Otherwise, it will fall back to node_cardinality.
        "node_cardinality_list": [2, 4, 6],
        # "node_cardinality": 6,  # <-- keep as fallback if you like

        # Dataset sizes (you can trim/extend as needed)
        "dataset_sizes": [250, 500, 750, 1000, 1250, 1500],

        # Training hyperparameters for neural CPDs
        "epochs": 25,
        "patience": 10,
        "repeats": 3,

        # Number of distinct graph structures per setting
        "num_structures": 5,
        "base_structure_seed": 1234,

        # Cross-validation folds
        "kfold_max": 5,

        # Neural architecture sweep
        "arch_depths": [4, 5, 6],
        "activations": ["relu", "gelu"],
        "optimizers": ["adamw", "sgd"],
        "lr": 1e-3,
        "weight_decay": 1e-4,

        # Complexity: vary max indegree (this is where “complexity increases” lives)
        # You can start with [3, 5, 7] or just [2, 4, 7].
        "max_indegree_list": [3, 5, 7],

        "debug_mode": False,   # debug prints slow things down

        # =========================
        # Tabular baseline options
        # =========================
        # If True, use BayesianEstimator with a prior (smoothed tabular).
        # If False, use raw MaximumLikelihoodEstimator (unsmoothed MLE).
        "tabular_use_bayesian": True,

        # Prior settings for BayesianEstimator
        # "BDeu" is a standard choice; you can also try "dirichlet".
        "tabular_prior_type": "BDeu",
        # Equivalent sample size (pseudo-count strength). 1–10 is a reasonable range.
        "tabular_equiv_sample_size": 5.0,
    },
}

# ============================
# Helpers
# ============================

def fit_tabular_bn(edges, train_df, states, rq1_cfg):
    use_bayesian = bool(rq1_cfg.get("tabular_use_bayesian", False))

    tab_bn = BayesianNetwork(edges)

    if use_bayesian:
        prior_type = rq1_cfg.get("tabular_prior_type", "BDeu")
        ess = float(rq1_cfg.get("tabular_equiv_sample_size", 5.0))
        tab_bn.fit(
            train_df,
            estimator=BayesianEstimator,
            prior_type=prior_type,
            equivalent_sample_size=ess,
            state_names={col: states for col in train_df.columns},
        )
    else:
        tab_bn.fit(
            train_df,
            estimator=MaximumLikelihoodEstimator,
            state_names={col: states for col in train_df.columns},
        )

    return tab_bn

def fit_dt_bn(edges, train_df, card):
    """Fit a Decision Tree CPD per node (Context-Specific Independence baseline)."""
    dt_models = {}
    structure = BayesianNetwork(edges)
    for node in structure.nodes():
        parents = list(structure.get_parents(node))
        y = train_df[node].values.astype(int)
        if parents:
            X_int = train_df[parents].values.astype(int)
            X = one_hot_encode_parent_matrix(X_int, card)
        else:
            X = np.zeros((len(train_df), 1), dtype=np.float32)
        clf = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=5,
            random_state=42,
        )
        clf.fit(X, y)
        dt_models[node] = (clf, parents)
    return dt_models


def fit_lr_bn(edges, train_df, card):
    """Fit a Logistic Regression CPD per node (linear structured baseline)."""
    lr_models = {}
    structure = BayesianNetwork(edges)
    for node in structure.nodes():
        parents = list(structure.get_parents(node))
        y = train_df[node].values.astype(int)
        if parents:
            X_int = train_df[parents].values.astype(int)
            X = one_hot_encode_parent_matrix(X_int, card)
        else:
            X = np.zeros((len(train_df), 1), dtype=np.float32)
        clf = LR(
            max_iter=500,
            random_state=42,
            solver="lbfgs",
            multi_class="multinomial",
            C=1.0,
        )
        clf.fit(X, y)
        lr_models[node] = (clf, parents)
    return lr_models


def node_probs_from_sklearn(models_dict, node, ev_int, card):
    """Query a sklearn CPD (DT or LR) for P(node | parents=ev_int)."""
    clf, parents = models_dict[node]
    if parents:
        x = one_hot_encode_parent_vector(ev_int, card).reshape(1, -1)
    else:
        x = np.zeros((1, 1), dtype=np.float32)
    probs = clf.predict_proba(x).flatten()
    # predict_proba only returns columns for classes seen in training
    # so we need to map back to the full cardinality
    full_probs = np.full(card, 1e-8, dtype=float)
    for i, cls in enumerate(clf.classes_):
        full_probs[int(cls)] = probs[i]
    return full_probs / full_probs.sum()


def get_k_folds(dataset_size, max_k=5):
    if dataset_size < 10:
        return 2
    elif dataset_size < 50:
        return min(5, max(2, dataset_size // 10))
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
        return 8
    elif training_size <= 500:
        return 32
    else:
        return 128


def get_adaptive_dropout(training_size):
    if training_size < 50:
        return 0.0
    elif training_size < 200:
        return 0.05
    elif training_size < 500:
        return 0.1
    else:
        return 0.15


def get_adaptive_weight_decay(training_size):
    if training_size < 100:
        return 1e-5
    elif training_size < 500:
        return 5e-5
    else:
        return 1e-4


def choose_layers_config_fixed(depth, activation, optimizer):
    arch_configs = {
        4: [10,8],
        5: [12,10,8],
        6: [14,12, 10, 8]
    }
    return {
        "layers": arch_configs[depth],
        "activation": activation,
        "optimizer": optimizer,
    }


def one_hot_encode_parent_matrix(X_int: np.ndarray, card: int) -> np.ndarray:
    """
    Vectorized one-hot for (N, p) integer matrix -> (N, p*card)
    """
    if X_int.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {X_int.shape}")
    N, p = X_int.shape
    X_int = X_int.astype(np.int64)
    if p == 0:
        return np.zeros((N, 1), dtype=np.float32)

    if X_int.min() < 0 or X_int.max() >= card:
        raise ValueError(f"Parent matrix has values outside 0..{card-1}")

    # Build indices for flat one-hot
    rows = np.repeat(np.arange(N), p)
    cols = (np.tile(np.arange(p), N) * card) + X_int.reshape(-1)
    out = np.zeros((N, p * card), dtype=np.float32)
    out[rows, cols] = 1.0
    return out


def one_hot_encode_parent_vector(ev_int: np.ndarray, card: int) -> np.ndarray:
    ev_int = np.asarray(ev_int, dtype=np.int64).reshape(-1)
    p = ev_int.shape[0]
    if p == 0:
        return np.zeros((1,), dtype=np.float32)

    if ev_int.min() < 0 or ev_int.max() >= card:
        raise ValueError(f"Evidence vector has values outside 0..{card-1}")

    out = np.zeros((p * card,), dtype=np.float32)
    cols = np.arange(p) * card + ev_int
    out[cols] = 1.0
    return out


def smooth_distribution(distribution, epsilon=1e-8):
    arr = np.asarray(distribution, dtype=float)
    arr = np.clip(arr, a_min=epsilon, a_max=None)
    return arr / arr.sum()


def node_probs_from_tabular(cpd, evidence_tuples):
    red = cpd.reduce(evidence_tuples, inplace=False)
    return red.values.flatten()


# ============================
# Network class
# ============================
class CPDNetwork(nn.Module):
    def __init__(self, input_size, output_size, layer_units, activation="relu", dropout=0.1):
        super().__init__()
        layers = []
        in_features = input_size
        for units in layer_units:
            layers.append(nn.Linear(in_features, units))
            if dropout > 0.05:
                layers.append(nn.LayerNorm(units))

            act = activation.lower()
            if act == "gelu":
                layers.append(nn.GELU())
            elif act == "leakyrelu":
                layers.append(nn.LeakyReLU())
            elif act == "tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_features = units

        layers.append(nn.Linear(in_features, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class NeuralBayesianNetwork:
    def __init__(self, edges, card: int):
        self.structure = BayesianNetwork(edges)
        self.models = {}
        self.card = card

    def nodes(self):
        return list(self.structure.nodes())

    def parents(self, node):
        return list(self.structure.get_parents(node))

    def fit(self, data, epochs=60, batch_size=32, patience=10,
            fixed_architecture=None, lr=1e-3, weight_decay=1e-4):
        for node in self.nodes():
            parents = self.parents(node)
            self._fit_node(node, parents, data, epochs, batch_size, patience,
                           fixed_architecture, lr, weight_decay)

    def _fit_node(self, variable, evidence, data, epochs, batch_size, patience,
                  fixed_architecture, lr, weight_decay):

        if evidence:
            X_int = data[evidence].values.astype(int)
            X = one_hot_encode_parent_matrix(X_int, self.card)
        else:
            X = np.zeros((len(data), 1), dtype=np.float32)

        y = data[variable].values.astype(np.int64)
        y_min, y_max = int(y.min()), int(y.max())
        if y_min < 0 or y_max >= self.card:
            raise ValueError(
                f"[label error] {variable}: y in [{y_min},{y_max}] expected [0..{self.card-1}]"
            )

        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.long, device=device)

        model = self._train_nn(
            X_tensor, y_tensor,
            input_size=X_tensor.shape[1],
            output_size=self.card,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            layers_config=fixed_architecture,
            lr=lr,
            weight_decay=weight_decay,
        )
        self.models[variable] = model

    def _train_nn(self, X_train, y_train, input_size, output_size,
                  epochs, batch_size, patience, layers_config,
                  lr, weight_decay):

        activation = "relu"
        optimizer_name = "adamw"
        if isinstance(layers_config, dict):
            layer_units = layers_config.get("layers", [])
            activation = layers_config.get("activation", "relu")
            optimizer_name = layers_config.get("optimizer", "adamw")
        else:
            layer_units = layers_config

        n = len(X_train)
        adaptive_dropout = get_adaptive_dropout(n)
        adaptive_wd = get_adaptive_weight_decay(n)

        model = CPDNetwork(
            input_size=input_size,
            output_size=output_size,
            layer_units=layer_units,
            activation=activation,
            dropout=adaptive_dropout,
        ).to(device)

        criterion = nn.CrossEntropyLoss()

        if optimizer_name.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=adaptive_wd)
        else:
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=adaptive_wd)

        # validation split + early stopping
        if n >= 20:
            val_frac = 0.2
            split = int(n * (1 - val_frac))
            perm = torch.randperm(n, device=device)
            train_idx = perm[:split]
            val_idx = perm[split:]
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            use_es = True
        elif n >= 10:
            split = n - 2
            X_tr, y_tr = X_train[:split], y_train[:split]
            X_val, y_val = X_train[split:], y_train[split:]
            use_es = True
            patience = min(patience, max(1, epochs // 2))
        else:
            X_tr, y_tr = X_train, y_train
            X_val, y_val = X_train, y_train
            use_es = False

        best_loss = float("inf")
        best_state = None
        es_counter = 0

        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr, y_tr),
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        for _ in range(epochs):
            model.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_loss = float(criterion(val_logits, y_val).item())

            if use_es:
                if val_loss < best_loss - 1e-8:
                    best_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    es_counter = 0
                else:
                    es_counter += 1
                if es_counter >= patience:
                    break
            else:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        return model


# ============================
# BNG integration
# ============================
def generate_from_bng(
    num_nodes=30,
    node_cardinality=4,
    sample_size=1000,
    topology_type="dag",
    quality_assessment=True,
    seed=None,
    max_indegree=None,
):
    set_seed(seed)
    generator = bng.NetworkGenerator()
    params = {
        "num_nodes": num_nodes,
        "node_cardinality": node_cardinality,
        "sample_size": sample_size,
        "topology_type": topology_type,
        "quality_assessment": quality_assessment,
    }
    if max_indegree is not None:
        params["max_indegree"] = max_indegree

    result = generator.generate_network(**params)
    model = result["model"]
    samples = result["samples"].astype(int)
    runtime = result.get("runtime", None)
    return model, samples, runtime


# ============================
# Evaluation
# ============================
def node_probs_from_neural(neural_bn: NeuralBayesianNetwork, node: str, parents, ev_int: np.ndarray, card: int):
    model_nn = neural_bn.models[node]
    if parents:
        x = one_hot_encode_parent_vector(ev_int, card)
        X = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    else:
        X = torch.tensor([[0.0]], dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model_nn(X)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy().flatten()
    return probs


def per_node_kl_and_nll(original_bn, neural_bn, tab_bn, dt_models, lr_models,
                         test_df, node, card: int):
    parents = list(original_bn.get_parents(node))
    N = len(test_df)

    config_counts = {}
    if parents:
        parent_mat = test_df[parents].values.astype(int)
        for row in parent_mat:
            t = tuple(row)
            config_counts[t] = config_counts.get(t, 0) + 1
    else:
        config_counts[(0,)] = N

    kl_nn = kl_tab = kl_dt = kl_lr = 0.0
    total_weight = 0.0

    for ev_tuple, cnt in config_counts.items():
        weight = cnt / N
        total_weight += weight

        if parents:
            ev = np.array(ev_tuple, dtype=int)
            ev_tuples = list(zip(parents, ev))
        else:
            ev = np.array([], dtype=int)
            ev_tuples = []

        p_true = smooth_distribution(node_probs_from_tabular(original_bn.get_cpds(node), ev_tuples))
        p_tab  = smooth_distribution(node_probs_from_tabular(tab_bn.get_cpds(node), ev_tuples))
        p_nn   = smooth_distribution(node_probs_from_neural(neural_bn, node, parents, ev, card))
        p_dt   = smooth_distribution(node_probs_from_sklearn(dt_models, node, ev, card))
        p_lr   = smooth_distribution(node_probs_from_sklearn(lr_models, node, ev, card))

        kl_nn  += weight * entropy(p_true, p_nn)
        kl_tab += weight * entropy(p_true, p_tab)
        kl_dt  += weight * entropy(p_true, p_dt)
        kl_lr  += weight * entropy(p_true, p_lr)

    if total_weight > 0 and abs(total_weight - 1.0) > 1e-6:
        kl_nn  /= total_weight
        kl_tab /= total_weight
        kl_dt  /= total_weight
        kl_lr  /= total_weight

    nll_nn = nll_tab = nll_dt = nll_lr = 0.0
    for _, row in test_df.iterrows():
        y = int(row[node])
        if parents:
            ev = row[parents].values.astype(int)
            ev_tuples = list(zip(parents, ev))
        else:
            ev = np.array([], dtype=int)
            ev_tuples = []

        p_nn_y  = max(float(node_probs_from_neural(neural_bn, node, parents, ev, card)[y]), 1e-12)
        p_tab_y = max(float(node_probs_from_tabular(tab_bn.get_cpds(node), ev_tuples)[y]), 1e-12)
        p_dt_y  = max(float(node_probs_from_sklearn(dt_models, node, ev, card)[y]), 1e-12)
        p_lr_y  = max(float(node_probs_from_sklearn(lr_models, node, ev, card)[y]), 1e-12)

        nll_nn  += -math.log(p_nn_y)
        nll_tab += -math.log(p_tab_y)
        nll_dt  += -math.log(p_dt_y)
        nll_lr  += -math.log(p_lr_y)

    return {
        "kl_nn":  float(kl_nn),
        "kl_tab": float(kl_tab),
        "kl_dt":  float(kl_dt),
        "kl_lr":  float(kl_lr),
        "nll_nn":  float(nll_nn / N),
        "nll_tab": float(nll_tab / N),
        "nll_dt":  float(nll_dt / N),
        "nll_lr":  float(nll_lr / N),
        "configs_seen": int(len(config_counts)),
    }


# ============================
# Task mapping
# ============================
def enumerate_tasks(rq1_cfg):
    """
    Creates the full list of (dataset_size, depth, activation, optimizer).
    Total tasks = len(dataset_sizes)*len(depths)*len(activations)*len(optimizers)
    """
    tasks = []
    for size in rq1_cfg["dataset_sizes"]:
        for depth in rq1_cfg["arch_depths"]:
            for act in rq1_cfg["activations"]:
                for opt in rq1_cfg["optimizers"]:
                    tasks.append((size, depth, act, opt))
    return tasks


# ============================
# Single task run (array slice)
# ============================
def run_single_task(cfg, task_id: int, output_dir: str):
    rq1 = cfg["rq1"]
    tasks = enumerate_tasks(rq1)
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"task_id {task_id} out of range 0..{len(tasks)-1}")

    size, depth, activation, optimizer_name = tasks[task_id]
    layers_config = choose_layers_config_fixed(depth, activation, optimizer_name)

    # Support both a single cardinality or a list of cardinalities
    if "node_cardinality_list" in rq1:
        card_list = [int(c) for c in rq1["node_cardinality_list"]]
    else:
        card_list = [int(rq1["node_cardinality"])]

    base_struct_seed = int(rq1.get("base_structure_seed", 1234))
    num_structures = int(rq1.get("num_structures", 1))
    structure_seeds = [base_struct_seed + s_idx * 10 for s_idx in range(num_structures)]

    out_rows = []
    start = time.time()

    for topo in rq1["topologies"]:
        for max_indegree in rq1.get("max_indegree_list", [None]):
            for card in card_list:
                states = list(range(card))

                print(
                    f"[task {task_id}] topo={topo} N={size} depth={depth} "
                    f"act={activation} opt={optimizer_name} "
                    f"max_indegree={max_indegree} card={card}"
                )

                for rep in range(int(rq1["repeats"])):
                    for structure_idx in range(num_structures):

                        struct_seed = int(structure_seeds[structure_idx])

                        model_gt, samples, gen_time = generate_from_bng(
                            num_nodes=rq1["num_nodes"],
                            node_cardinality=card,
                            sample_size=size,
                            topology_type=topo,
                            quality_assessment=True,
                            seed=struct_seed,
                            max_indegree=max_indegree,
                        )

                        edges = list(model_gt.edges())
                        kfold = get_k_folds(len(samples), max_k=rq1["kfold_max"])
                        kf = KFold(n_splits=kfold, shuffle=True, random_state=struct_seed)

                        fold_idx = 0
                        for tr_idx, te_idx in kf.split(samples):
                            fold_idx += 1
                            train_df = samples.iloc[tr_idx].copy()
                            test_df = samples.iloc[te_idx].copy()

                            train_seed = int(struct_seed + rep * 100_000 + fold_idx * 1_000 + task_id)
                            set_seed(train_seed)

                            neural_bn = NeuralBayesianNetwork(edges, card=card)
                            batch_size = get_dynamic_batch_size(len(train_df))

                            t0 = time.time()
                            neural_bn.fit(
                                train_df,
                                epochs=rq1["epochs"],
                                batch_size=batch_size,
                                patience=rq1["patience"],
                                fixed_architecture=layers_config,
                                lr=rq1.get("lr", 1e-3),
                                weight_decay=rq1.get("weight_decay", 1e-4),
                            )
                            neural_time = time.time() - t0

                            # Tabular baseline (MLE or Bayesian-smoothed depending on config)
                            tab_bn = fit_tabular_bn(edges, train_df, states, rq1)
                            dt_models = fit_dt_bn(edges, train_df, card)
                            lr_models = fit_lr_bn(edges, train_df, card)

                            for node in tab_bn.nodes():
                                scores = per_node_kl_and_nll(model_gt, neural_bn, tab_bn, dt_models, lr_models,test_df, node, card=card)
                                out_rows.append({
    "RQ":               "RQ1",
    "topology":         topo,
    "num_nodes":        rq1["num_nodes"],
    "node_cardinality": card,
    "max_indegree":     max_indegree,
    "dataset_size":     size,
    "repeat":           rep + 1,
    "structure_idx":    structure_idx + 1,
    "structure_seed":   struct_seed,
    "train_seed":       train_seed,
    "fold":             fold_idx,
    "node":             node,
    "arch_depth":       depth,
    "activation":       activation,
    "optimizer":        optimizer_name,
    "layers_config":    str(layers_config["layers"]),
    "neural_time_sec":  float(neural_time),
    "kl_nn":            scores["kl_nn"],
    "kl_tab":           scores["kl_tab"],
    "kl_dt":            scores["kl_dt"],
    "kl_lr":            scores["kl_lr"],
    "nll_nn":           scores["nll_nn"],
    "nll_tab":          scores["nll_tab"],
    "nll_dt":           scores["nll_dt"],
    "nll_lr":           scores["nll_lr"],
    "configs_seen":     scores["configs_seen"],
    "gen_time_sec":     gen_time,
    "task_id":          int(task_id),
})

    df = pd.DataFrame(out_rows)

    os.makedirs(output_dir, exist_ok=True)
    chunk_path = os.path.join(
        output_dir,
        f"rq1_chunk_task{task_id}_N{size}_D{depth}_{activation}_{optimizer_name}.csv"
    )
    df.to_csv(chunk_path, index=False)

    print(f"[task {task_id}] wrote {chunk_path} rows={len(df)} time={time.time()-start:.1f}s")
    return chunk_path

# ============================
# Merge mode
# ============================
def merge_chunks(output_dir: str):
    pattern = os.path.join(output_dir, "rq1_chunk_task*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No chunk files found at {pattern}")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    raw_path = os.path.join(output_dir, "rq1_arch_sweep.csv")
    df.to_csv(raw_path, index=False)

    per_structure_summary = (
        df.groupby(
            [
                "dataset_size",
                "structure_idx",
                "structure_seed",
                "node_cardinality",
                "arch_depth",
                "layers_config",
                "activation",
                "optimizer",
                "max_indegree",
            ]
        )
        .agg(
    mean_kl_nn=("kl_nn", "mean"),
    std_kl_nn=("kl_nn", "std"),
    mean_kl_tab=("kl_tab", "mean"),
    std_kl_tab=("kl_tab", "std"),
    mean_kl_dt=("kl_dt", "mean"),   
    std_kl_dt=("kl_dt", "std"),     
    mean_kl_lr=("kl_lr", "mean"),   
    std_kl_lr=("kl_lr", "std"),     
    mean_nll_nn=("nll_nn", "mean"),
    std_nll_nn=("nll_nn", "std"),
    mean_nll_tab=("nll_tab", "mean"), 
    std_nll_tab=("nll_tab", "std"),     
    mean_nll_dt=("nll_dt", "mean"),     
    std_nll_dt=("nll_dt", "std"),       
    mean_nll_lr=("nll_lr", "mean"),     
    std_nll_lr=("nll_lr", "std"),       
)

        .reset_index()
    )
    per_structure_path = os.path.join(output_dir, "rq1_per_structure_summary.csv")
    per_structure_summary.to_csv(per_structure_path, index=False)

    across_summary = (
    per_structure_summary.groupby(
        [
            "dataset_size",
            "node_cardinality",
            "arch_depth",
            "layers_config",
            "activation",
            "optimizer",
            "max_indegree",
        ]
    )
    .agg(
        mean_kl_nn_mean=("mean_kl_nn", "mean"),
        mean_kl_nn_std=("mean_kl_nn", "std"),
        mean_kl_tab_mean=("mean_kl_tab", "mean"),
        mean_kl_tab_std=("mean_kl_tab", "std"),
        mean_kl_dt_mean=("mean_kl_dt", "mean"),
        mean_kl_dt_std=("mean_kl_dt", "std"),
        mean_kl_lr_mean=("mean_kl_lr", "mean"),
        mean_kl_lr_std=("mean_kl_lr", "std"),
        mean_nll_nn_mean=("mean_nll_nn", "mean"),
        mean_nll_nn_std=("mean_nll_nn", "std"),
        mean_nll_tab_mean=("mean_nll_tab", "mean"),
        mean_nll_tab_std=("mean_nll_tab", "std"),
        mean_nll_dt_mean=("mean_nll_dt", "mean"),
        mean_nll_dt_std=("mean_nll_dt", "std"),
        mean_nll_lr_mean=("mean_nll_lr", "mean"),
        mean_nll_lr_std=("mean_nll_lr", "std"),
        num_structures_tested=("structure_seed", "nunique"),
    )
    .reset_index()
)


    across_summary_path = os.path.join(output_dir, "rq1_across_structures_summary.csv")
    across_summary.to_csv(across_summary_path, index=False)

    print(f"[merge] wrote {raw_path}")
    print(f"[merge] wrote {per_structure_path}")
    print(f"[merge] wrote {across_summary_path}")
    return raw_path, per_structure_path, across_summary_path

# ============================
# Main
# ============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["single", "merge"], required=True)
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="./RQ1_RESULTS")
    parser.add_argument("--use_scratch", type=int, default=0)
    args = parser.parse_args()

    # Configure output dir: optionally stage in /scratch and later copy back yourself.
    out_dir = args.output_dir
    if args.use_scratch == 1:
        user = os.environ.get("USER", "user")
        scratch = f"/scratch/{user}"
        out_dir = os.path.join(scratch, os.path.basename(args.output_dir))

    cfg = DEFAULT_CONFIG

    if args.mode == "single":
        run_single_task(cfg, task_id=args.task_id, output_dir=out_dir)
    else:
        merge_chunks(out_dir)


if __name__ == "__main__":
    main()


