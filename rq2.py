 #!/usr/bin/env python3
"""
RQ2 (True DeepCPD scoring, feasible):
- Synthetic discrete BN generation (known ground-truth DAG + CPDs)
- Candidate restriction via mutual information (MI) shortlist per node
- Hill-climbing structure learning with decomposable local scores:
    (A) Tabular BIC  : tabular local LL + tabular param-count BIC penalty
    (B) DeepCPD mBIC : neural local LL + (scaled) neural param-count BIC penalty
- Final evaluation on held-out test data:
    test NLL (primary), SHD (secondary, since ground truth exists), runtime

Run:
  python rq2.py --print_tasks
  python rq2.py --task_id 0 --outputdir RQ2_RESULTS
  python rq2.py --merge --outputdir RQ2_RESULTS
"""

import os
import time
import math
import json
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import networkx as nx
from sklearn.metrics import mutual_info_score

import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# Synthetic BN generator
# -----------------------------
def generate_random_dag(
    nodes: List[str],
    max_indegree: int,
    edge_prob: float,
    seed: int,
) -> List[Tuple[str, str]]:
    """
    Acyclic by construction: only allow edges i->j for i<j in a fixed ordering.
    """
    rng = np.random.default_rng(seed)
    edges: List[Tuple[str, str]] = []
    idx = {n: i for i, n in enumerate(nodes)}
    indeg = {n: 0 for n in nodes}

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if rng.random() < edge_prob and indeg[nodes[j]] < max_indegree:
                edges.append((nodes[i], nodes[j]))
                indeg[nodes[j]] += 1

    return edges


def sample_from_bn(
    nodes: List[str],
    edges: List[Tuple[str, str]],
    cardinality: int,
    n_samples: int,
    seed: int,
) -> pd.DataFrame:
    """
    Samples from a discrete BN defined by (nodes, edges) and randomly generated CPDs.

    Implementation: topological sampling using parent configurations + Dirichlet CPDs.
    This avoids relying on pgmpy sampling/inference and keeps state space integer-coded 0..k-1.
    """
    rng = np.random.default_rng(seed)

    # Build DAG
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    assert nx.is_directed_acyclic_graph(G)
    topo = list(nx.topological_sort(G))

    parents = {n: list(G.predecessors(n)) for n in nodes}

    # Create CPDs: for each node, for each parent configuration, sample a categorical distribution
    # Store as dict: node -> dict[parent_config_tuple] = probs[k]
    cpds: Dict[str, Dict[Tuple[int, ...], np.ndarray]] = {}
    for n in topo:
        p = parents[n]
        cpds[n] = {}
        if len(p) == 0:
            probs = rng.dirichlet(np.ones(cardinality))
            cpds[n][()] = probs
        else:
            n_cfg = cardinality ** len(p)
            # For each parent config, sample probs
            for cfg_idx in range(n_cfg):
                cfg = []
                tmp = cfg_idx
                for _ in range(len(p)):
                    cfg.append(tmp % cardinality)
                    tmp //= cardinality
                cfg_t = tuple(cfg)  # order corresponds to parents[n]
                probs = rng.dirichlet(np.ones(cardinality))
                cpds[n][cfg_t] = probs

    # Sample data
    data = np.zeros((n_samples, len(nodes)), dtype=np.int64)
    col_idx = {n: i for i, n in enumerate(nodes)}

    for r in range(n_samples):
        for n in topo:
            p = parents[n]
            if len(p) == 0:
                probs = cpds[n][()]
            else:
                cfg = tuple(int(data[r, col_idx[pa]]) for pa in p)
                probs = cpds[n][cfg]
            data[r, col_idx[n]] = rng.choice(cardinality, p=probs)

    df = pd.DataFrame(data, columns=nodes)
    return df


# -----------------------------
# MI candidate restriction
# -----------------------------
def compute_mi_candidates(
    df: pd.DataFrame,
    top_k: int,
) -> Dict[str, List[str]]:
    """
    For each node X, compute MI(X;Y) for all Y!=X and keep top_k as candidate parents of X.
    """
    cols = list(df.columns)
    candidates: Dict[str, List[str]] = {}

    # Precompute MI matrix (symmetric, but we compute row-wise)
    for x in cols:
        scores = []
        xvals = df[x].values
        for y in cols:
            if y == x:
                continue
            yvals = df[y].values
            mi = mutual_info_score(xvals, yvals)
            scores.append((mi, y))
        scores.sort(reverse=True, key=lambda t: t[0])
        candidates[x] = [y for _, y in scores[:top_k]]

    return candidates


# -----------------------------
# Local scorers
# -----------------------------
@dataclass
class DeepCPDArch:
    hidden: Tuple[int, ...] = (32, 32)
    dropout: float = 0.10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 25
    patience: int = 6
    batch_size: int = 256


class CPDNet(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        d = input_dim
        for h in hidden:
            layers.append(nn.Linear(d, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = h
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits


def onehot_parents_matrix(df: pd.DataFrame, parents: List[str], cardinality: int) -> np.ndarray:
    """
    Vectorized one-hot for discrete parents:
      input: (N, |parents|) integers in [0..k-1]
      output: (N, |parents|*k) float32
    If no parents: return (N, 1) zeros.
    """
    N = len(df)
    p = len(parents)
    if p == 0:
        return np.zeros((N, 1), dtype=np.float32)

    Xint = df[parents].values.astype(np.int64)
    if Xint.min() < 0 or Xint.max() >= cardinality:
        raise ValueError("Parent values out of range for one-hot encoding.")
    out = np.zeros((N, p * cardinality), dtype=np.float32)

    # Fill blocks per parent
    for j in range(p):
        idx = Xint[:, j]
        out[np.arange(N), j * cardinality + idx] = 1.0
    return out


def count_tabular_params(child_card: int, parent_cards: List[int]) -> int:
    # (r-1)*q where q=prod(parent_cards), r=child_card
    if len(parent_cards) == 0:
        return child_card - 1
    q = int(np.prod(parent_cards))
    return (child_card - 1) * q


def count_neural_params(input_dim: int, out_dim: int, hidden: Tuple[int, ...]) -> int:
    # Count trainable params for MLP: linear layers only (bias included).
    dims = [input_dim] + list(hidden) + [out_dim]
    total = 0
    for a, b in zip(dims[:-1], dims[1:]):
        total += a * b + b
    return total


class LocalScoreCache:
    def __init__(self):
        self._cache: Dict[Tuple[str, Tuple[str, ...]], float] = {}

    def get(self, node: str, parents: List[str]) -> Optional[float]:
        return self._cache.get((node, tuple(parents)))

    def set(self, node: str, parents: List[str], score: float):
        self._cache[(node, tuple(parents))] = score


class TabularBICScorer:
    """
    Local score: LL(child | parents) - 0.5 * k * log(N)
    where k is tabular parameter count.
    """
    def __init__(self, df: pd.DataFrame, cardinality: int):
        self.df = df
        self.N = len(df)
        self.k = cardinality
        self.cache = LocalScoreCache()

    def local_score(self, child: str, parents: List[str]) -> float:
        parents = sorted(parents)
        cached = self.cache.get(child, parents)
        if cached is not None:
            return cached

        # Compute local LL via grouped counts on observed configurations
        if len(parents) == 0:
            counts = np.bincount(self.df[child].values, minlength=self.k).astype(np.float64)
            probs = (counts + 1e-12) / (counts.sum() + 1e-12 * self.k)
            ll = float(np.sum(counts * np.log(probs)))
            k_params = count_tabular_params(self.k, [])
        else:
            grp = self.df.groupby(parents + [child], observed=True).size().reset_index(name="cnt")
            # total counts per parent config
            tot = grp.groupby(parents, observed=True)["cnt"].transform("sum").values.astype(np.float64)
            cnt = grp["cnt"].values.astype(np.float64)
            ll = float(np.sum(cnt * (np.log(cnt + 1e-12) - np.log(tot + 1e-12))))

            k_params = count_tabular_params(self.k, [self.k] * len(parents))

        bic = ll - 0.5 * k_params * math.log(self.N)
        self.cache.set(child, parents, bic)
        return bic


class DeepCPDmBICScorer:
    """
    Local score: neural_LL(child|parents) - 0.5 * k_eff * log(N)
    where k_eff = (num_neural_params / alpha) is a scaled parameter count.

    This keeps the "modified BIC for neural CPDs" idea from your proposal, but avoids the
    earlier mismatch by using neural LL (true DeepCPD scoring).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        cardinality: int,
        arch: DeepCPDArch,
        alpha: float,
        device: str,
        seed: int,
        debug: bool = False,
    ):
        self.df = df
        self.N = len(df)
        self.k = cardinality
        self.arch = arch
        self.alpha = float(alpha)
        self.device = device
        self.seed = seed
        self.debug = debug
        self.cache = LocalScoreCache()

    def _train_and_ll(self, child: str, parents: List[str]) -> float:
        # Root node: categorical MLE LL
        if len(parents) == 0:
            counts = np.bincount(self.df[child].values, minlength=self.k).astype(np.float64)
            probs = (counts + 1e-12) / (counts.sum() + 1e-12 * self.k)
            return float(np.sum(counts * np.log(probs)))

        X = onehot_parents_matrix(self.df, parents, self.k)
        y = self.df[child].values.astype(np.int64)

        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device)

        # Train/val split for early stopping
        n = len(y)
        rng = np.random.default_rng(self.seed + hash((child, tuple(parents))) % 10_000_000)
        perm = rng.permutation(n)
        split = int(0.8 * n)
        tr_idx = perm[:split]
        va_idx = perm[split:] if split < n else perm[:0]

        Xtr, ytr = X_t[tr_idx], y_t[tr_idx]
        Xva, yva = X_t[va_idx], y_t[va_idx]

        model = CPDNet(
            input_dim=X_t.shape[1],
            out_dim=self.k,
            hidden=self.arch.hidden,
            dropout=self.arch.dropout,
        ).to(self.device)

        opt = optim.AdamW(model.parameters(), lr=self.arch.lr, weight_decay=self.arch.weight_decay)
        crit = nn.CrossEntropyLoss()

        # Dataloader
        bs = min(self.arch.batch_size, len(tr_idx)) if len(tr_idx) > 0 else len(tr_idx)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, ytr),
            batch_size=max(1, bs),
            shuffle=True,
        )

        best = float("inf")
        best_state = None
        bad = 0

        for epoch in range(self.arch.max_epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            # validation
            if len(va_idx) > 0:
                model.eval()
                with torch.no_grad():
                    v_logits = model(Xva)
                    v_loss = float(crit(v_logits, yva).item())
                if v_loss < best - 1e-6:
                    best = v_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= self.arch.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # In-sample LL on training data (BIC-style)
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
            logp = torch.log_softmax(logits, dim=1)
            ll = float(logp[torch.arange(n, device=self.device), y_t].sum().item())
        return ll

    def local_score(self, child: str, parents: List[str]) -> float:
        parents = sorted(parents)
        cached = self.cache.get(child, parents)
        if cached is not None:
            return cached

        ll = self._train_and_ll(child, parents)

        # k_eff
        if len(parents) == 0:
            # match tabular root dof
            k_params = self.k - 1
        else:
            input_dim = len(parents) * self.k
            k_raw = count_neural_params(input_dim=input_dim, out_dim=self.k, hidden=self.arch.hidden)
            alphaEff = self.alpha * (1.0 + len(parents)) 	
            k_params = k_raw /alphaEff

        score = ll - 0.5 * k_params * math.log(self.N)

        if self.debug and len(parents) == 1:
            print(f"[debug] node={child} parents={parents} ll={ll:.2f} k_eff={k_params:.2f} score={score:.2f}")

        self.cache.set(child, parents, score)
        return score


# -----------------------------
# Hill-climbing with MI restriction
# -----------------------------
def would_create_cycle(edges: List[Tuple[str, str]], u: str, v: str) -> bool:
    G = nx.DiGraph()
    G.add_edges_from(edges)
    G.add_edge(u, v)
    return not nx.is_directed_acyclic_graph(G)


def hill_climb(
    nodes: List[str],
    scorer,
    candidates: Dict[str, List[str]],
    max_indegree: int,
    max_iters: int,
    debug: bool = False,
) -> List[Tuple[str, str]]:
    """
    Greedy hill-climb:
      moves: add, remove, reverse
    Candidate restriction: allowed add/reverse-to-parent must be in candidates[child].
    """
    edges: List[Tuple[str, str]] = []
    parents: Dict[str, List[str]] = {n: [] for n in nodes}

    def local(child: str) -> float:
        return scorer.local_score(child, parents[child])

    # Precompute current locals (cheap due to caching anyway)
    for it in range(max_iters):
        best_delta = 0.0
        best_op = None  # (op, u, v) meaning u->v affected

        # ADD moves (restricted)
        for v in nodes:
            if len(parents[v]) >= max_indegree:
                continue
            cur = local(v)
            for u in candidates[v]:
                if u == v:
                    continue
                if u in parents[v]:
                    continue
                if would_create_cycle(edges, u, v):
                    continue
                new_par = parents[v] + [u]
                delta = scorer.local_score(v, new_par) - cur
                if delta > best_delta:
                    best_delta = delta
                    best_op = ("add", u, v)

        # REMOVE moves (not restricted)
        for v in nodes:
            if len(parents[v]) == 0:
                continue
            cur = local(v)
            for u in list(parents[v]):
                new_par = [p for p in parents[v] if p != u]
                delta = scorer.local_score(v, new_par) - cur
                if delta > best_delta:
                    best_delta = delta
                    best_op = ("remove", u, v)

        # REVERSE moves (restricted by candidates of new child)
        # reverse u->v to v->u
        for (u, v) in list(edges):
            # removing u->v reduces indegree(v), adding v->u increases indegree(u)
            if len(parents[u]) >= max_indegree:
                continue
            # The reversed edge would be v->u, so v must be a candidate parent of u
            if v not in candidates[u]:
                continue

            # Build proposed edges list
            proposed = [(a, b) for (a, b) in edges if not (a == u and b == v)]
            if would_create_cycle(proposed, v, u):
                continue

            # delta = (score_u(new_parents_u) + score_v(new_parents_v)) - (old_u + old_v)
            old_u = local(u)
            old_v = local(v)

            new_par_u = parents[u] + [v]
            new_par_v = [p for p in parents[v] if p != u]

            delta = (scorer.local_score(u, new_par_u) + scorer.local_score(v, new_par_v)) - (old_u + old_v)
            if delta > best_delta:
                best_delta = delta
                best_op = ("reverse", u, v)

        if best_op is None or best_delta <= 1e-12:
            if debug:
                print(f"[iter {it}] best_delta=0.000000 best_op=None")
            break

        op, u, v = best_op
        if debug:
            print(f"[iter {it}] best_delta={best_delta:.6f} best_op={best_op}")

        if op == "add":
            edges.append((u, v))
            parents[v].append(u)
        elif op == "remove":
            edges = [(a, b) for (a, b) in edges if not (a == u and b == v)]
            parents[v] = [p for p in parents[v] if p != u]
        elif op == "reverse":
            edges = [(a, b) for (a, b) in edges if not (a == u and b == v)]
            parents[v] = [p for p in parents[v] if p != u]
            edges.append((v, u))
            parents[u].append(v)
        else:
            raise ValueError("Unknown op")

    # normalize edge order
    edges.sort()
    return edges


# -----------------------------
# Evaluation
# -----------------------------
def shd(learned_edges: List[Tuple[str, str]], true_edges: List[Tuple[str, str]]) -> int:
    L = set(learned_edges)
    T = set(true_edges)
    extra = L - T
    missing = T - L

    # treat reversed edges as 1 edit (common SHD convention)
    reversed_edges = set()
    for (u, v) in extra:
        if (v, u) in missing:
            reversed_edges.add((u, v))

    return int(len(extra) + len(missing) - len(reversed_edges))


def fit_tabular_mle_counts(train_df: pd.DataFrame, nodes: List[str], edges: List[Tuple[str, str]], k: int):
    """
    Returns parent lists and CPT counts to query P(child|parents) on test rows.
    Stores MLE probabilities for observed parent configs; unseen configs -> uniform fallback.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    parents = {n: sorted(list(G.predecessors(n))) for n in nodes}

    tables: Dict[str, Dict[Tuple[int, ...], np.ndarray]] = {}
    for n in nodes:
        pa = parents[n]
        tables[n] = {}
        if len(pa) == 0:
            cnt = np.bincount(train_df[n].values, minlength=k).astype(np.float64)
            prob = (cnt + 1e-6) / (cnt.sum() + 1e-6 * k)
            tables[n][()] = prob
        else:
            grp = train_df.groupby(pa + [n], observed=True).size().reset_index(name="cnt")
            # Build dict of counts per parent config
            cfg_totals: Dict[Tuple[int, ...], float] = {}
            cfg_counts: Dict[Tuple[int, ...], np.ndarray] = {}
            for _, row in grp.iterrows():
                cfg = tuple(int(row[p]) for p in pa)
                y = int(row[n])
                c = float(row["cnt"])
                if cfg not in cfg_counts:
                    cfg_counts[cfg] = np.zeros(k, dtype=np.float64)
                    cfg_totals[cfg] = 0.0
                cfg_counts[cfg][y] += c
                cfg_totals[cfg] += c

            for cfg, vec in cfg_counts.items():
                prob = (vec + 1e-6) / (vec.sum() + 1e-6 * k)
                tables[n][cfg] = prob

    return parents, tables


def fit_deepcpd_models(train_df: pd.DataFrame, nodes: List[str], edges: List[Tuple[str, str]], k: int, arch: DeepCPDArch, device: str, seed: int):
    """
    Train one neural CPD per node using learned parents.
    Returns parent lists and trained torch models.
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    parents = {n: sorted(list(G.predecessors(n))) for n in nodes}

    models: Dict[str, Optional[CPDNet]] = {}
    for n in nodes:
        pa = parents[n]
        if len(pa) == 0:
            models[n] = None
            continue

        set_seed(seed + hash((n, tuple(pa))) % 10_000_000)
        X = onehot_parents_matrix(train_df, pa, k)
        y = train_df[n].values.astype(np.int64)

        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.long, device=device)

        model = CPDNet(X_t.shape[1], k, arch.hidden, arch.dropout).to(device)
        opt = optim.AdamW(model.parameters(), lr=arch.lr, weight_decay=arch.weight_decay)
        crit = nn.CrossEntropyLoss()

        nrows = len(y)
        rng = np.random.default_rng(seed + 12345 + hash((n, tuple(pa))) % 10_000_000)
        perm = rng.permutation(nrows)
        split = int(0.8 * nrows)
        tr_idx = perm[:split]
        va_idx = perm[split:] if split < nrows else perm[:0]

        Xtr, ytr = X_t[tr_idx], y_t[tr_idx]
        Xva, yva = X_t[va_idx], y_t[va_idx]

        bs = min(arch.batch_size, len(tr_idx)) if len(tr_idx) > 0 else len(tr_idx)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(Xtr, ytr),
            batch_size=max(1, bs),
            shuffle=True,
        )

        best = float("inf")
        best_state = None
        bad = 0

        for epoch in range(arch.max_epochs):
            model.train()
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                loss = crit(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()

            if len(va_idx) > 0:
                model.eval()
                with torch.no_grad():
                    vloss = float(crit(model(Xva), yva).item())
                if vloss < best - 1e-6:
                    best = vloss
                    best_state = {k_: v_.detach().cpu().clone() for k_, v_ in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                if bad >= arch.patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        models[n] = model

    return parents, models


def nll_on_test_tabular(test_df: pd.DataFrame, nodes: List[str], parents: Dict[str, List[str]], tables: Dict[str, Dict[Tuple[int, ...], np.ndarray]], k: int) -> float:
    eps = 1e-12
    total = 0.0
    for _, row in test_df.iterrows():
        for n in nodes:
            pa = parents[n]
            if len(pa) == 0:
                prob = tables[n][()][int(row[n])]
            else:
                cfg = tuple(int(row[p]) for p in pa)
                dist = tables[n].get(cfg, None)
                if dist is None:
                    prob = 1.0 / k
                else:
                    prob = float(dist[int(row[n])])
            total -= math.log(max(prob, eps))
    return total / len(test_df)


def nll_on_test_deepcpd(test_df: pd.DataFrame, nodes: List[str], parents: Dict[str, List[str]], models: Dict[str, Optional[CPDNet]], train_df_for_roots: pd.DataFrame, k: int, device: str) -> float:
    eps = 1e-12

    # Root dists from training data (smoothed)
    root_prob: Dict[str, np.ndarray] = {}
    for n in nodes:
        if len(parents[n]) == 0:
            cnt = np.bincount(train_df_for_roots[n].values, minlength=k).astype(np.float64)
            root_prob[n] = (cnt + 1e-6) / (cnt.sum() + 1e-6 * k)

    total = 0.0
    # Vectorize per node for speed
    for n in nodes:
        pa = parents[n]
        y = test_df[n].values.astype(np.int64)
        if len(pa) == 0:
            probs = root_prob[n][y]
            total -= float(np.log(np.maximum(probs, eps)).sum())
        else:
            X = onehot_parents_matrix(test_df, pa, k)
            X_t = torch.tensor(X, dtype=torch.float32, device=device)
            y_t = torch.tensor(y, dtype=torch.long, device=device)
            model = models[n]
            assert model is not None
            model.eval()
            with torch.no_grad():
                logp = torch.log_softmax(model(X_t), dim=1)
                total -= float(logp[torch.arange(len(y_t), device=device), y_t].sum().item())

    return total / len(test_df)


# -----------------------------
# Task grid / runner
# -----------------------------
def enumerate_tasks(cfg: dict) -> List[dict]:
    tasks = []
    for n_nodes in cfg["n_nodes_list"]:
        for card in cfg["cardinality_list"]:
            for max_indeg in cfg["max_indegree_list"]:
                for N in cfg["dataset_size_list"]:
                    for seed in cfg["structure_seed_list"]:
                        for method in cfg["methods"]:
                            tasks.append({
                                "n_nodes": n_nodes,
                                "cardinality": card,
                                "max_indegree": max_indeg,
                                "dataset_size": N,
                                "structure_seed": seed,
                                "method": method,
                            })
    return tasks


def run_single_task(task: dict, cfg: dict, outputdir: str, debug: bool = False) -> str:
    t0 = time.time()
    set_seed(cfg["global_seed"] + task["structure_seed"])

    n_nodes = task["n_nodes"]
    k = task["cardinality"]
    N = task["dataset_size"]
    max_indeg = task["max_indegree"]
    seed = task["structure_seed"]
    method = task["method"]

    nodes = [f"N{i}" for i in range(n_nodes)]

    # Ground-truth DAG + data
    true_edges = generate_random_dag(
        nodes=nodes,
        max_indegree=max_indeg,
        edge_prob=cfg["edge_prob"],
        seed=seed,
    )
    df = sample_from_bn(
        nodes=nodes,
        edges=true_edges,
        cardinality=k,
        n_samples=N,
        seed=seed + 999,
    )

    # Train/test split
    rng = np.random.default_rng(seed + 2026)
    perm = rng.permutation(len(df))
    split = int(cfg["train_frac"] * len(df))
    tr_idx = perm[:split]
    te_idx = perm[split:]
    train_df = df.iloc[tr_idx].copy()
    test_df = df.iloc[te_idx].copy()

    # Candidate restriction (MI)
    top_k = int(cfg["candidate_k_mult"] * max_indeg)
    candidates = compute_mi_candidates(train_df, top_k=top_k)

    # Scorer + structure learning
    device = cfg["device"]
    if method == "tabular_bic":
        scorer = TabularBICScorer(train_df, cardinality=k)
    elif method == "deepcpd_mbic":
        arch = DeepCPDArch(**cfg["deep_arch_search"])
        scorer = DeepCPDmBICScorer(
            train_df,
            cardinality=k,
            arch=arch,
            alpha=cfg["deep_alpha"],
            device=device,
            seed=seed,
            debug=debug,
        )
    else:
        raise ValueError(f"Unknown method={method}")

    t_search0 = time.time()
    learned_edges = hill_climb(
        nodes=nodes,
        scorer=scorer,
        candidates=candidates,
        max_indegree=max_indeg,
        max_iters=cfg["hc_max_iters"],
        debug=debug,
    )
    search_sec = time.time() - t_search0

    # Fit final CPDs and evaluate predictive NLL on held-out test
    t_fit0 = time.time()
    if method == "tabular_bic":
        pa, tables = fit_tabular_mle_counts(train_df, nodes, learned_edges, k)
        test_nll = nll_on_test_tabular(test_df, nodes, pa, tables, k)
    else:
        arch_final = DeepCPDArch(**cfg["deep_arch_final"])
        pa, models = fit_deepcpd_models(
            train_df, nodes, learned_edges, k, arch_final, device, seed=seed + 777
        )
        test_nll = nll_on_test_deepcpd(test_df, nodes, pa, models, train_df, k, device)


    # Structure metric (secondary, synthetic only)
    shd_val = shd(learned_edges, true_edges)

    out = {
        "RQ": "RQ2",
        "method": method,
        "n_nodes": n_nodes,
        "cardinality": k,
        "max_indegree": max_indeg,
        "dataset_size": N,
        "train_frac": cfg["train_frac"],
        "structure_seed": seed,
        "candidate_top_k": top_k,
        "true_edges": len(true_edges),
        "learned_edges": len(learned_edges),
        "SHD": shd_val,
        "test_NLL": float(test_nll),
        "device": device,
        "cfg_deep_alpha": float(cfg["deep_alpha"]),
        "cfg_deep_arch_search": json.dumps(cfg["deep_arch_search"]),
        "cfg_deep_arch_final": json.dumps(cfg["deep_arch_final"]),
    }

    os.makedirs(outputdir, exist_ok=True)
    fname = (
        f"rq2_{method}_seed{seed}_N{N}_nodes{n_nodes}_k{k}_maxin{max_indeg}.csv"
    )
    path = os.path.join(outputdir, fname)
    pd.DataFrame([out]).to_csv(path, index=False)
    return path


def merge_results(outputdir: str) -> Tuple[str, str]:
    import glob
    files = sorted(glob.glob(os.path.join(outputdir, "rq2_*.csv")))
    if not files:
        raise FileNotFoundError(f"No rq2_*.csv files in {outputdir}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    combined_path = os.path.join(outputdir, "rq2_combined.csv")
    df.to_csv(combined_path, index=False)

    agg = (
        df.groupby(["method", "n_nodes", "cardinality", "max_indegree", "dataset_size"], as_index=False)
          .agg(
              mean_test_NLL=("test_NLL", "mean"),
              std_test_NLL=("test_NLL", "std"),
              mean_SHD=("SHD", "mean"),
              std_SHD=("SHD", "std"),
              mean_edges=("learned_edges", "mean"),
 #             mean_search_sec=("search_sec", "mean"),
#              mean_fit_sec=("fit_sec", "mean"),
              runs=("test_NLL", "count"),
          )
    )
    summary_path = os.path.join(outputdir, "rq2_summary.csv")
    agg.to_csv(summary_path, index=False)
    return combined_path, summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=None, help="Global task id (0..num_tasks-1).")
    parser.add_argument("--outputdir", type=str, default="RQ2_RESULTS")
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--print_tasks", action="store_true")
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--run_all", action="store_true", help="Run all tasks sequentially")
    args = parser.parse_args()

    # Device: set to 'cuda' if you have GPU nodes available, else 'cpu'
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------------
    # RQ2 config
    # -----------------------------
    CFG = {
        "global_seed": 123,
        "device": device,
        "edge_prob": 0.20,
        "train_frac": 0.8,
        "hc_max_iters": 250,

        # Candidate restriction: top_k = candidate_k_mult * max_indegree
        "candidate_k_mult": 6,

        # Base alpha0, later scaled by (1 + indegree) in the scorer
        "deep_alpha": 10000.0,

        # Small nets for structure scoring (search)
        "deep_arch_search": {
            "hidden": (32,),        # compact MLP for mBIC scoring
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 15,
            "patience": 4,
            "batch_size": 256,
        },

        # Richer nets for final CPD estimation (RQ1-style best config)
        "deep_arch_final": {
            "hidden": (64, 64),     # two hidden layers, moderate width
            "dropout": 0.10,
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "max_epochs": 40,       # more training for final CPDs
            "patience": 8,
            "batch_size": 512,
        },

        # Task grid for RQ2
        "n_nodes_list": [40],
        "cardinality_list": [4, 6],
        "max_indegree_list": [5, 7],
        "dataset_size_list": [1000, 3000],
        "structure_seed_list": [200, 201],
        "methods": ["tabular_bic", "deepcpd_mbic"],
    }

    tasks = enumerate_tasks(CFG)

    if args.print_tasks:
        print(f"Total tasks: {len(tasks)}")
        for i, t in enumerate(tasks):
            print(i, t)
        return

    if args.merge:
        combined_path, summary_path = merge_results(args.outputdir)
        print("Wrote:", combined_path)
        print("Wrote:", summary_path)
        return

    debug = bool(args.debug)

    # Run the full grid sequentially
    if args.run_all:
        os.makedirs(args.outputdir, exist_ok=True)
        for i, t in enumerate(tasks):
            print(f"=== Running task {i}/{len(tasks)-1}: {t} ===")
            run_single_task(t, CFG, args.outputdir, debug=debug)
        print("All tasks finished.")
        return

    # Single-task mode (for arrays / debugging)
    if args.task_id is None:
        raise SystemExit("Provide --task_id or use --merge / --print_tasks / --run_all")

    if args.task_id < 0 or args.task_id >= len(tasks):
        raise SystemExit(f"--task_id out of range: 0..{len(tasks)-1}")

    path = run_single_task(tasks[args.task_id], CFG, args.outputdir, debug=debug)
    print("Wrote:", path)



if __name__ == "__main__":
    main()
