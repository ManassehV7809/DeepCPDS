import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── directories ───────────────────────────────────────────────────────────────
RQ3_DIR    = "RQ3_RESULTS"
SCARCE_DIR = "RQ3_SCARCE_RESULTS"
PLOT_DIR   = "RQ3_PLOTS"
os.makedirs(PLOT_DIR, exist_ok=True)

# ── load CSVs ─────────────────────────────────────────────────────────────────
rq3        = pd.read_csv(f"{RQ3_DIR}/rq3_results.csv")
per_node   = pd.read_csv(f"{RQ3_DIR}/rq3_per_node_nll.csv")
arch_sweep = pd.read_csv(f"{RQ3_DIR}/rq3_arch_sweep.csv")
scarce     = pd.read_csv(f"{SCARCE_DIR}/rq3_scarce_results.csv")

# ── shared colours ────────────────────────────────────────────────────────────
TAB_COL  = "#4C72B0"
DEEP_COL = "#DD8452"

def save(name):
    path = f"{PLOT_DIR}/{name}.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — 2x2 grouped bar: Test NLL by Structure × CPD
# Best as a grouped bar — 4 conditions, one continuous metric
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4))
bic_rows  = rq3[rq3["Structure"] == "BIC"].reset_index(drop=True)
mbic_rows = rq3[rq3["Structure"] == "mBIC"].reset_index(drop=True)
x = [0, 1]
w = 0.3

ax.bar([i - w/2 for i in x],
       [bic_rows.loc[bic_rows["CPD"]=="Tabular",  "Test_NLL"].values[0],
        mbic_rows.loc[mbic_rows["CPD"]=="Tabular", "Test_NLL"].values[0]],
       width=w, label="Tabular", color=TAB_COL)
ax.bar([i + w/2 for i in x],
       [bic_rows.loc[bic_rows["CPD"]=="DeepCPD",  "Test_NLL"].values[0],
        mbic_rows.loc[mbic_rows["CPD"]=="DeepCPD", "Test_NLL"].values[0]],
       width=w, label="DeepCPD", color=DEEP_COL)

ax.set_xticks(x)
ax.set_xticklabels(["BIC structure", "mBIC structure"])
ax.set_ylabel("Mean Test NLL (avg over nodes)")
ax.set_title("RQ3: Test NLL — Structure × CPD (MIMIC-IV, Full Data)")
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.4)
save("rq3_2x2_nll")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — PR-AUC grouped bar (only if sepsis had parents in both structures)
# Skip if all PR_AUC values are None
# ══════════════════════════════════════════════════════════════════════════════
rq3_prauc = rq3.dropna(subset=["PR_AUC"])
if len(rq3_prauc) >= 2:
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (score_name, xpos) in enumerate([("BIC", 0), ("mBIC", 1)]):
        subset = rq3_prauc[rq3_prauc["Structure"] == score_name]
        tab_val  = subset.loc[subset["CPD"] == "Tabular",  "PR_AUC"].values
        deep_val = subset.loc[subset["CPD"] == "DeepCPD", "PR_AUC"].values
        if len(tab_val):
            ax.bar(xpos - w/2, tab_val[0],  width=w, color=TAB_COL,
                   label="Tabular"  if i == 0 else "")
        if len(deep_val):
            ax.bar(xpos + w/2, deep_val[0], width=w, color=DEEP_COL,
                   label="DeepCPD" if i == 0 else "")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["BIC structure", "mBIC structure"])
    ax.set_ylabel("PR-AUC (Sepsis prediction)")
    ax.set_title("RQ3: Sepsis PR-AUC — Structure × CPD (MIMIC-IV)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    save("rq3_2x2_prauc")
else:
    print("Skipped rq3_2x2_prauc — insufficient PR-AUC values (sepsis likely a root node)")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — Per-node NLL heatmap (nodes × Structure+CPD combos)
# Heatmap is natural here — 13 nodes × 4 conditions
# ══════════════════════════════════════════════════════════════════════════════
try:
    per_node["Condition"] = per_node["Structure"] + " + " + per_node["CPD"]
    pivot = per_node.pivot_table(index="Node", columns="Condition", values="NLL")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax)
    ax.set_title("Per-node Test NLL — Structure × CPD (MIMIC-IV)")
    ax.set_xlabel("")
    ax.set_ylabel("Node")
    plt.xticks(rotation=20, ha="right")
    save("rq3_per_node_heatmap")
except Exception as e:
    print(f"Skipped rq3_per_node_heatmap: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 4 — Architecture sweep: horizontal bar (mBIC only, most relevant)
# Horizontal bar is natural — comparing named architectures on one metric
# ══════════════════════════════════════════════════════════════════════════════
mbic_arch = arch_sweep[arch_sweep["Structure"] == "mBIC"].sort_values("Test_NLL", ascending=True)
if not mbic_arch.empty:
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(mbic_arch["Arch"], mbic_arch["Test_NLL"], color=DEEP_COL)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.set_xlabel("Mean Test NLL")
    ax.set_title("Architecture Sweep — DeepCPD on mBIC Structure (MIMIC-IV)")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    save("rq3_arch_sweep_mbic")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 5 — Scarce crossover: NLL vs training size, BIC and mBIC side by side
# Line + error bars — continuous trend across sizes, two conditions per panel
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, score_name in zip(axes, ["BIC", "mBIC"]):
    subset = scarce[scarce["Structure"] == score_name]
    for cpd_name, marker, color in [
        ("Tabular", "o", TAB_COL),
        ("DeepCPD", "s", DEEP_COL),
    ]:
        cpd_sub = subset[subset["CPD"] == cpd_name].sort_values("Train_size")
        ax.errorbar(
            cpd_sub["Train_size"],
            cpd_sub["NLL_mean"],
            yerr=cpd_sub["NLL_std"],
            label=cpd_name, marker=marker,
            color=color, capsize=4, linewidth=2,
        )
    ax.set_xscale("log")
    ax.set_xticks(sorted(scarce["Train_size"].unique()))
    ax.set_xticklabels([str(s) for s in sorted(scarce["Train_size"].unique())])
    ax.set_xlabel("Training set size (log scale)")
    ax.set_ylabel("Mean Test NLL")
    ax.set_title(f"{score_name} structure")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

plt.suptitle(
    "DeepCPD vs Tabular under Data Scarcity (MIMIC-IV)\n"
    "End-to-end: structure + CPDs re-learned per subset"
)
plt.tight_layout()
save("rq3_scarce_crossover")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 6 — Delta NLL bar: how much DeepCPD wins/loses at each size
# Bar chart with +/- axis — immediately shows regime boundary
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

for ax, score_name in zip(axes, ["BIC", "mBIC"]):
    subset = scarce[scarce["Structure"] == score_name]
    tab    = subset[subset["CPD"] == "Tabular"].set_index("Train_size")["NLL_mean"]
    deep   = subset[subset["CPD"] == "DeepCPD"].set_index("Train_size")["NLL_mean"]
    sizes  = sorted(set(tab.index) & set(deep.index))
    deltas = [tab[s] - deep[s] for s in sizes]
    colors = ["#2ca02c" if d > 0 else "#d62728" for d in deltas]

    bars = ax.bar([str(s) for s in sizes], deltas, color=colors, width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Training set size")
    ax.set_ylabel("Δ NLL  (Tabular − DeepCPD)")
    ax.set_title(f"{score_name} structure\n(green = DeepCPD wins, red = Tabular wins)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.suptitle("DeepCPD Advantage Under Data Scarcity (MIMIC-IV)")
plt.tight_layout()
save("rq3_scarce_delta_nll")


# ══════════════════════════════════════════════════════════════════════════════
# PLOT 7 — Scarce PR-AUC (only if enough non-null values exist)
# Line plot — same logic as crossover but for sepsis classification quality
# ══════════════════════════════════════════════════════════════════════════════
scarce_prauc = scarce.dropna(subset=["PR_AUC_mean"])
min_points_needed = 4  # need at least 2 sizes × 2 CPDs to make a line plot meaningful
if len(scarce_prauc) >= min_points_needed:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, score_name in zip(axes, ["BIC", "mBIC"]):
        subset = scarce_prauc[scarce_prauc["Structure"] == score_name]
        for cpd_name, marker, color in [
            ("Tabular", "o", TAB_COL),
            ("DeepCPD", "s", DEEP_COL),
        ]:
            cpd_sub = subset[subset["CPD"] == cpd_name].sort_values("Train_size")
            if len(cpd_sub) < 2:
                continue
            ax.plot(
                cpd_sub["Train_size"],
                cpd_sub["PR_AUC_mean"],
                label=cpd_name, marker=marker,
                color=color, linewidth=2,
            )
        ax.set_xscale("log")
        ax.set_xlabel("Training set size (log scale)")
        ax.set_ylabel("PR-AUC (Sepsis)")
        ax.set_title(f"{score_name} structure")
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.suptitle("Sepsis PR-AUC under Data Scarcity (MIMIC-IV)")
    plt.tight_layout()
    save("rq3_scarce_prauc")
else:
    print("Skipped rq3_scarce_prauc — too few non-null PR-AUC values "
          "(sepsis likely a root node at small training sizes)")


print(f"\nAll plots saved to {PLOT_DIR}/")
