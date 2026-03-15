# DeepCPD: Neural Conditional Probability Distributions for Bayesian Networks

> **MSc Dissertation** — *Introducing DeepCPDs for Bayesian Network Parameterization*
> University of the Witwatersrand, Johannesburg · 2026
> Vusani Radzilani · Supervised by Prof. Ritesh Ajoodha & Dr. Rudzani Mulaudzi

---

## Overview

This repository provides all experimental code for the three research questions addressed in the dissertation:

| # | Research Question | Scripts |
|---|---|---|
| RQ1 | When do neural CPDs outperform tabular CPDs? | `rq1.py`, `plot_rq1.py` |
| RQ2 | Can mBIC align structure learning with neural models? | `rq2.py`, `combine.py`, `run_stats.py` |
| RQ3 | Does the framework transfer to real clinical data? | `rq3.py`, `rq3_scarce.py`, `rq3p2.py`, `rq3_baselines.py` |

---

## Repository Structure

### Top-level scripts

| Script | Purpose |
|---|---|
| `rq1.py` | RQ1 synthetic CPD experiments |
| `rq2.py` | RQ2 structure learning experiments |
| `rq3.py` | RQ3 main MIMIC-IV experiments |
| `rq3_scarce.py` | RQ3 data scarcity sensitivity analysis |
| `rq3p2.py` | RQ3 architecture sweeps and variants |
| `rq3_baselines.py` | Standalone discriminative baselines (LR, RF) |
| `mimic_loader.py` | MIMIC-IV loading and preprocessing helpers |
| `build_rq3_cohort.py` / `build_rq3_cohort_2.py` | Sepsis cohort construction from raw MIMIC-IV |
| `preprocess_rq3_cohort.py` / `preprocess_rq3_cohort_2.py` | Feature engineering and discretisation for RQ3 |
| `plot_rq1.py` / `plot.py` | Figure generation for RQ1 and RQ3 |
| `run_rq1_stats.py` / `run_stats.py` | Post-hoc statistics and aggregation |
| `combine.py` | Merge per-job CSVs into combined result tables |

### Result and plot directories

| Directory | Contents |
|---|---|
| `RQ1_RESULTS_BASELINES`, `RQ1_RESULTS_COMBINED`, `RQ1_RESULTS_COMBINED_2` | RQ1 result CSVs |
| `RQ2_RESULTS`, `RQ2_RESULTS0–2`, `RQ2_PILOT_RESULTS` | RQ2 result CSVs |
| `RQ3_RESULTS`, `RQ3_RESULTS_BB_31126`, `RQ3_SCARCE_RESULTS` | RQ3 result CSVs |
| `RQ1_PLOTS`, `RQ3_PLOTS` | PNG figures used in the dissertation |

### Data artefacts (not included)

The following files are derived from MIMIC-IV and are **not** distributed in this repository due to data use agreements:

```
mimic_sepsis_rq3_cohort.csv
mimic_sepsis_rq3_extended.csv
rq3_clinical_extended_discretised.csv
rq3_full_discretised.csv
rq3_metadata.json
```

Users must obtain MIMIC-IV access independently and rebuild these artefacts using the cohort-building and preprocessing scripts described below.

---

## Environment and Dependencies

Experiments were run with **Python 3** on a SLURM-based HPC cluster. Core dependencies are:

```
numpy
pandas
scikit-learn
networkx
pgmpy
torch
matplotlib
seaborn
tqdm
```

To set up a local environment:

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

---

## Reproducing RQ1: CPD Experiments

RQ1 compares **Tabular BN** (BDeu-smoothed), **DT-CPD**, **LR-CPD**, and **DeepCPD** on synthetic Bayesian networks across controlled grids of node cardinality, maximum indegree, and dataset size.

### Run the experiments

```bash
python rq1.py
```

This script samples random BN structures, generates synthetic data, fits each CPD model on training data, evaluates conditional KL divergence and predictive NLL on held-out data, and writes per-configuration result CSVs to `RQ1_RESULTS_BASELINES/` and `RQ1_RESULTS_COMBINED*/`. The experimental grid (cardinality, indegree, dataset sizes, seeds) is configurable at the top of `rq1.py`.

### Generate figures

```bash
python plot_rq1.py
```

Reads the combined RQ1 CSVs and produces regime boundary plots, cardinality and indegree effect plots, and predictive NLL parity plots. All figures are written to `RQ1_PLOTS/` as PNG files.

---

## Reproducing RQ2: Structure Learning with mBIC

RQ2 studies greedy hill-climbing structure search under classical BIC versus the proposed mBIC, using both Tabular BN and DeepCPD parameterisations.

### Run the experiments

```bash
python rq2.py
```

The script samples synthetic BN structures at specified complexity settings, performs hill-climbing search under BIC and mBIC, fits Tabular and DeepCPD models, and saves metrics (test NLL, Structural Hamming Distance, edge counts) to the `RQ2_RESULTS*/` directories. DeepCPD hyperparameters are defined inside `rq2.py` and match the dissertation configuration.

### Aggregate and compute statistics

```bash
python combine.py
python run_stats.py
```

These scripts merge individual job outputs into combined CSVs and compute aggregated statistics (means, standard deviations) for the tables reported in Chapter 5.

---

## Reproducing RQ3: MIMIC-IV Experiments

RQ3 applies the DeepCPD + mBIC framework to a discretised MIMIC-IV sepsis cohort. Because MIMIC-IV is access-restricted, this repository provides only the scripts to construct and preprocess the cohort on a system where raw MIMIC-IV tables are available.

### Step 1 — Build the cohort

```bash
python build_rq3_cohort.py
# or, for an alternate cohort definition:
python build_rq3_cohort_2.py
```

Reads from the local MIMIC-IV installation and writes `mimic_sepsis_rq3_cohort.csv` and `mimic_sepsis_rq3_extended.csv`.

### Step 2 — Preprocess and discretise

```bash
python preprocess_rq3_cohort.py
python preprocess_rq3_cohort_2.py
```

Performs feature selection and engineering, train/validation/test splitting, and clinical variable discretisation. Produces `rq3_clinical_extended_discretised.csv`, `rq3_full_discretised.csv`, and `rq3_metadata.json`.

### Step 3 — Run the main BN experiments

```bash
python rq3.py
```

Loads the discretised cohort via `mimic_loader.py`, trains Tabular BN and DeepCPD models under BIC and mBIC, evaluates structural metrics (edge counts, SHD) and predictive metrics (per-node NLL, sepsis PR-AUC, F1), and writes outputs to `RQ3_RESULTS/` and `RQ3_RESULTS_BB_31126/`.

### Step 4 — Scarcity and architecture analyses

```bash
# Data scarcity sensitivity
python rq3_scarce.py

# Architecture sweeps
python rq3p2.py
```

`rq3_scarce.py` subsamples training data at multiple fractions and evaluates Tabular vs DeepCPD performance under both scoring criteria, writing results to `RQ3_SCARCE_RESULTS/`. `rq3p2.py` sweeps DeepCPD architectures on fixed learned structures and produces CSVs for the architecture-sensitivity analysis.

### Step 5 — Standalone discriminative baselines

```bash
python rq3_baselines.py
```

Trains logistic regression and random forest classifiers on the discretised cohort, evaluates sepsis prediction performance (PR-AUC, F1, NLL), and writes a summary to `rq3_baseline_results.csv`.

### Generate RQ3 figures

```bash
python plot.py
```

Reads the RQ3 result CSVs and generates structural comparison plots, predictive performance plots, and scarcity curves. All figures are saved to `RQ3_PLOTS/`.

---

## HPC and Shell Scripts

The repository includes shell scripts used on the Wits `mscluster` HPC system (`submit_all.sh`, `train_job.sh`, `rq2all.sh`, `rq3_scarce.sh`, and others). These wrap the Python commands in SLURM job submissions and configure resource requests and environment setup. They are cluster-specific and not required for local execution. To run on a different cluster, replace the SLURM directives with those appropriate for your scheduler and adjust paths as needed.

---

## Citation

If you use this code in academic work, please cite:

```
V. Radzilani. Introducing DeepCPDs for Bayesian Network Parameterization.
MSc Dissertation, University of the Witwatersrand, 2026.
```
