import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Publication styling
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "serif",
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
})

COLORS = {
    'Tabular BN': '#d62728',     # Red
    'DeepCPD': '#1f77b4',        # Blue
    'Decision Tree': '#2ca02c',  # Green
    'Logistic Regression': '#ff7f0e' # Orange
}

LINE_STYLES = {
    'Tabular BN': ':',
    'DeepCPD': '-',
    'Decision Tree': '--',
    'Logistic Regression': '-.'
}

def load_data(file_path):
    df = pd.read_csv(file_path)
    # Filter to default architecture config (depth 5, relu, adamw)
    df = df[(df['arch_depth'] == 5) & (df['activation'] == 'relu') & (df['optimizer'] == 'adamw')]
    return df

def plot_shared_axis_panels(df, sweep_param, sweep_values, fixed_param, fixed_value,
                            title, ylabel, metric_cols, save_path):
    """
    Creates a 1x3 grid of subplots with strictly shared Y-axes to satisfy examiner feedback.
    """
    # Create 1x3 figure with shared Y-axis
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    metric_map = {
        metric_cols['nn']: 'DeepCPD',
        metric_cols['tab']: 'Tabular BN',
        metric_cols['dt']: 'Decision Tree',
        metric_cols['lr']: 'Logistic Regression'
    }

    # Filter for the fixed parameter
    base_subset = df[df[fixed_param] == fixed_value].copy()

    for idx, (ax, val) in enumerate(zip(axes, sweep_values)):
        # Filter for the specific panel's parameter
        subset = base_subset[base_subset[sweep_param] == val]

        if subset.empty:
            ax.set_title(f"{sweep_param.replace('_', ' ').title()}: {val}\n(No Data)")
            continue

        for col, label in metric_map.items():
            if col in subset.columns:
                ax.plot(
                    subset['dataset_size'],
                    subset[col],
                    marker='o',
                    markersize=6,
                    linewidth=2,
                    color=COLORS[label],
                    linestyle=LINE_STYLES[label],
                    label=label if idx == 0 else "" # Only add legend labels on the first panel
                )

                # Add error bands
                std_col = col.replace('_mean', '_std')
                if std_col in subset.columns:
                    ax.fill_between(
                        subset['dataset_size'],
                        subset[col] - subset[std_col],
                        subset[col] + subset[std_col],
                        color=COLORS[label],
                        alpha=0.1
                    )

        # Panel formatting
        param_clean_name = "Cardinality" if "cardinality" in sweep_param else "Max Indegree"
        ax.set_title(f"{param_clean_name}: {val}", fontweight='bold')
        ax.set_xlabel('Dataset Size (N)')
        ax.set_yscale('log') # Log scale across all panels handles the tabular explosion gracefully
        ax.grid(True, which="both", ls="-", alpha=0.3)

    # Set the y-label only on the first axis (since they are shared)
    axes[0].set_ylabel(f"{ylabel} (Log Scale)")

    # Global figure formatting
    fixed_clean_name = "Cardinality" if "cardinality" in fixed_param else "Max Indegree"
    fig.suptitle(f"{title} (Fixed {fixed_clean_name}: {fixed_value})", y=1.05, fontsize=16, fontweight='bold')

    # Place a single legend outside the plots
    fig.legend(title='Parameterisation Method', loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def generate_all_plots():
    data_path = "RQ1_RESULTS_BASELINES/rq1_across_structures_summary.csv"
    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return

    df = load_data(data_path)
    os.makedirs("RQ1_PLOTS", exist_ok=True)

    kl_metrics = {'nn': 'mean_kl_nn_mean', 'tab': 'mean_kl_tab_mean', 'dt': 'mean_kl_dt_mean', 'lr': 'mean_kl_lr_mean'}
    nll_metrics = {'nn': 'mean_nll_nn_mean', 'tab': 'mean_nll_tab_mean', 'dt': 'mean_nll_dt_mean', 'lr': 'mean_nll_lr_mean'}

    print("Generating Figure 4.1: High Complexity Boundary...")
    plot_shared_axis_panels(
        df, sweep_param='max_indegree', sweep_values=[3, 5, 7],
        fixed_param='node_cardinality', fixed_value=6,
        title="Regime Boundary Across Indegrees",
        ylabel="Mean KL Divergence", metric_cols=kl_metrics,
        save_path="RQ1_PLOTS/fig4_1_regime_boundary_KL.png"
    )

    print("Generating Figure 4.5: Cardinality Effect...")
    plot_shared_axis_panels(
        df, sweep_param='node_cardinality', sweep_values=[2, 4, 6],
        fixed_param='max_indegree', fixed_value=5,
        title="The Effect of Node Cardinality",
        ylabel="Mean KL Divergence", metric_cols=kl_metrics,
        save_path="RQ1_PLOTS/fig4_5_cardinality_effect_KL.png"
    )

    print("Generating Figure 4.6: Indegree Effect...")
    plot_shared_axis_panels(
        df, sweep_param='max_indegree', sweep_values=[3, 5, 7],
        fixed_param='node_cardinality', fixed_value=4,
        title="The Effect of Parent Set Size",
        ylabel="Mean KL Divergence", metric_cols=kl_metrics,
        save_path="RQ1_PLOTS/fig4_6_indegree_effect_KL.png"
    )

    print("Generating Figure 4.7: NLL Equivalent...")
    plot_shared_axis_panels(
        df, sweep_param='max_indegree', sweep_values=[3, 5, 7],
        fixed_param='node_cardinality', fixed_value=6,
        title="Predictive Negative Log-Likelihood",
        ylabel="Mean Test NLL", metric_cols=nll_metrics,
        save_path="RQ1_PLOTS/fig4_7_nll_equivalent.png"
    )

    print("\nSuccess! Shared-axis plots saved to RQ1_PLOTS/.")

if __name__ == "__main__":
    generate_all_plots()
