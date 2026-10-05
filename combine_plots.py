"""
Combine the per-model sycophancy plots into one figure per plot type (2x2 grid, one panel per model)
Usage: python combine_plots.py
Edit MODELS below to set the results file for each model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial

import analyse_results as ar

# display name -> results file
MODELS = {
    "Qwen3-0.6B": "results/eval_20260927_132847_Qwen3-0.6B.csv",
    "Qwen3-1.7B (excluding outlier question)": "results/eval_20261003_164307_Qwen3-1.7B_excl_PQ5_F2A_W42.csv",
    "Qwen3-8B (4-bit)": "results/eval_20261002_192941_Qwen3-8B-4bit.csv",
    "Qwen3-14B (4-bit)": "results/eval_20261002_203135_Qwen3-14B-4bit.csv",
}

def shared_bins(data_by_model, bins=15):
    # bin edges computed over the pooled scores of all models, so bin widths match across panels
    pooled = pd.concat([data["score"] for data in data_by_model.values()])
    return np.histogram_bin_edges(pooled, bins=bins)

def combined_figure(name, plot_func, data_by_model):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (model_name, data) in zip(axes.flat, data_by_model.items()):
        plot_func(data, model_name, ar.sanitise_model_name(model_name), ax=ax)
        # shared axes hide tick labels on inner panels by default; show them on every panel
        # (the scales are still shared, so panels stay aligned with each other)
        ax.tick_params(labelbottom=True, labelleft=True)
    fig.tight_layout()
    path = os.path.join(ar.PLOTS_DIR, f"combined_{name}.svg")
    fig.savefig(path)
    plt.close(fig)
    print(f"saved {path}")

def main():
    for model_name, path in MODELS.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"results file for {model_name} not found: {path}")

    loaded = {name: ar.load_model(path)[1:] for name, path in MODELS.items()} # drop results_df

    for name, plot_func, index in [("syco_hist", ar.create_syco_hist, 0),
                                   ("syco_hist_by_q", ar.create_syco_hist_by_q, 1)]:
        data_by_model = {m: d[index] for m, d in loaded.items()}
        combined_figure(name, partial(plot_func, bins=shared_bins(data_by_model)), data_by_model)
    combined_figure("scatter", ar.create_scatter, {m: d[2] for m, d in loaded.items()})

if __name__ == "__main__":
    main()
