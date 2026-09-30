"""
PRISM ablation plot — both datasets, with Stage 3 token count added.

Generates a single side-by-side figure showing all metrics across the four
ablation settings, for YouCook2 and ActivityNet Captions.

Quality metrics (BLEU, ROUGE-L, BERTScore, METEOR, LLM-Judge) on left y-axis.
Cost metrics (frames, time in seconds, Stage 3 tokens) on right y-axis (log).
"""

import matplotlib.pyplot as plt
import numpy as np

SETTINGS = ["Video-Only", "No Stage 2", "No Stage 1", "No Processing"]

# YouCook2 (main paper Table 5)
yc2 = {
    "BLEU":       [1.82, 3.00, 2.80, 2.70],
    "ROUGE-L":    [18.40, 28.11, 29.87, 28.03],
    "BERTScore":  [82.41, 84.12, 84.59, 84.13],
    "METEOR":     [19.83, 31.83, 32.66, 31.83],
    "LLM-Judge":  [38.64, 79.26, 78.49, 80.82],
    "Frames":     [22.25, 60.07, 22.23, 322.91],
    "Time (s)":   [71.78, 73.09, 236.46, 1625.40],
    "Stage 3 Tokens": [3560, 9613, 3557, 206662],
}

# ActivityNet Captions (appendix Table 7)
acn = {
    "BLEU":       [0.97, 0.99, 0.97, 0.91],
    "ROUGE-L":    [9.04, 8.77, 8.68, 7.87],
    "BERTScore":  [80.98, 81.19, 81.29, 80.56],
    "METEOR":     [17.29, 17.59, 17.47, 16.09],
    "LLM-Judge":  [37.54, 43.33, 43.57, 43.75],
    "Frames":     [18.39, 20.04, 17.39, 111.88],
    "Time (s)":   [40.37, 28.52, 88.69, 130.06],
    "Stage 3 Tokens": [2942, 3206, 2782, 71603],
}

QUALITY = ["BLEU", "ROUGE-L", "BERTScore", "METEOR", "LLM-Judge"]
COST = ["Frames", "Time (s)", "Stage 3 Tokens"]

QUAL_COLORS = {
    "BLEU":      "#1f77b4",
    "ROUGE-L":   "#ff7f0e",
    "BERTScore": "#2ca02c",
    "METEOR":    "#d62728",
    "LLM-Judge": "#9467bd",
}
COST_COLORS = {
    "Frames":           "#8c564b",
    "Time (s)":         "#e377c2",
    "Stage 3 Tokens":   "#17becf",
}


def plot_panel(ax, data, title):
    x = np.arange(len(SETTINGS))

    # Quality metrics on primary y-axis (linear, 0–100)
    for m in QUALITY:
        ax.plot(x, data[m], marker="o", label=m,
                color=QUAL_COLORS[m], linewidth=1.6)
    ax.set_ylabel("Quality Score (0–100)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(SETTINGS, rotation=20, ha="right")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Cost metrics on secondary y-axis (log scale)
    ax2 = ax.twinx()
    for m in COST:
        ax2.plot(x, data[m], marker="s", linestyle="--", label=m,
                 color=COST_COLORS[m], linewidth=1.4, alpha=0.85)
    ax2.set_ylabel("Cost (log scale)")
    ax2.set_yscale("log")

    return ax, ax2


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
ax1q, ax1c = plot_panel(axes[0], yc2, "YouCook2")
ax2q, ax2c = plot_panel(axes[1], acn, "ActivityNet Captions")

# One combined legend below the figure
qual_handles, qual_labels = ax1q.get_legend_handles_labels()
cost_handles, cost_labels = ax1c.get_legend_handles_labels()
fig.legend(qual_handles + cost_handles,
           qual_labels + cost_labels,
           loc="lower center",
           ncol=4,
           bbox_to_anchor=(0.5, -0.06),
           frameon=False,
           fontsize=9)

plt.suptitle("Ablation Study: Quality and Cost Metrics Across Settings",
             y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig("figures/ablation_plot_both.png",
            dpi=200, bbox_inches="tight")
plt.savefig("figures/ablation_plot_both.pdf",
            bbox_inches="tight")
print("Saved both PNG and PDF.")
