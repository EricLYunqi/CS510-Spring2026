import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

method = [
    "PageContext",
    "Baseline-18",
    "Baseline-14"
]

LLMs_backend = [
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Claude-Haiku-4.5",
    "ChatGPT-5-mini",
    "DeepSeek-V4-Flash",
    "Kimi-K2.5"
]

F1_scores = [
    # PageContext, Baseline-18, Baseline-14
    # order is the same as in `LLMs_backend`
    0.048, 0.048, 0.048,
    0.055, 0.056, 0.056,
    0.055, 0.058, 0.060,
    0.058, 0.062, 0.063,
    0.096, 0.099, 0.099,
    0.152, 0.159, 0.165,
    0.174, 0.183, 0.178,
    0.103, 0.108, 0.109
]


def draw_f1_bar_chart(save_path="results/f1_bar_chart.png"):
    n_llms = len(LLMs_backend)
    n_methods = len(method)

    # reshape to (n_llms, n_methods)
    scores = np.array(F1_scores).reshape(n_llms, n_methods)

    bar_width = 0.25
    x = np.arange(n_llms)
    offsets = np.array([-1, 0, 1]) * bar_width

    colors = ["#4C72B0", "#DD8452", "#55A868"]

    _, ax = plt.subplots(figsize=(12, 5))

    for i, (m, color, offset) in enumerate(zip(method, colors, offsets)):
        ax.bar(x + offset, scores[:, i], width=bar_width, label=m, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(LLMs_backend, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score by LLM Backend and Method")
    ax.legend(title="Method")
    ax.set_ylim(0, max(F1_scores) * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved to {save_path}")


commercial_llms = [
    ("Claude-Haiku-4.5",   "claude-haiku-4-5"),
    ("ChatGPT-5-mini",     "gpt-5-mini"),
    ("DeepSeek-V4-Flash",  "deepseek-v4-flash"),
    ("Kimi-K2.5",          "kimi-k2.5"),
]

method_files = {
    "PageContext":  "results_pagecontext_{slug}.json",
    "Baseline-18":  "results_baseline_{slug}.json",
    "Baseline-14": "results_baseline_14_{slug}.json",
}


def _load_scores(slug, method_key):
    path = method_files[method_key].format(slug=slug)
    with open(path) as f:
        data = json.load(f)
    return [item["score"] for item in data]


def draw_f1_distribution(save_path="results/f1_distribution.png"):
    colors = {"PageContext": "#4C72B0", "Baseline-18": "#DD8452", "Baseline-14": "#55A868"}
    x_grid = np.linspace(0, 1, 500)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=False)
    axes = axes.flatten()

    for ax, (display_name, slug) in zip(axes, commercial_llms):
        for m in method_files:
            scores = np.array(_load_scores(slug, m))
            kde = gaussian_kde(scores, bw_method=0.15)
            ax.plot(x_grid, kde(x_grid), label=m, color=colors[m], linewidth=2)
            ax.fill_between(x_grid, kde(x_grid), alpha=0.15, color=colors[m])

        ax.set_title(display_name, fontsize=12)
        ax.set_xlabel("F1 Score")
        ax.set_ylabel("Density")
        ax.set_xlim(0, 1)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Method", loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=10)

    plt.suptitle("Per-Question F1 Score Distribution (Commercial LLMs)", fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")


def draw_length_distribution(save_path="results/length_distribution.png"):
    """Violin plots of non-empty prediction length per method × commercial LLM (log y-axis).
    Empty predictions are excluded from the violin body; their fraction is annotated above each violin.
    """
    from matplotlib.patches import Patch

    method_prefix = {
        "PageContext":  "pagecontext",
        "Baseline-18":  "baseline",
        "Baseline-14": "baseline_14",
    }
    colors = {"PageContext": "#4C72B0", "Baseline-18": "#DD8452", "Baseline-14": "#55A868"}
    method_order = ["PageContext", "Baseline-18", "Baseline-14"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=False)
    axes = axes.flatten()

    for ax, (display_name, slug) in zip(axes, commercial_llms):
        nonempty_lengths = []
        empty_fracs = []

        for m in method_order:
            path = f"results_{method_prefix[m]}_{slug}.json"
            with open(path) as f:
                records = json.load(f)
            all_lens = [len(str(r["pred"])) for r in records]
            nonempty = [l for l in all_lens if l > 0]
            nonempty_lengths.append(nonempty)
            empty_fracs.append(1 - len(nonempty) / len(all_lens))

        positions = [1, 2, 3]
        parts = ax.violinplot(nonempty_lengths, positions=positions,
                              showmedians=True, showextrema=False)

        for pc, m in zip(parts["bodies"], method_order):
            pc.set_facecolor(colors[m])
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)

        # annotate empty fraction above each violin
        for pos, frac, lengths in zip(positions, empty_fracs, nonempty_lengths):
            if frac > 0.01:
                ax.text(pos, max(lengths) * 1.3, f"{frac*100:.0f}% empty",
                        ha="center", va="bottom", fontsize=8, color="gray")

        # gold median reference
        path = f"results_{method_prefix['Baseline-18']}_{slug}.json"
        with open(path) as f:
            records = json.load(f)
        gold_median = np.median([len(str(r["gold"])) for r in records])
        ax.axhline(gold_median, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"Gold median ({int(gold_median)} chars)")

        ax.set_yscale("log")
        ax.set_xticks(positions)
        ax.set_xticklabels(method_order, fontsize=10)
        ax.set_ylabel("Prediction length (chars, log scale)")
        ax.set_title(display_name, fontsize=12)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8)

    legend_handles = [Patch(facecolor=colors[m], alpha=0.8, label=m) for m in method_order]
    fig.legend(handles=legend_handles, title="Method", loc="upper center",
               ncol=3, bbox_to_anchor=(0.5, 1.01), fontsize=10)

    plt.suptitle("Prediction Length Distribution by Method (Commercial LLMs)\n"
                 "(Non-empty predictions only; % empty annotated per violin)",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    draw_f1_bar_chart()
    draw_f1_distribution()
    draw_length_distribution()