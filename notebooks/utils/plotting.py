import numpy as np
import matplotlib.pyplot as plt


OBS_LABELS = [
    "vel_x", "vel_y", "pos_x", "pos_y",
    "lm1_x", "lm1_y", "lm2_x", "lm2_y",
    "agent_x", "agent_y"
]

GROUP_LABELS = ["Velocity", "Position", "Landmark", "Other Agent"]

GROUP_SLICES = [
    slice(0, 2),
    slice(2, 4),
    slice(4, 8),
    slice(8, 10),
]


def plot_saliency_heatmap(saliency_results):
    methods = list(saliency_results.keys())

    sal_matrix = np.stack([saliency_results[m] for m in methods])
    sal_normed = sal_matrix / (sal_matrix.max(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(12, max(4, len(methods) * 0.7)))

    im = ax.imshow(sal_normed, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(OBS_LABELS)))
    ax.set_xticklabels(OBS_LABELS, rotation=45, ha="right")

    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods)

    ax.set_title(
        "Saliency Map: Feature Importance by Method",
        fontweight="bold",
        pad=15,
    )

    plt.colorbar(im, ax=ax, label="Normalized |grad|")
    for i in range(len(methods)):
        for j in range(len(OBS_LABELS)):
            val = sal_normed[i, j]
            ax.text(
                j, i, f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if val > 0.7 else "black"
            )

    plt.tight_layout()
    plt.show()


def plot_grouped_importance(saliency_results):
    methods = list(saliency_results.keys())

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(GROUP_LABELS))
    width = 0.8 / len(methods)

    for i, name in enumerate(methods):
        grouped = []

        for s in GROUP_SLICES:
            grouped.append(saliency_results[name][s].mean())

        total = sum(grouped)
        grouped = [(g / total) * 100 for g in grouped]

        ax.bar(
            x + i * width,
            grouped,
            width,
            label=name,
            alpha=0.85,
        )

    ax.set_xticks(x + width * len(methods) / 2)
    ax.set_xticklabels(GROUP_LABELS)

    ax.set_ylabel("Relative Importance (%)")
    ax.set_title("Feature Group Importance by Method", fontweight="bold")

    ax.legend(ncol=min(len(methods), 5), fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()