from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_language_map(vectors, n_records, pivot_name="Indonesian", branch_map=None, figsize=(11, 9)):
    langs = [l for l in vectors if vectors[l] is not None]
    X = np.array([vectors[l] for l in langs])

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#fafafa")
    ax.set_facecolor("#fafafa")

    default_color = "#6b7280"

    # pivot / origin
    ax.scatter(0, 0, color="black", marker='*', s=420, zorder=5,
               edgecolors="white", linewidth=1.2)
    ax.annotate(f"{pivot_name} (Pivot)", (0, 0), xytext=(8, 8), textcoords="offset points",
                fontsize=12, fontweight="bold")

    for i, lang in enumerate(langs):
        x, y = coords[i]
        color = branch_map.get(lang, default_color) if branch_map else default_color
        n = n_records[lang]

        # panah lembut dari origin
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, alpha=0.35,
                                    lw=1.3, shrinkA=0, shrinkB=8))

        ax.scatter(x, y, color=color, s=180, alpha=0.95,
                   edgecolors="white", linewidth=1.2, zorder=4)

        ax.annotate(f"{lang}", (x, y), xytext=(7, 6), textcoords="offset points",
                    fontsize=10.5, fontweight="semibold", color="#1f2937")
        ax.annotate(f"n={n}", (x, y), xytext=(7, -6), textcoords="offset points",
                    fontsize=8, color="#9ca3af", style="italic")

    ax.axhline(0, color="#d1d5db", linestyle="-", linewidth=1, zorder=1)
    ax.axvline(0, color="#d1d5db", linestyle="-", linewidth=1, zorder=1)

    ax.set_title(f"Language Vector Map — Indonesia Pivot",
                fontsize=16, fontweight="bold", pad=6)
    # ax.text(0.5, 1.02, f"Pivot: {pivot_name}",
    #         transform=ax.transAxes, ha="center", fontsize=11, color="#6b7280")

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% varians)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% varians)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.25)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#d1d5db")

    if branch_map:
        seen = {}
        for lang, color in branch_map.items():
            seen[color] = seen.get(color, lang)  # satu entri legend per warna unik
        handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col,
                              markersize=11, label=name, markeredgecolor="white")
                   for col, name in seen.items()]
        ax.legend(handles=handles, title="Language", loc="best", fontsize=9,
                 frameon=True, framealpha=0.9, edgecolor="#e5e7eb")

    plt.tight_layout()
    plt.show()
    return fig, ax