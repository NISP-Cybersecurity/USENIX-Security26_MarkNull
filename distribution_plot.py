import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import fontManager
import seaborn as sns
import os

# ================= Configuration (keep consistent across figures) =================

# High-quality color palette (Nature/Science-like)
COLORS = [
    "#8470C5",  # 0. Classic blue-purple
    "#E698BF",  # 1. Teal/pink-ish
    "#3C5488",  # 2. Deep blue (for Clean)
    "#FF7950",  # 3. Light orange
    "#84919E",  # 4. Gray-blue
    "#BEF529",  # 5. Light cyan
    "#7E6148",  # 6. Deep brown
]

# Color for "Ours"
OURS_COLOR = "#E64B35"   # Red (for Attacked)
OURS_COLOR2 = "#55E9D1FF"  # Alternative accent color (green-ish)

def set_paper_style(font_name="Noto Sans", use_tex=False):
    """Set a publication-ready plotting style."""
    available_fonts = set([f.name for f in fontManager.ttflist])

    # Prefer the requested font; fall back to common serif fonts if unavailable
    if font_name in available_fonts:
        target_font = font_name
    else:
        candidates = ["Liberation Serif", "DejaVu Serif", "Bitstream Vera Serif"]
        target_font = "DejaVu Serif"  # safe fallback
        for c in candidates:
            if c in available_fonts:
                target_font = c
                break
        # print(f"⚠️ Warning: '{font_name}' not found. Using '{target_font}' instead.")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": [target_font],
        "font.size": 14,             # slightly larger for histograms
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "axes.unicode_minus": False,
        "text.usetex": use_tex,
        "lines.linewidth": 1.5,
        "axes.linewidth": 1.0,       # thicker axes spines
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
    })

def plot_distribution(clean_scores, att_scores, threshold, save_path,
                      metric_name="LPIPS"):
    """
    Plot distributions (histogram + KDE) for clean vs. attacked samples
    under a consistent publication style.
    """
    # 1) Apply style
    set_paper_style()

    # 2) Create figure (size suitable for single/double column)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # 3) Colors (consistent with other figures)
    color_clean = COLORS[2]          # deep blue
    color_attacked = OURS_COLOR      # red
    color_threshold = OURS_COLOR2    # accent green for the threshold

    # 4) Histogram + KDE (Seaborn)
    # element="step" gives a cleaner modern look; alpha enables overlap visibility.
    sns.histplot(
        clean_scores,
        color=color_clean,
        label="Clean (Watermarked)",
        kde=True,
        stat="density",
        alpha=0.2,
        element="step",
        linewidth=0,      # remove histogram border for a cleaner look
        ax=ax
    )

    sns.histplot(
        att_scores,
        color=color_attacked,
        label="Attacked (Watermark Removed)",
        kde=True,
        stat="density",
        alpha=0.2,
        element="step",
        linewidth=0,
        ax=ax
    )

    # 5) Thicken KDE lines (histplot adds KDE lines to ax.lines)
    for line in ax.lines:
        line.set_linewidth(2.0)

    # 6) Draw threshold line
    ax.axvline(
        threshold,
        color=color_threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.3f}"
    )

    # 7) Figure cosmetics
    ax.set_title(f"Distribution of Reconstruction Error ({metric_name})")
    ax.set_xlabel(f"{metric_name} Distance")
    ax.set_ylabel("Density")

    # Legend: no frame, standard location
    ax.legend(frameon=False, loc="upper right")

    # Grid: typically only y-grid is cleaner for distributions
    ax.grid(True, axis="y")

    # Remove top/right spines (Nature-like style)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    # Ensure output directory exists
    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Distribution plot saved to {save_path}")


# ================= Demo / sanity check =================
if __name__ == "__main__":
    # Generate synthetic data for demonstration
    np.random.seed(42)
    clean_data = np.random.normal(loc=0.05, scale=0.015, size=500)     # clean LPIPS tends to be low
    attacked_data = np.random.normal(loc=0.15, scale=0.03, size=500)   # attacked error increases

    # Example threshold
    threshold_val = 0.09

    plot_distribution(
        clean_data,
        attacked_data,
        threshold_val,
        save_path="./figs/dist_lpips_demo.png",
        metric_name="LPIPS"
    )
