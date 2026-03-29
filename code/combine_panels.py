"""
Combine SHAP + LIME figures into 2 panel figures
Panel 1 (SHAP): (a) Beeswarm  (b) Bar Importance  (c) Dependence  (d) Waterfall
Panel 2 (LIME): (a) High-Risk (b) Low-Risk         (c) Aggregate   (d) LIME vs SHAP

Place this script in the same folder as the 8 PNG files.
Output: panel_shap.png  and  panel_lime.png
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg

# ── helper ──────────────────────────────────────────────────────────────────
def load(fname):
    return mpimg.imread(fname)

LABEL_STYLE = dict(
    fontsize=18, fontweight="bold", color="black",
    ha="left", va="top",
    transform=None,          # will be set per axis
    bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
              edgecolor="black", linewidth=1.2, alpha=0.90)
)

def add_label(ax, letter):
    """Add (a)/(b)/(c)/(d) label in top-left corner of axis."""
    ax.text(0.012, 0.985, f"({letter})",
            transform=ax.transAxes,
            fontsize=18, fontweight="bold", color="black",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="black", linewidth=1.2, alpha=0.92))

# ════════════════════════════════════════════════════════════════════════════
# Panel 1 — SHAP (2 × 2)
# ════════════════════════════════════════════════════════════════════════════
shap_files = [
    "step4_shap_fig1_beeswarm.png",       # (a)
    "step4_shap_fig2_bar_importance.png", # (b)
    "step4_shap_fig3_dependence.png",     # (c)
    "step4_shap_fig4_waterfall.png",      # (d)
]
shap_labels = ["a", "b", "c", "d"]

fig1, axes1 = plt.subplots(2, 2, figsize=(22, 16))
fig1.patch.set_facecolor("white")
fig1.subplots_adjust(hspace=0.06, wspace=0.06)

for ax, fname, letter in zip(axes1.flatten(), shap_files, shap_labels):
    img = load(fname)
    ax.imshow(img)
    ax.axis("off")
    add_label(ax, letter)

fig1.suptitle(
    "SHAP Interpretability Analysis — MCAF-Net\n"
    "Wind Farm Cluster Integrated Risk Assessment",
    fontsize=16, fontweight="bold", color="black", y=1.005
)

plt.tight_layout(pad=0.8)
plt.savefig("panel_shap.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.show()
print("Saved: panel_shap.png")

# ════════════════════════════════════════════════════════════════════════════
# Panel 2 — LIME (2 × 2)
# ════════════════════════════════════════════════════════════════════════════
lime_files = [
    "step4_lime_fig1_highrisk.png",   # (a)
    "step4_lime_fig2_lowrisk.png",    # (b)
    "step4_lime_fig3_aggregate.png",  # (c)
    "step4_lime_fig4_vs_shap.png",    # (d)
]
lime_labels = ["a", "b", "c", "d"]

fig2, axes2 = plt.subplots(2, 2, figsize=(22, 16))
fig2.patch.set_facecolor("white")
fig2.subplots_adjust(hspace=0.06, wspace=0.06)

for ax, fname, letter in zip(axes2.flatten(), lime_files, lime_labels):
    img = load(fname)
    ax.imshow(img)
    ax.axis("off")
    add_label(ax, letter)

fig2.suptitle(
    "LIME Interpretability Analysis — MCAF-Net\n"
    "Wind Farm Cluster Integrated Risk Assessment",
    fontsize=16, fontweight="bold", color="black", y=1.005
)

plt.tight_layout(pad=0.8)
plt.savefig("panel_lime.png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.show()
print("Saved: panel_lime.png")
