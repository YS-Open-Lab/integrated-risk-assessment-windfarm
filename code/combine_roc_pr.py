"""
Combine ROC + PR Curve figures into one panel (a)(b)
Place in same folder as the PNG files.
Output: panel_roc_pr.png
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.patch.set_facecolor("white")
fig.subplots_adjust(wspace=0.04)

files  = ["step3_fig1_roc_curves.png", "step3_fig2_pr_curves.png"]
labels = ["a", "b"]

for ax, fname, letter in zip(axes, files, labels):
    img = mpimg.imread(fname)
    ax.imshow(img)
    ax.axis("off")
    ax.text(0.012, 0.985, f"({letter})",
            transform=ax.transAxes,
            fontsize=20, fontweight="bold", color="black",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                      edgecolor="black", linewidth=1.4, alpha=0.92))

fig.suptitle(
    "Model Comparison — ROC Curves and Precision-Recall Curves\n"
    "Wind Farm Cluster Integrated Risk Assessment",
    fontsize=15, fontweight="bold", color="black", y=1.02
)

plt.tight_layout(pad=0.5)
plt.savefig("panel_roc_pr.png", dpi=200, bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: panel_roc_pr.png")
