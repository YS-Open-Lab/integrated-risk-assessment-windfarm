"""
Step 3: Model Visualization — 5 Comparison Plots (White Background / Black Text)
  Fig 1: ROC Curves
  Fig 2: Precision-Recall Curves
  Fig 3: Radar Chart
  Fig 4: Confusion Matrix Grid (all 11 models, 4x3)
  Fig 5: Performance Metrics Heatmap
Requires: model_results.pkl (from Step 2)
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    'font.family': 'DejaVu Sans',
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.edgecolor': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score, confusion_matrix)

# ── Load ────────────────────────────────────────────────────────────────────
with open("model_results.pkl", "rb") as f:
    data = pickle.load(f)

results   = data["results"]
all_proba = data["all_proba"]
all_preds = data["all_preds"]
y_test    = data["y_test"]

model_names   = list(results.keys())
short_names   = [n.replace("\n", " ") for n in model_names]
METRIC_ORDER  = ["Accuracy","Precision","Recall","F1","AUC-ROC","MCC"]

# Colour palette (white-bg friendly)
COLORS = [
    "#D62728",                                           # MCAF-Net — red
    "#1F77B4","#2CA02C","#9467BD","#FF7F0E","#17BECF",   # SOTA DL
    "#7F7F7F","#8C564B","#BCBD22","#E377C2","#AEC7E8",   # Traditional
]

# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — ROC Curves
# ════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(10, 8))
fig1.patch.set_facecolor("white")
ax1.set_facecolor("white")

for name, color in zip(model_names, COLORS):
    fpr, tpr, _ = roc_curve(y_test, all_proba[name])
    roc_auc = auc(fpr, tpr)
    lw = 2.8 if "MCAF" in name else 1.5
    ls = "-"
    ax1.plot(fpr, tpr, color=color, lw=lw, linestyle=ls,
             label=f"{name.replace(chr(10),' ')}  (AUC = {roc_auc:.3f})")

ax1.plot([0,1],[0,1], "k--", lw=1, alpha=0.5, label="Random Classifier")
ax1.set_xlabel("False Positive Rate", fontsize=12)
ax1.set_ylabel("True Positive Rate",  fontsize=12)
ax1.set_title("ROC Curves — All Models Comparison\nWind Farm Cluster Risk Assessment",
              fontsize=13, fontweight="bold")
ax1.legend(loc="lower right", fontsize=8.5, framealpha=0.9,
           edgecolor="black", facecolor="white")
ax1.grid(color="#DDDDDD", linewidth=0.6, linestyle="--")
ax1.set_xlim(0, 1); ax1.set_ylim(0, 1.02)
for spine in ax1.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
fig1.savefig("step3_fig1_roc_curves.png", dpi=200,
             bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step3_fig1_roc_curves.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Precision-Recall Curves
# ════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(10, 8))
fig2.patch.set_facecolor("white")
ax2.set_facecolor("white")

for name, color in zip(model_names, COLORS):
    prec, rec, _ = precision_recall_curve(y_test, all_proba[name])
    ap = average_precision_score(y_test, all_proba[name])
    lw = 2.8 if "MCAF" in name else 1.5
    ax2.plot(rec, prec, color=color, lw=lw,
             label=f"{name.replace(chr(10),' ')}  (AP = {ap:.3f})")

baseline = y_test.mean()
ax2.axhline(baseline, color="black", linestyle=":", lw=1.2, alpha=0.6,
            label=f"Baseline (prevalence = {baseline:.2f})")
ax2.set_xlabel("Recall",    fontsize=12)
ax2.set_ylabel("Precision", fontsize=12)
ax2.set_title("Precision-Recall Curves — All Models Comparison\nWind Farm Cluster Risk Assessment",
              fontsize=13, fontweight="bold")
ax2.legend(loc="upper right", fontsize=8.5, framealpha=0.9,
           edgecolor="black", facecolor="white")
ax2.grid(color="#DDDDDD", linewidth=0.6, linestyle="--")
ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
for spine in ax2.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
fig2.savefig("step3_fig2_pr_curves.png", dpi=200,
             bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step3_fig2_pr_curves.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Radar Chart
# ════════════════════════════════════════════════════════════════════════════
def norm_metric(val, metric):
    return (val + 1) / 2 if metric == "MCC" else val

N = len(METRIC_ORDER)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig3 = plt.figure(figsize=(11, 9))
fig3.patch.set_facecolor("white")
ax3 = fig3.add_subplot(111, polar=True)
ax3.set_facecolor("#F8F8F8")
ax3.spines["polar"].set_color("#AAAAAA")
ax3.grid(color="#CCCCCC", linewidth=0.7, linestyle="--")
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(METRIC_ORDER, color="black", size=11, fontweight="bold")
ax3.set_yticks([0.4, 0.6, 0.8, 1.0])
ax3.set_yticklabels(["0.4","0.6","0.8","1.0"], color="#555555", size=8)
ax3.set_ylim(0.3, 1.0)

for name, color in zip(model_names, COLORS):
    vals = [norm_metric(results[name][m], m) for m in METRIC_ORDER]
    vals += vals[:1]
    lw    = 3.0 if "MCAF" in name else 1.5
    alpha = 0.20 if "MCAF" in name else 0.03
    ax3.plot(angles, vals, color=color, lw=lw, label=name.replace("\n", " "))
    ax3.fill(angles, vals, color=color, alpha=alpha)

ax3.set_title(
    "Model Performance Radar Chart\nWind Farm Cluster Risk Assessment",
    color="black", fontsize=13, fontweight="bold", pad=25)
ax3.legend(loc="upper right", bbox_to_anchor=(1.42, 1.15),
           fontsize=8.5, framealpha=0.9, edgecolor="black",
           facecolor="white", labelcolor="black")
plt.tight_layout()
fig3.savefig("step3_fig3_radar.png", dpi=200,
             bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step3_fig3_radar.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Confusion Matrix Grid (4 rows × 3 cols = 12 cells, 11 models)
# ════════════════════════════════════════════════════════════════════════════
ROWS, COLS = 4, 3
fig4, axes4 = plt.subplots(ROWS, COLS, figsize=(14, 17))
fig4.patch.set_facecolor("white")
fig4.suptitle(
    "Confusion Matrices — All 11 Models\nWind Farm Cluster Risk Assessment",
    color="black", fontsize=14, fontweight="bold", y=0.995)

cmap_cm = plt.cm.Blues

for idx, (name, color) in enumerate(zip(model_names, COLORS)):
    r, c = divmod(idx, COLS)
    ax = axes4[r][c]
    ax.set_facecolor("white")

    cm_val  = confusion_matrix(y_test, all_preds[name])
    cm_norm = cm_val.astype(float) / cm_val.sum(axis=1, keepdims=True)

    ax.imshow(cm_norm, cmap=cmap_cm, vmin=0, vmax=1, aspect="auto")

    for i in range(2):
        for j in range(2):
            txt_color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_val[i,j]}\n({cm_norm[i,j]:.2f})",
                    ha="center", va="center",
                    color=txt_color, fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Low", "Pred: High"], fontsize=8, color="black")
    ax.set_yticklabels(["True: Low", "True: High"], fontsize=8, color="black",
                       rotation=90, va="center")
    ax.set_title(name.replace("\n", " "), color=color,
                 fontsize=10, fontweight="bold", pad=5)
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)

# Hide empty cell
for idx in range(len(model_names), ROWS*COLS):
    r, c = divmod(idx, COLS)
    axes4[r][c].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.98])
fig4.savefig("step3_fig4_confusion_matrices.png", dpi=200,
             bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step3_fig4_confusion_matrices.png")

# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Performance Metrics Heatmap Table
# ════════════════════════════════════════════════════════════════════════════
df_heat = pd.DataFrame(
    {name.replace("\n", " "): {m: results[name][m] for m in METRIC_ORDER}
     for name in model_names}).T

fig5, ax5 = plt.subplots(figsize=(13, 8))
fig5.patch.set_facecolor("white")
ax5.set_facecolor("white")

sns.heatmap(
    df_heat, ax=ax5,
    cmap="RdYlGn",
    annot=True, fmt=".3f",
    annot_kws={"size": 10, "weight": "bold", "color": "black"},
    linewidths=0.5, linecolor="#CCCCCC",
    vmin=0.65, vmax=1.0,
    cbar_kws={"shrink": 0.8}
)

cbar5 = ax5.collections[0].colorbar
plt.setp(cbar5.ax.yaxis.get_ticklabels(), color="black", fontsize=9)
cbar5.set_label("Score", color="black", fontsize=10)

ax5.set_xticklabels(ax5.get_xticklabels(), rotation=0,
                     color="black", fontsize=11, fontweight="bold")
ax5.set_yticklabels(ax5.get_yticklabels(), rotation=0,
                     color="black", fontsize=9)

# Highlight proposed model row with red border
proposed_idx = [i for i, n in enumerate(df_heat.index) if "MCAF" in n]
if proposed_idx:
    ax5.add_patch(plt.Rectangle(
        (0, proposed_idx[0]), len(METRIC_ORDER), 1,
        fill=False, edgecolor="#D62728", lw=3, clip_on=False))

ax5.set_title(
    "Performance Metrics Heatmap — All 11 Models\n"
    "★  MCAF-Net (Proposed) highlighted with red border",
    color="black", fontsize=13, fontweight="bold", pad=12)
for spine in ax5.spines.values(): spine.set_edgecolor("black")

plt.tight_layout()
fig5.savefig("step3_fig5_metrics_heatmap.png", dpi=200,
             bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step3_fig5_metrics_heatmap.png")
print("\nAll Step 3 figures saved.")