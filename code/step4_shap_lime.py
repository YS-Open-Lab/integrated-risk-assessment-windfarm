"""
Step 4: SHAP + LIME Analysis — MCAF-Net (Best Model)
White Background / Black Text
Requires: model_results.pkl + mcafnet_weights.pth (from Step 2)

Install:  pip install shap lime torch scikit-learn matplotlib seaborn
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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

import shap
from lime import lime_tabular
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

FEATURE_NAMES = [
    "Total Output (MW)",    "Line Fault Rate",       "Oil Leak Count",
    "Foundation Settlement","Road Capacity (MPa)",   "Cable Trench Water",
    "Voltage Unbalance (%)","Substation Temp Rise",  "Ground Resistance",
    "Turbine Availability",
]

# ── Load saved data ──────────────────────────────────────────────────────────
with open("model_results.pkl", "rb") as f:
    saved = pickle.load(f)
X_train = saved["X_train"]
X_test  = saved["X_test"]
y_train = saved["y_train"]
y_test  = saved["y_test"]

# ── Re-define MCAF-Net (identical to Step 2) ────────────────────────────────
class SEBlock(nn.Module):
    def __init__(self, ch, r=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch, max(ch//r, 8)), nn.GELU(),
            nn.Linear(max(ch//r, 8), ch), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x)

class ResBlock(nn.Module):
    def __init__(self, d, dropout=0.15):
        super().__init__()
        self.blk = nn.Sequential(
            nn.Linear(d, d*2), nn.LayerNorm(d*2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d*2, d), nn.LayerNorm(d))
        self.act = nn.GELU()
    def forward(self, x): return self.act(self.blk(x) + x)

class MCAFNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.elec = nn.Sequential(
            nn.Linear(6, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU())
        self.civil = nn.Sequential(
            nn.Linear(4, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 64), nn.LayerNorm(64), nn.GELU())
        self.se  = SEBlock(128, r=8)
        self.clf = nn.Sequential(
            nn.Linear(128,128), nn.LayerNorm(128), nn.GELU(),
            ResBlock(128, 0.10), ResBlock(128, 0.10), ResBlock(128, 0.10),
            nn.Dropout(0.15),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))
    def forward(self, x):
        e = self.elec(x[:, [0,1,6,7,8,9]])
        c = self.civil(x[:, [2,3,4,5]])
        return self.clf(self.se(torch.cat([e, c], dim=1)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Try loading saved weights; if missing, retrain
model = MCAFNet().to(device)
if os.path.exists("mcafnet_weights.pth"):
    model.load_state_dict(torch.load("mcafnet_weights.pth", map_location=device))
    print("Loaded saved MCAF-Net weights.")
else:
    print("Weights not found — retraining MCAF-Net ...")
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=32, shuffle=True)
    opt   = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-5)
    sched = optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, epochs=200,
                                           steps_per_epoch=len(loader), pct_start=0.3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    model.train()
    for _ in range(200):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    print("Retraining done.")

model.eval()

def predict_proba(X_np):
    t = torch.tensor(X_np.astype(np.float32)).to(device)
    with torch.no_grad():
        return torch.softmax(model(t), 1).cpu().numpy()

# ════════════════════════════════════════════════════════════════════════════
# ───────────────────────── SHAP ANALYSIS ────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
background   = shap.sample(X_train, 100, random_state=42)
X_test_shap  = X_test[:200]
y_test_shap  = y_test[:200]

explainer   = shap.KernelExplainer(lambda x: predict_proba(x)[:, 1], background)
print("Computing SHAP values (~1–2 min) ...")
shap_values = explainer.shap_values(X_test_shap, nsamples=200)
print("SHAP done.\n")

mean_abs_shap = np.abs(shap_values).mean(axis=0)

# ── SHAP Figure 1: Beeswarm (Summary Plot) ──────────────────────────────────
print("SHAP Fig 1: Beeswarm ...")
plt.style.use("default")
fig_s1, ax_s1 = plt.subplots(figsize=(10, 7))
fig_s1.patch.set_facecolor("white")
shap.summary_plot(shap_values, X_test_shap,
                  feature_names=FEATURE_NAMES,
                  plot_type="dot", show=False, plot_size=None)
plt.gcf().patch.set_facecolor("white")
plt.gca().set_facecolor("white")
plt.title("SHAP Beeswarm Summary — MCAF-Net\nFeature Impact on High-Risk Prediction",
          color="black", fontsize=12, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig("step4_shap_fig1_beeswarm.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_shap_fig1_beeswarm.png\n")

# ── SHAP Figure 2: Global Bar Importance ────────────────────────────────────
print("SHAP Fig 2: Bar Importance ...")
sorted_idx = np.argsort(mean_abs_shap)

fig_s2, ax_s2 = plt.subplots(figsize=(10, 7))
fig_s2.patch.set_facecolor("white")
ax_s2.set_facecolor("white")

palette = plt.cm.RdYlGn(np.linspace(0.25, 0.85, len(FEATURE_NAMES)))
bars_s2 = ax_s2.barh([FEATURE_NAMES[i] for i in sorted_idx],
                      mean_abs_shap[sorted_idx],
                      color=palette, edgecolor="black", linewidth=0.5)
for b in bars_s2[-3:]:
    b.set_edgecolor("#D62728"); b.set_linewidth(2)

for bar, val in zip(bars_s2, mean_abs_shap[sorted_idx]):
    ax_s2.text(val + 0.0005, bar.get_y() + bar.get_height()/2,
               f"{val:.4f}", va="center", color="black", fontsize=9)

ax_s2.set_xlabel("Mean |SHAP Value|", fontsize=11)
ax_s2.set_title("SHAP Global Feature Importance — MCAF-Net\n"
                "Top-3 features highlighted with red border",
                color="black", fontsize=12, fontweight="bold")
ax_s2.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
for spine in ax_s2.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
plt.savefig("step4_shap_fig2_bar_importance.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_shap_fig2_bar_importance.png\n")

# ── SHAP Figure 3: Dependence Plots (top 2 features) ───────────────────────
print("SHAP Fig 3: Dependence Plots ...")
top2 = np.argsort(mean_abs_shap)[-2:][::-1]

fig_s3, axes_s3 = plt.subplots(1, 2, figsize=(13, 6))
fig_s3.patch.set_facecolor("white")

for ax_d, fi in zip(axes_s3, top2):
    ax_d.set_facecolor("white")
    fvals = X_test_shap[:, fi]
    svals = shap_values[:, fi]
    sc = ax_d.scatter(fvals, svals, c=fvals, cmap="coolwarm",
                      alpha=0.7, s=22, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax_d, pad=0.02)
    cbar.set_label(FEATURE_NAMES[fi], color="black", fontsize=9)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="black", fontsize=8)
    ax_d.axhline(0, color="black", linestyle="--", lw=0.8, alpha=0.5)
    ax_d.set_xlabel(FEATURE_NAMES[fi], fontsize=11)
    ax_d.set_ylabel("SHAP Value",       fontsize=11)
    ax_d.set_title(f"SHAP Dependence — {FEATURE_NAMES[fi]}",
                   fontsize=11, fontweight="bold")
    ax_d.grid(color="#DDDDDD", linewidth=0.6, linestyle="--", alpha=0.7)
    for spine in ax_d.spines.values(): spine.set_edgecolor("black")

fig_s3.suptitle("SHAP Dependence Plots — Top 2 Most Important Features",
                fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("step4_shap_fig3_dependence.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_shap_fig3_dependence.png\n")

# ── SHAP Figure 4: Waterfall (single high-risk sample) ─────────────────────
print("SHAP Fig 4: Waterfall ...")
hr_idx = np.where(y_test_shap == 1)[0]
si = hr_idx[0] if len(hr_idx) > 0 else 0

sv   = shap_values[si]
base = float(explainer.expected_value)
sord = np.argsort(np.abs(sv))[::-1]

starts, widths, clrs, labs = [], [], [], []
running = base
for fi in reversed(sord):
    starts.append(running)
    widths.append(sv[fi])
    running += sv[fi]
    clrs.append("#D62728" if sv[fi] > 0 else "#2CA02C")
    labs.append(f"{FEATURE_NAMES[fi]}\n= {X_test_shap[si, fi]:.2f}")

fig_s4, ax_s4 = plt.subplots(figsize=(11, 7))
fig_s4.patch.set_facecolor("white")
ax_s4.set_facecolor("white")

ax_s4.barh(range(len(sord)), widths, left=starts, color=clrs,
           edgecolor="black", linewidth=0.4, height=0.6)
for i, (s, w, lab) in enumerate(zip(starts, widths, labs)):
    ax_s4.text(s+w+(0.003 if w>=0 else -0.003), i, f"{w:+.4f}",
               va="center", ha="left" if w>=0 else "right",
               color="black", fontsize=8)
ax_s4.set_yticks(range(len(sord)))
ax_s4.set_yticklabels(labs, fontsize=8)
ax_s4.axvline(base,    color="#1F77B4", linestyle="--", lw=1.5,
              label=f"Base value: {base:.3f}")
ax_s4.axvline(running, color="#D62728", linestyle="--", lw=1.8,
              label=f"Prediction: {running:.3f}")
ax_s4.set_xlabel("SHAP Value (contribution to High-Risk probability)", fontsize=10)
ax_s4.set_title(f"SHAP Waterfall — Sample #{si}  (True Label: High Risk)\n"
                "Red bars = increase risk,  Green bars = decrease risk",
                fontsize=12, fontweight="bold")
ax_s4.legend(fontsize=9, framealpha=0.9, edgecolor="black")
ax_s4.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
for spine in ax_s4.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
plt.savefig("step4_shap_fig4_waterfall.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_shap_fig4_waterfall.png\n")

# ════════════════════════════════════════════════════════════════════════════
# ───────────────────────── LIME ANALYSIS ────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
lime_exp = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=FEATURE_NAMES,
    class_names=["Low Risk","High Risk"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

hr_lime = np.where(y_test == 1)[0]
lr_lime = np.where(y_test == 0)[0]

# ── LIME Figure 1: High-Risk single sample ──────────────────────────────────
print("LIME Fig 1: High-Risk single sample ...")
hi = hr_lime[0]
exp1 = lime_exp.explain_instance(X_test[hi], predict_proba,
                                  num_features=10, num_samples=1000, top_labels=1)
lime_list1 = exp1.as_list(label=1)
lf1 = [x[0] for x in lime_list1]
lv1 = [x[1] for x in lime_list1]
lc1 = ["#D62728" if v>0 else "#2CA02C" for v in lv1]
s1  = sorted(zip(lv1,lf1,lc1), key=lambda x: abs(x[0]))

fig_l1, ax_l1 = plt.subplots(figsize=(11,7))
fig_l1.patch.set_facecolor("white"); ax_l1.set_facecolor("white")
b1 = ax_l1.barh(range(len(s1)), [x[0] for x in s1],
                 color=[x[2] for x in s1], edgecolor="black", linewidth=0.4)
ax_l1.set_yticks(range(len(s1)))
ax_l1.set_yticklabels([x[1] for x in s1], fontsize=8)
ax_l1.axvline(0, color="black", lw=0.8, linestyle="--")
for bar, val in zip(b1, [x[0] for x in s1]):
    ax_l1.text(val+(0.002 if val>=0 else -0.002), bar.get_y()+bar.get_height()/2,
               f"{val:+.4f}", va="center", ha="left" if val>=0 else "right",
               color="black", fontsize=8)
ax_l1.set_xlabel("LIME Feature Weight (contribution to High-Risk class)", fontsize=10)
ax_l1.set_title(f"LIME Explanation — Sample #{hi}  (True: High Risk)\n"
                "Red = supports High Risk,  Green = supports Low Risk",
                fontsize=12, fontweight="bold")
ax_l1.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
for spine in ax_l1.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
plt.savefig("step4_lime_fig1_highrisk.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_lime_fig1_highrisk.png\n")

# ── LIME Figure 2: Low-Risk single sample ───────────────────────────────────
print("LIME Fig 2: Low-Risk single sample ...")
li = lr_lime[0]
exp2 = lime_exp.explain_instance(X_test[li], predict_proba,
                                  num_features=10, num_samples=1000, top_labels=2)
# Auto-detect available label (low-risk samples may not have label=1)
_label2 = 1 if 1 in exp2.as_map() else list(exp2.as_map().keys())[0]
lime_list2 = exp2.as_list(label=_label2)
lf2 = [x[0] for x in lime_list2]
lv2 = [x[1] for x in lime_list2]
lc2 = ["#D62728" if v>0 else "#2CA02C" for v in lv2]
s2  = sorted(zip(lv2,lf2,lc2), key=lambda x: abs(x[0]))

fig_l2, ax_l2 = plt.subplots(figsize=(11,7))
fig_l2.patch.set_facecolor("white"); ax_l2.set_facecolor("white")
b2 = ax_l2.barh(range(len(s2)), [x[0] for x in s2],
                 color=[x[2] for x in s2], edgecolor="black", linewidth=0.4)
ax_l2.set_yticks(range(len(s2)))
ax_l2.set_yticklabels([x[1] for x in s2], fontsize=8)
ax_l2.axvline(0, color="black", lw=0.8, linestyle="--")
for bar, val in zip(b2, [x[0] for x in s2]):
    ax_l2.text(val+(0.002 if val>=0 else -0.002), bar.get_y()+bar.get_height()/2,
               f"{val:+.4f}", va="center", ha="left" if val>=0 else "right",
               color="black", fontsize=8)
ax_l2.set_xlabel("LIME Feature Weight (contribution to High-Risk class)", fontsize=10)
ax_l2.set_title(f"LIME Explanation — Sample #{li}  (True: Low Risk)\n"
                "Red = supports High Risk,  Green = supports Low Risk",
                fontsize=12, fontweight="bold")
ax_l2.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
for spine in ax_l2.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
plt.savefig("step4_lime_fig2_lowrisk.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_lime_fig2_lowrisk.png\n")

# ── LIME Figure 3: Aggregate Importance (50 samples) ────────────────────────
print("LIME Fig 3: Aggregate importance (50 samples) ...")
N_LIME = 50
lime_mat = np.zeros((N_LIME, 10))
for i in range(N_LIME):
    ei = lime_exp.explain_instance(X_test[i], predict_proba,
                                    num_features=10, num_samples=500, top_labels=2)
    _lbl = 1 if 1 in ei.as_map() else list(ei.as_map().keys())[0]
    for fidx, wt in ei.as_map()[_lbl]:
        if fidx < 10:
            lime_mat[i, fidx] = wt

mean_lime = np.abs(lime_mat).mean(axis=0)
std_lime  = np.abs(lime_mat).std(axis=0)
sid3 = np.argsort(mean_lime)

fig_l3, ax_l3 = plt.subplots(figsize=(11,7))
fig_l3.patch.set_facecolor("white"); ax_l3.set_facecolor("white")
pal3 = plt.cm.plasma(np.linspace(0.25, 0.85, 10))
b3 = ax_l3.barh([FEATURE_NAMES[i] for i in sid3], mean_lime[sid3],
                 xerr=std_lime[sid3], color=pal3,
                 edgecolor="black", linewidth=0.5,
                 error_kw={"ecolor":"black","capsize":4,"lw":1.2})
for bar, val in zip(b3, mean_lime[sid3]):
    ax_l3.text(val+0.0005, bar.get_y()+bar.get_height()/2,
               f"{val:.4f}", va="center", color="black", fontsize=8)
ax_l3.set_xlabel("Mean |LIME Weight| ± Std Dev  (N=50 test samples)", fontsize=11)
ax_l3.set_title("LIME Aggregate Feature Importance — MCAF-Net\n"
                "Averaged over 50 Test Samples",
                fontsize=12, fontweight="bold")
ax_l3.grid(axis="x", color="#DDDDDD", linewidth=0.6, linestyle="--")
for spine in ax_l3.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
plt.savefig("step4_lime_fig3_aggregate.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_lime_fig3_aggregate.png\n")

# ── LIME Figure 4: LIME vs SHAP Comparison Bar Chart ────────────────────────
print("LIME Fig 4: LIME vs SHAP comparison ...")
shap_norm = mean_abs_shap / (mean_abs_shap.max() + 1e-12)
lime_norm = mean_lime     / (mean_lime.max()     + 1e-12)

x_pos = np.arange(len(FEATURE_NAMES))
w = 0.38

fig_l4, ax_l4 = plt.subplots(figsize=(13,7))
fig_l4.patch.set_facecolor("white"); ax_l4.set_facecolor("white")
ax_l4.bar(x_pos - w/2, shap_norm, w, label="SHAP (normalized)",
           color="#1F77B4", edgecolor="black", linewidth=0.5, alpha=0.85)
ax_l4.bar(x_pos + w/2, lime_norm, w, label="LIME (normalized)",
           color="#FF7F0E", edgecolor="black", linewidth=0.5, alpha=0.85)
ax_l4.set_xticks(x_pos)
ax_l4.set_xticklabels(FEATURE_NAMES, rotation=30, ha="right", fontsize=9)
ax_l4.set_ylabel("Normalized Feature Importance", fontsize=11)
ax_l4.set_title("LIME vs SHAP Feature Importance Comparison — MCAF-Net\n"
                "Both normalized to [0, 1] for cross-method validation",
                fontsize=12, fontweight="bold")
ax_l4.legend(fontsize=10, framealpha=0.9, edgecolor="black")
ax_l4.grid(axis="y", color="#DDDDDD", linewidth=0.6, linestyle="--")
for spine in ax_l4.spines.values(): spine.set_edgecolor("black")
plt.tight_layout()
plt.savefig("step4_lime_fig4_vs_shap.png", dpi=200,
            bbox_inches="tight", facecolor="white")
plt.show()
print("Saved: step4_lime_fig4_vs_shap.png\n")

print("="*60)
print("All Step 4 figures saved!")
print("""
SHAP: step4_shap_fig1_beeswarm.png
      step4_shap_fig2_bar_importance.png
      step4_shap_fig3_dependence.png
      step4_shap_fig4_waterfall.png
LIME: step4_lime_fig1_highrisk.png
      step4_lime_fig2_lowrisk.png
      step4_lime_fig3_aggregate.png
      step4_lime_fig4_vs_shap.png
""")