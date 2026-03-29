"""
Step 1: Correlation Heatmap
Wind Farm Cluster Risk Assessment
Data file: data.csv (current directory)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

# ── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data.csv", encoding="utf-8-sig")

# Rename columns to English for all plots
col_map = {
    "风电场群总出力_MW":       "Total Output (MW)",
    "集电线路故障率_次百km年":  "Line Fault Rate",
    "箱变渗漏油次数_次月":      "Oil Leak Count",
    "风机基础沉降速率_mm月":    "Foundation Settlement",
    "场区道路承载力_MPa":       "Road Capacity (MPa)",
    "电缆沟积水深度_mm":        "Cable Trench Water",
    "集电系统电压不平衡度_%":   "Voltage Unbalance (%)",
    "升压站设备温升_℃":         "Substation Temp Rise",
    "防雷接地电阻_Ω":           "Ground Resistance",
    "风机可利用率_%":            "Turbine Availability",
    "CRI综合风险指数":           "CRI Index",
    "风险等级_二分类":           "Risk Label",
}
df.rename(columns=col_map, inplace=True)

# Feature columns (10 features + CRI, excluding label)
feature_cols = [
    "Total Output (MW)", "Line Fault Rate", "Oil Leak Count",
    "Foundation Settlement", "Road Capacity (MPa)", "Cable Trench Water",
    "Voltage Unbalance (%)", "Substation Temp Rise", "Ground Resistance",
    "Turbine Availability", "CRI Index"
]

df_feat = df[feature_cols + ["Risk Label"]]

# ── Pearson Correlation Matrix ──────────────────────────────────────────────
corr = df_feat.corr(method="pearson")

# ── Significance mask (p < 0.05) ───────────────────────────────────────────
n = len(df_feat)
p_matrix = pd.DataFrame(np.ones_like(corr.values), index=corr.index, columns=corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if i != j:
            r = corr.iloc[i, j]
            t = r * np.sqrt((n - 2) / (1 - r**2 + 1e-12))
            p = 2 * (1 - stats.t.cdf(abs(t), df=n-2))
            p_matrix.iloc[i, j] = p

# ── Plot ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor("#0D1117")
ax.set_facecolor("#0D1117")

# Custom diverging palette
cmap = sns.diverging_palette(230, 20, as_cmap=True)

mask = np.zeros_like(corr, dtype=bool)
# No mask — show full matrix

sns.heatmap(
    corr,
    ax=ax,
    cmap=cmap,
    center=0,
    vmin=-1, vmax=1,
    annot=True,
    fmt=".2f",
    annot_kws={"size": 8, "color": "white", "weight": "bold"},
    linewidths=0.5,
    linecolor="#1C2333",
    square=True,
    cbar_kws={"shrink": 0.75, "pad": 0.02}
)

# Significance stars
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        if i != j and p_matrix.iloc[i, j] < 0.05:
            ax.text(j + 0.85, i + 0.18, "*",
                    ha="center", va="center",
                    color="#FFD700", fontsize=10, weight="bold")

# Style colorbar
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_tick_params(color="white")
cbar.outline.set_edgecolor("white")
plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
cbar.set_label("Pearson Correlation Coefficient", color="white", fontsize=10, labelpad=10)

# Axis styling
ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right",
                    fontsize=9, color="white")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0,
                    fontsize=9, color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#2E3A50")

# Title
ax.set_title(
    "Pearson Correlation Heatmap — Wind Farm Cluster Risk Features\n"
    "★ p < 0.05 significance marked with *",
    color="white", fontsize=13, fontweight="bold", pad=18
)

# Highlight Risk Label row/col borders
for tick in ax.get_xticklabels():
    if "Risk" in tick.get_text():
        tick.set_color("#FF6B6B")
        tick.set_fontweight("bold")
for tick in ax.get_yticklabels():
    if "Risk" in tick.get_text():
        tick.set_color("#FF6B6B")
        tick.set_fontweight("bold")

plt.tight_layout()
plt.savefig("step1_correlation_heatmap.png", dpi=200, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.show()
print("Saved: step1_correlation_heatmap.png")
