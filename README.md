

# <div align="center">Integrated Risk Assessment of Load Shedding and Civil Infrastructure Failure in Wind Farm Clusters Using Machine Learning and SHAP</div>

<div align="center">

A machine learning and SHAP-based framework for integrated risk assessment in wind farm clusters.

</div>

---

## Table of Contents

- [Overview](#overview)
- [Highlights](#highlights)
- [Methodology](#methodology)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Environment](#environment)
- [How to Run](#how-to-run)
- [Experimental Results](#experimental-results)
- [Explainability Analysis](#explainability-analysis)
- [Saved Artifacts](#saved-artifacts)
- [Report](#report)
- [Citation](#citation)
- [Notes](#notes)

---

## Overview

This repository contains the **project code**, **dataset**, **experimental report**, **saved model artifacts**, and **visualization results** for an integrated risk assessment study on wind farm clusters.

The project targets the coupled risk of:

- **power-system load shedding**
- **civil infrastructure failure**

To address this problem, the project proposes **MCAF-Net (Multi-Channel Attention Fusion Network)** for binary risk classification, and further introduces **SHAP** and **LIME** for interpretable analysis of prediction behavior. The workflow covers correlation analysis, model comparison, result visualization, and explainability analysis. :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}

---

## Highlights

- **1,200 samples** collected from multi-source monitoring systems
- **10 core risk-related features**
- Binary labels derived from the **Comprehensive Risk Index (CRI)**
- Proposed **MCAF-Net** with dual-stream extraction and SE attention fusion
- Comparison against **5 deep learning baselines** and **5 traditional machine learning baselines**
- Global and local explainability via **SHAP** and **LIME**
- Includes complete figures, tables, saved weights, and experiment report :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}

---

## Methodology

### Framework

The proposed framework consists of the following stages:

1. **Data collection**
2. **Preprocessing**
3. **Feature engineering**
4. **Model training**
5. **Performance evaluation**
6. **Interpretability analysis**
7. **Risk decision support** :contentReference[oaicite:4]{index=4}

### MCAF-Net

The proposed **MCAF-Net** includes three major components:

- **Dual-stream feature extractor**  
  Separates the input into:
  - electrical stream
  - civil engineering stream

- **Cross-channel SE attention fusion module**  
  Learns adaptive feature importance after stream fusion.

- **Deep residual classifier**  
  Improves representation learning and boundary discrimination for binary risk prediction. :contentReference[oaicite:5]{index=5}

### Input Features

The project uses 10 feature variables spanning electrical-system and civil-infrastructure dimensions:

- Total Output
- Line Fault Rate
- Oil Leak Count
- Foundation Settlement
- Road Capacity
- Cable Trench Water
- Voltage Unbalance
- Substation Temperature Rise
- Ground Resistance
- Turbine Availability :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}

---

## Dataset

The dataset is stored in:

```text
data/wind_farm_risk_dataset.csv
## Dataset

It contains **1,200 records** and includes both feature variables and labels. The label is binary:

- `0` = Low Risk
- `1` = High Risk

The data are approximately balanced:

- **622 low-risk samples**
- **578 high-risk samples**

### Data Fields

| Field | Description |
|---|---|
| `风电场群总出力_MW` | Total output of wind farm cluster |
| `集电线路故障率_次百km年` | Line fault rate |
| `箱变渗漏油次数_次月` | Oil leak count |
| `风机基础沉降速率_mm月` | Foundation settlement |
| `场区道路承载力_MPa` | Road capacity |
| `电缆沟积水深度_mm` | Cable trench water depth |
| `集电系统电压不平衡度_%` | Voltage unbalance |
| `升压站设备温升_℃` | Substation temperature rise |
| `防雷接地电阻_Ω` | Ground resistance |
| `风机可利用率_%` | Turbine availability |
| `CRI综合风险指数` | Comprehensive Risk Index |
| `风险等级_二分类` | Binary risk label |

### Label Construction

The binary target is derived from the **Comprehensive Risk Index (CRI)**. The CRI is constructed by weighted fusion of multiple risk-related variables, with the median used as the classification threshold.

---

## Repository Structure

```text
integrated-risk-assessment-windfarm/
├─ README.md
├─ requirements.txt
├─ code/
│  ├─ step1_correlation_heatmap.py
│  ├─ step2_models_training.py
│  ├─ step3_visualizations.py
│  ├─ step4_shap_lime.py
│  ├─ combine_panels.py
│  └─ combine_roc_pr.py
├─ data/
│  └─ wind_farm_risk_dataset.csv
├─ figures/
│  ├─ logo.png
│  ├─ step1_correlation_heatmap.png
│  ├─ step2_metrics_barplot.png
│  ├─ step3_fig1_roc_curves.png
│  ├─ step3_fig2_pr_curves.png
│  ├─ step3_fig3_radar.png
│  ├─ step3_fig4_confusion_matrices.png
│  ├─ step3_fig5_metrics_heatmap.png
│  ├─ step4_shap_fig1_beeswarm.png
│  ├─ step4_shap_fig2_bar_importance.png
│  ├─ step4_shap_fig3_dependence.png
│  ├─ step4_shap_fig4_waterfall.png
│  ├─ step4_lime_fig1_highrisk.png
│  ├─ step4_lime_fig2_lowrisk.png
│  ├─ step4_lime_fig3_aggregate.png
│  ├─ step4_lime_fig4_vs_shap.png
│  ├─ panel_roc_pr.png
│  ├─ panel_shap.png
│  └─ panel_lime.png
├─ tables/
│  └─ step2_metrics_table.csv
├─ models/
│  └─ model_results.pkl
├─ weights/
│  └─ mcafnet_weights.pth
└─ report/
   └─ wind_farm_experiment_report.docx
