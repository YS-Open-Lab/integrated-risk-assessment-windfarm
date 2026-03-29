# Integrated Risk Assessment of Load Shedding and Civil Infrastructure Failure in Wind Farm Clusters Using Machine Learning and SHAP

A machine learning and SHAP-based framework for integrated risk assessment in wind farm clusters.

---

## Overview

This repository contains the **project code**, **dataset**, **experimental report**, **saved model artifacts**, and **visualization results** for an integrated risk assessment study on wind farm clusters.

The project focuses on the coupled risk of:

- **power-system load shedding**
- **civil infrastructure failure**

To address this problem, this project proposes **MCAF-Net (Multi-Channel Attention Fusion Network)** for binary risk classification, and further uses **SHAP** and **LIME** to interpret model predictions.

This repository includes a complete workflow covering:

- correlation analysis
- model training and comparison
- result visualization
- SHAP / LIME explainability analysis
- report-ready panel figure generation

---

## Highlights

- 1,200 wind farm cluster samples
- 10 multi-source monitoring features
- CRI-based binary risk labels
- Proposed **MCAF-Net**
- Comparison with **10 baseline models**
- SHAP and LIME interpretability analysis
- Complete figures, tables, saved weights, and experiment report

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
