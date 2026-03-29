

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
## Environment

Install dependencies with:

```bash
pip install -r requirements.txt
```

A recommended `requirements.txt` is:

```txt
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
scikit-learn>=1.3
torch>=2.0
shap>=0.44
lime>=0.2.0.1
```

The current codebase uses `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy`, `scikit-learn`, `torch`, `shap`, and `lime`.

---

## How to Run

Run the scripts in the following order:

```bash
python code/step1_correlation_heatmap.py
python code/step2_models_training.py
python code/step3_visualizations.py
python code/step4_shap_lime.py
python code/combine_roc_pr.py
python code/combine_panels.py
```

### Workflow Summary

- **Step 1:** Generate correlation heatmap
- **Step 2:** Train models and save metrics, weights, and intermediate outputs
- **Step 3:** Generate model comparison figures
- **Step 4:** Generate SHAP and LIME explainability figures
- **Panel scripts:** Merge figures into paper-ready panel images

---

## Experimental Results

### 1. Feature Correlation Analysis

![Correlation Heatmap](figures/step1_correlation_heatmap.png)

This heatmap presents Pearson correlations among the 10 core features, the CRI index, and the binary risk label. The plot shows that **CRI Index** and **Risk Label** have notable associations with variables such as **Total Output**, **Line Fault Rate**, and several infrastructure-related indicators. This figure is used as an initial feature relationship diagnostic before model training.

### 2. Overall Model Performance Comparison

![Metrics Barplot](figures/step2_metrics_barplot.png)

The bar-chart comparison shows that **MCAF-Net (Proposed)** achieves the best overall performance across all six evaluation metrics: **Accuracy, Precision, Recall, F1, AUC-ROC, and MCC**. This is consistent with the reported test-set results in the paper, where MCAF-Net outperforms all ten baselines.

### 3. ROC and Precision-Recall Curves

![ROC and PR Curves](figures/panel_roc_pr.png)

The combined ROC and Precision-Recall panel highlights the comparative discriminative ability of all 11 models. The proposed MCAF-Net curve consistently dominates the baselines, especially in the **low-false-positive region** and **high-recall region**. This indicates stronger ranking ability and better robustness for high-risk sample identification.

#### Individual ROC Curve

![ROC Curves](figures/step3_fig1_roc_curves.png)

#### Individual Precision-Recall Curve

![PR Curves](figures/step3_fig2_pr_curves.png)

### 4. Radar Chart

![Radar Chart](figures/step3_fig3_radar.png)

The radar chart provides a compact comparison across six evaluation metrics. The proposed model forms the outermost envelope over most axes, visually confirming its balanced superiority rather than improvement on only a single metric.

### 5. Confusion Matrices

![Confusion Matrices](figures/step3_fig4_confusion_matrices.png)

The confusion matrix grid shows prediction behavior for all compared models. The MCAF-Net confusion matrix demonstrates a relatively balanced detection ability for both low-risk and high-risk classes, supporting its stronger overall **F1** and **MCC** performance.

### 6. Metrics Heatmap

![Metrics Heatmap](figures/step3_fig5_metrics_heatmap.png)

The heatmap summarizes all model scores in matrix form and highlights the proposed MCAF-Net row. It clearly shows that MCAF-Net leads across all six metrics, consistent with the ranking logic implemented in the training pipeline.

---

## Explainability Analysis

### 1. SHAP Overview Panel

![SHAP Panel](figures/panel_shap.png)

The SHAP panel combines four complementary views:

- global beeswarm summary
- global feature importance
- dependence plots
- local waterfall explanation

Together, they provide both population-level and sample-level interpretation for the proposed model. The main influential variables include **Total Output**, **Line Fault Rate**, and **Voltage Unbalance**.

#### SHAP Beeswarm Summary

![SHAP Beeswarm](figures/step4_shap_fig1_beeswarm.png)

The SHAP beeswarm plot shows both feature importance ordering and directional impact. **Total Output (MW)** is the most dominant factor, followed by **Line Fault Rate** and **Voltage Unbalance (%)**. The color distribution also shows how different feature magnitudes push predictions toward higher or lower risk.

#### SHAP Global Feature Importance

![SHAP Bar Importance](figures/step4_shap_fig2_bar_importance.png)

This global bar chart quantifies the mean absolute SHAP value of each feature. It confirms that **Total Output** contributes far more strongly than the remaining variables, while **Line Fault Rate**, **Voltage Unbalance**, and **Foundation Settlement** also play important roles.

#### SHAP Dependence Plots

![SHAP Dependence](figures/step4_shap_fig3_dependence.png)

The dependence plots reveal nonlinear relationships between feature magnitude and SHAP contribution. In the current result, **Total Output** and **Line Fault Rate** are selected as the two most important features, and their contribution trends change systematically with feature value.

#### SHAP Waterfall Plot

![SHAP Waterfall](figures/step4_shap_fig4_waterfall.png)

The waterfall plot explains one representative high-risk sample. It shows how individual features cumulatively move the prediction from the base value toward a high-risk probability. In the paper, **Voltage Unbalance** and **Line Fault Rate** are identified as major positive contributors for this type of sample.

### 2. LIME Overview Panel

![LIME Panel](figures/panel_lime.png)

The LIME panel complements SHAP with instance-level local explanations and cross-method comparison. It includes:

- high-risk sample explanation
- low-risk sample explanation
- aggregate feature importance
- LIME vs SHAP comparison

#### LIME High-Risk Sample Explanation

![LIME High Risk](figures/step4_lime_fig1_highrisk.png)

This figure explains a true high-risk prediction at the local level. Positive bars indicate variables that support the high-risk class, while negative bars reduce that tendency. The chart is useful for understanding why a specific sample is flagged as risky.

#### LIME Low-Risk Sample Explanation

![LIME Low Risk](figures/step4_lime_fig2_lowrisk.png)

This figure shows a local explanation for a low-risk sample. Compared with the high-risk example, the contribution pattern is different, illustrating how the same model responds differently under safer operating conditions.

#### LIME Aggregate Feature Importance

![LIME Aggregate](figures/step4_lime_fig3_aggregate.png)

The aggregate LIME figure summarizes mean absolute feature importance across 50 test samples. The resulting ranking is broadly consistent with the SHAP global ranking, which supports the robustness of the interpretability findings.

#### LIME vs SHAP Comparison

![LIME vs SHAP](figures/step4_lime_fig4_vs_shap.png)

This comparison plot normalizes feature importance from SHAP and LIME into the same scale. The close alignment between the two methods supports cross-method consistency in feature attribution. The paper reports strong agreement between SHAP and LIME rankings.

---

## Saved Artifacts

### Saved Weights

```text
weights/mcafnet_weights.pth
```

This file stores trained MCAF-Net weights for downstream inference and explainability.

### Serialized Results

```text
models/model_results.pkl
```

This file stores saved model outputs, including predictions, probabilities, and test-set related artifacts used by later visualization scripts.

### Metrics Table

```text
tables/step2_metrics_table.csv
```

This CSV file records the core metric comparison across all models.

---

## Report

The project report is located at:

```text
report/wind_farm_experiment_report.docx
```

The report documents:

- research background
- data collection and preprocessing
- CRI-based label construction
- statistical description
- machine learning validation
- dataset field definitions

---

## Citation

If you use this repository in academic or technical work, please cite the corresponding paper:

```text
Integrated Risk Assessment of Load Shedding and Civil Infrastructure Failure in Wind Farm Clusters Using Machine Learning and SHAP
```

---

## Notes

- If `figures/logo.png` does not exist yet, delete the top logo line or add your lab logo there.
- This repository contains both intermediate files and report-ready figures.
- The current implementation is organized as a complete experiment pipeline rather than a lightweight demo.
- The project is suitable for academic research, result presentation, and follow-up extension work.


