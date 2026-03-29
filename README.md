

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
