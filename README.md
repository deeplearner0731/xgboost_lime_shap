# Explaining Individualized Treatment Rules: Integrating LIME and SHAP with XGBoost in Precision Medicine

## Overview

**XGBoostML** extends the standard XGBoost framework by incorporating modified loss functions tailored for estimating individualized treatment rules (ITRs). This repository provides a suite of tools to improve the interpretability of ITR models using LIME and SHAP, along with statistical testing procedures to assess treatment effect heterogeneity.

## Key Components

- **XGBoostML_LIME**  
  Incorporates Local Interpretable Model-agnostic Explanations (LIME) into the XGBoostML framework to generate patient-specific explanations of treatment recommendations.

- **XGBoostML_SHAP**  
  Integrates SHapley Additive exPlanations (SHAP) to attribute feature-level contributions to treatment assignments, both locally and globally.

- **Global Permutation Test**  
  Implements a permutation-based hypothesis test to detect global treatment effect heterogeneity under the XGBoostML framework. This helps determine whether interpretability tools should be applied in a given dataset.

- **Doubly Robust (DR) Modified Loss Functions**  
  Provides DR-enhanced versions of the modified loss function for continuous and binary outcomes in ITR estimation.

## Getting Started

### Main Notebooks for SHAP and LIME Interpretability

The following notebooks provide end-to-end implementations for different outcome types:

- `XGBoostML_LIME_SHAP_continuous.ipynb` – Continuous outcomes  
- `XGBoostML_LIME_SHAP_binary.ipynb` – Binary outcomes  
- `XGBoostML_LIME_SHAP_time_to_event.ipynb` – Time-to-event (survival) outcomes  

> **Note:** Each notebook uses a fixed set of hyperparameters for XGBoostML (not necessarily optimal). You are encouraged to tune hyperparameters based on your data and application.

### Prerequisites

Please ensure the following Python packages are installed:

- Python 3.7+
- `xgboost`
- `scikit-learn`
- `shap`
- `lime`
- `numpy`
- `pandas`
- `matplotlib`


