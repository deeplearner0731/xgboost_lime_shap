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

### Prerequisites

Please ensure the following dependencies are installed:

- Python 3.7+
- `xgboost`
- `scikit-learn`
- `shap`
- `lime`
- `numpy`
- `pandas`
- `matplotlib`






