# Credit Card Default Prediction - Project Guide

## Overview
This project implements a comprehensive machine learning analysis comparing 5 different models for predicting credit card default risk, with a focus on the interpretability-performance tradeoff using SHAP values.

## Project Structure
```
project/
│
├── credit_default_analysis.py    # Main analysis script
├── requirements.txt               # Python dependencies
├── bibliography.md                # 30+ citations for your paper
├── train.csv                      # Training data (70%)
├── test.csv                       # Test data (30%)
├── data_descriptions.csv          # Feature descriptions
│
└── outputs/                       # Generated results
    ├── model_comparison_results.csv
    ├── roc_curves.png
    ├── confusion_matrices.png
    ├── shap_*.png
    └── interpretability_performance_tradeoff.png
```

## Setup Instructions

### 1. Install Dependencies
```bash
# Make sure you're in your conda environment
conda activate new_dl_env

# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Your Data
Make sure you have these files in your working directory:
- `train.csv` - Your 70% training data
- `test.csv` - Your 30% test data
- Both should have the same columns as described in `data_descriptions.csv`

### 3. Run the Analysis

#### Option A: Run Complete Script (Recommended for final results)
```bash
python credit_default_analysis.py
```

This will:
- Load and preprocess your data
- Train all 5 models (Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network)
- Perform hyperparameter tuning
- Generate evaluation metrics
- Create SHAP visualizations
- Analyze interpretability-performance tradeoff
- Save all results to the `outputs/` folder

**Note:** This may take 30-60 minutes depending on your hardware.

#### Option B: Interactive Analysis in Jupyter/VS Code
1. Open the script in VS Code as a notebook (or copy code to .ipynb)
2. Run cells interactively to explore results
3. Modify parameters as needed

## Models Implemented

### 1. Logistic Regression
- **Interpretability:** Very High
- **Use Case:** Baseline model, regulatory compliance
- **Explainability:** Direct coefficient interpretation

### 2. Decision Tree
- **Interpretability:** Very High
- **Use Case:** Visual decision rules
- **Explainability:** Tree structure visualization

### 3. Random Forest
- **Interpretability:** Medium
- **Use Case:** Balanced performance and interpretability
- **Explainability:** Feature importance + SHAP

### 4. XGBoost
- **Interpretability:** Medium
- **Use Case:** High performance ensemble
- **Explainability:** Feature importance + SHAP

### 5. Neural Network
- **Interpretability:** Low
- **Use Case:** Maximum predictive power
- **Explainability:** SHAP values only

## Evaluation Metrics

All models are evaluated on:
- **Accuracy:** Overall correct predictions
- **Precision:** Accuracy of default predictions
- **Recall:** Ability to catch actual defaults
- **F1-Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the ROC curve (primary metric)

## SHAP Analysis

SHAP (SHapley Additive exPlanations) provides:
- **Global Interpretability:** Which features matter most overall
- **Local Interpretability:** Why a specific prediction was made
- **Model Comparison:** Consistency of explanations across models

Each model gets its own SHAP visualization saved to `outputs/`

## Key Research Questions Addressed

1. **Which model performs best?**
   - See `model_comparison_results.csv` for ROC-AUC rankings

2. **What's the cost of interpretability?**
   - See `interpretability_performance_tradeoff.png`
   - Quantifies performance gap between interpretable and black-box models

3. **Can SHAP make complex models explainable?**
   - Compare SHAP visualizations across models
   - Assess consistency of feature importance

## Expected Runtime

- Logistic Regression: ~1 minute
- Decision Tree: ~5 minutes (with GridSearch)
- Random Forest: ~15 minutes (with GridSearch)
- XGBoost: ~10 minutes (with GridSearch)
- Neural Network: ~5 minutes (with early stopping)
- SHAP Analysis: ~10 minutes total

**Total:** ~45-60 minutes on a standard laptop

## Tips for Your Paper

### Results Section
- Use the comparison table from `model_comparison_results.csv`
- Include ROC curves and confusion matrices
- Highlight the best model and the interpretability tradeoff

### Discussion Section
- Discuss which features are most important (from SHAP)
- Compare consistency of explanations across models
- Address whether SHAP makes neural networks "explainable enough"
- Discuss business implications

### Citations
- Use the 34 citations from `bibliography.md`
- Add SHAP paper: Lundberg & Lee (2017)
- Cite specific model papers (XGBoost, Random Forest, etc.)

## Troubleshooting

### Out of Memory Errors
- Reduce GridSearch parameters
- Use smaller validation folds (cv=3 instead of cv=5)
- Sample data for SHAP analysis

### Long Training Times
- Reduce hyperparameter grid size
- Use fewer trees in Random Forest/XGBoost
- Reduce Neural Network epochs

### SHAP Takes Too Long
- Sample background data (already set to 100 samples for NN)
- Reduce test set size for SHAP analysis

## Customization

### Adjust Hyperparameters
Edit the `param_grid` dictionaries in each training function

### Change Metrics
Modify the `evaluate_models()` function to add custom business metrics

### Add Models
Follow the template in `ModelTrainer` class to add new models

## Next Steps

1. Run the complete analysis
2. Review all outputs
3. Identify key findings for your paper
4. Create additional visualizations if needed
5. Write your 3,000-5,000 word paper using the structure provided

## Questions?

If you encounter issues:
1. Check that data files are in the correct format
2. Verify all packages are installed
3. Review error messages for specific issues
4. Adjust parameters if running into resource constraints

Good luck with your final project! 🚀
