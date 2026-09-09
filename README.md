# M505 Intro to AI and Machine Learning - Individual Project

**Student ID:** GH1057991
**Module:** M505 Introduction to AI and Machine Learning
**Task:** Predicting online purchase intention from session behaviour

## Overview
An end-to-end scikit-learn machine learning pipeline that predicts whether a browsing
session ends in a purchase. Covers problem definition, exploratory analysis,
preprocessing and feature engineering, model training with hyperparameter tuning,
evaluation on an unseen test set, and a final discussion.

## Files
| File | Description |
|---|---|
| `M505_GH1057991.ipynb` | Main Jupyter Notebook (code + analysis + saved outputs) |
| `M505_GH1057991.html` | Exported HTML version with all charts and outputs |
| `requirements.txt` | Python packages needed to run the notebook |

Repository: https://github.com/sonu2002/M505-AI-ML-GH1057991

## Dataset
Online Shoppers Purchasing Intention Dataset (Sakar et al., 2019), UCI Machine
Learning Repository - 12,330 sessions, 17 features, binary target, 15.5% positive class.

https://archive.ics.uci.edu/ml/machine-learning-databases/00468/online_shoppers_intention.csv

The notebook loads the file directly from this URL, so no manual download is required.

## Models compared
Logistic Regression · Random Forest · HistGradientBoosting · SVM (RBF kernel)

SMOTE is applied inside an imbalanced-learn Pipeline so it is refitted within each
cross-validation fold, preventing synthetic samples from leaking across folds.

## Results
| | |
|---|---|
| Best model | HistGradientBoosting |
| CV F1 | 0.6935 |
| Test F1 | 0.6716 |
| Test AUC | 0.9263 |
| Test accuracy | 0.89 |

## How to run
```bash
pip install -r requirements.txt
jupyter notebook M505_GH1057991.ipynb
```
Run all cells from top to bottom. The notebook also opens in Google Colab.
