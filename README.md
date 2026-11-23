# Task 04 — Supervised Classification / Regression Modeling & Tuning

**Author:** LATHIKA I (25AD311)  
**RabTech Academy Student Learning Desk**

## Objective
Train and compare at least four distinct supervised classification models, tune hyperparameters using GridSearchCV with stratified cross-validation, evaluate precision, recall, F1, ROC-AUC and confusion matrices, select a champion based on validation ROC-AUC, and serialize the best model.

## Dataset
`data/synthetic_fraud_classification.csv` is generated synthetic practice data. It contains no real financial/customer records. The binary target is artificially generated; metrics are only workflow examples and are not evidence of fraud-detection performance. Replace it with an official academy dataset if supplied and update documentation.

## Models included
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Classifier (SVC)

## Proof
- Notebook: `notebooks/task04_model_comparison_tuning.ipynb`
- Reusable code: `src/train_tune_compare.py`
- Metrics, ROC curve and confusion matrix are generated into `reports/`
- Serialized champion model is saved as `models/champion_model.joblib`

## Run
```bash
pip install -r requirements.txt
python src/train_tune_compare.py
```
The script creates `reports/` and `models/` outputs on execution.

## GitHub
Create a public repository named `RabTech_Task04_Supervised_Modeling`. Upload the extracted project contents, commit, then submit the public repository URL in the Task 04 deliverable field.

## Responsible use
Do not deploy this synthetic model for financial decisions. Real fraud models require authorized data, leakage checks, robust validation, subgroup/error analysis, human review, monitoring, and rollback controls.
