# Modeling and Tuning Protocol

1. Define `fraud_flag` as target; exclude `record_id`.
2. Split stratified train/test before preprocessing.
3. Put imputers and One-Hot Encoding in a ColumnTransformer pipeline.
4. Fit four model families: Logistic Regression, Decision Tree, Random Forest, SVC.
5. Use GridSearchCV with StratifiedKFold on training data only; optimize ROC-AUC.
6. Compare cross-validation best score and held-out test metrics.
7. Select champion by validation ROC-AUC; inspect precision/recall and confusion matrix before any operational decision.
8. Save the fitted preprocessing+model pipeline using joblib.
9. Do not tune repeatedly against the test set.

## Metric interpretation
- Precision: fraction of predicted positives that are truly positive in the labeled test sample.
- Recall: fraction of actual positives detected.
- F1: harmonic mean of precision and recall.
- ROC-AUC: ranking ability across thresholds.
- Confusion matrix: true negatives, false positives, false negatives, true positives.
