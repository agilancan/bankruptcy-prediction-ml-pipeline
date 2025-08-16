# Lab 5: Model Implementation and Evaluation

This report summarizes the training, evaluation, and comparison of three models for the **Company Bankruptcy Prediction** dataset. The pipeline includes EDA, preprocessing, feature selection, model training, evaluation, SHAP analysis, and PSI calculation.

---

## Exploratory Data Analysis (EDA)

- Examined distributions and missing values (histograms and heatmap saved).  
- Observed some outliers, but kept them unless extreme (important for risky company behavior).  
- Correlation heatmap used to guide dropping redundant features.  
- Target column is imbalanced (far more non-bankrupt companies).

---

## Data Preprocessing

- Median imputation for missing values.  
- StandardScaler applied (needed for Logistic Regression; kept consistent across all models).  
- Stratified train/test split used to preserve class ratio.  
- Chose class weights instead of SMOTE (banking context → no synthetic companies).

---

## Feature Selection

- Correlation filtering to drop features with correlation > 0.9.  
- Reduced redundancy and multicollinearity.  
- Kept top important features from Random Forest/XGBoost for interpretability.  
- Final features matched Lab 4 decisions.

---

## Hyperparameter Tuning

- RandomizedSearchCV with Stratified K-Fold cross-validation.  
- Balanced speed and accuracy (not exhaustive grid search).  
- Tuned parameters:
  - Logistic Regression: `C` and `penalty`  
  - Random Forest: `max_depth`, `n_estimators`  
  - XGBoost: `max_depth`, `learning_rate`  
- Saved best parameters for each model.

---

## Model Training

- Trained 3 models: Logistic Regression (benchmark), Random Forest, XGBoost.  
- Used class weights (LR/RF) and `scale_pos_weight` (XGBoost) for imbalance.  
- Saved models as `.joblib` files for later use.  
- Random seeds fixed for reproducibility.

---

## Model Evaluation and Comparison

- Metrics: ROC-AUC, Brier Score, F1, Precision, Recall, Accuracy.  
- Plotted ROC curves (train vs test) and calibration curves.  
- Saved all plots under `outputs/evaluation/`.  
- Comparison table saved as `outputs/model_comparison.csv`.

---

## SHAP Values

- Ran SHAP on the best model (highest test ROC-AUC).  
- Saved global SHAP summary plot in `outputs/shap/`.  
- Showed which features most influenced bankruptcy predictions.  
- Useful for interpretability and regulatory review.

---

## Population Stability Index (PSI)

- Calculated PSI between train and test sets for all numeric features.  
- Rule of thumb:
  - `< 0.1`: no shift  
  - `0.1–0.25`: moderate shift  
  - `> 0.25`: significant shift  
- Found a few features with moderate drift → monitor in production.  
- Saved PSI values under `outputs/psi/`.

---

## Results

- **Logistic Regression (baseline)**: worked but lower ROC-AUC.  
- **Random Forest**: stronger performance, handled non-linear patterns.  
- **XGBoost**: best model by test ROC-AUC and calibration.  
- **Best model overall**: XGBoost.

---

## Challenges and Reflections

- Dataset imbalance: solved with class weights and `scale_pos_weight`.  
- Computation time: used RandomizedSearch instead of full Grid Search.  
- Outlier handling: careful not to remove real risky cases.  
- Learned how to combine preprocessing, training, evaluation, and explainability into one pipeline.

---

## Deployment Recommendation

- Recommend **XGBoost** for the best balance of ROC-AUC, calibration, and interpretability.  
- Monitor PSI regularly to check for drift in production.  
- Retrain if PSI > 0.25 or performance degrades.
