import argparse
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.utils.class_weight import compute_class_weight
import shap
import joblib
from psi import calculate_psi

# Define output directories
output_dir = "outputs"
models_dir = os.path.join(output_dir, "models")
eda_dir = os.path.join(output_dir, "eda")
psi_dir = os.path.join(output_dir, "psi")

# Helper function to safely create directories
def make_dir(path):
    os.makedirs(path, exist_ok=True)

# Create all necessary folders
make_dir(output_dir)
make_dir(models_dir)
make_dir(eda_dir)
make_dir(psi_dir)


import re

# Replace any character not allowed in filenames with an underscore
def sanitize_filename(s):
    return re.sub(r'[<>:"/\\|?*]', '_', s)

def run_eda(df, target_col, out_dir):
    make_dir(out_dir)

    # Save describe table
    df.describe().to_csv(os.path.join(out_dir, "describe.csv"))

    # Check Missing values
    df.isna().mean().sort_values(ascending=False).to_csv(os.path.join(out_dir, "missing.csv"))

    # Histograms
    num_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
    for col in num_cols[:10]:  # limit plots to 10 columns
        plt.hist(df[col].dropna(), bins=30)
        plt.title(f"Histogram: {col}")
        safe_col = sanitize_filename(col)
        plt.savefig(os.path.join(out_dir, f"{safe_col}_hist.png"))
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df[num_cols + [target_col]].corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    safe_target = sanitize_filename(target_col)
    plt.savefig(os.path.join(out_dir, f"correlation_heatmap_{safe_target}.png"))
    plt.close()


# Train/test split with stratification
def stratified_split(df, target, test_size, random_state):
    return train_test_split(df, test_size=test_size, stratify=df[target], random_state=random_state)


# Remove highly correlated features
def correlation_filter(df, target_col, thresh=0.9):
    num_cols = [c for c in df.columns if c != target_col and np.issubdtype(df[c].dtype, np.number)]
    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > thresh)]
    keep = [c for c in num_cols if c not in drop_cols]
    return keep


# Impute + scale numeric features
def build_preprocessor(num_cols):
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    pre = ColumnTransformer([
        ("num", pipe, num_cols)
    ])
    return pre


def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


# Evaluate model with metrics and plots
def evaluate_model(name, model, X_train, y_train, X_test, y_test, out_dir):
    make_dir(out_dir)

    # Predictions
    p_train = model.predict_proba(X_train)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    yhat_train = (p_train >= 0.5).astype(int)
    yhat_test = (p_test >= 0.5).astype(int)

    # Metrics
    metrics = {
        "train": {
            "roc_auc": roc_auc_score(y_train, p_train),
            "brier": brier_score_loss(y_train, p_train),
            "f1": f1_score(y_train, yhat_train),
            "precision": precision_score(y_train, yhat_train),
            "recall": recall_score(y_train, yhat_train),
            "accuracy": accuracy_score(y_train, yhat_train),
        },
        "test": {
            "roc_auc": roc_auc_score(y_test, p_test),
            "brier": brier_score_loss(y_test, p_test),
            "f1": f1_score(y_test, yhat_test),
            "precision": precision_score(y_test, yhat_test),
            "recall": recall_score(y_test, yhat_test),
            "accuracy": accuracy_score(y_test, yhat_test),
        },
    }

    # ROC curve
    fpr_tr, tpr_tr, _ = roc_curve(y_train, p_train)
    fpr_te, tpr_te, _ = roc_curve(y_test, p_test)
    plt.plot(fpr_tr, tpr_tr, label=f"Train AUC={metrics['train']['roc_auc']:.3f}")
    plt.plot(fpr_te, tpr_te, label=f"Test AUC={metrics['test']['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{name}_roc.png"))
    plt.close()

    # Calibration curve
    prob_true_tr, prob_pred_tr = calibration_curve(y_train, p_train, n_bins=10)
    prob_true_te, prob_pred_te = calibration_curve(y_test, p_test, n_bins=10)
    plt.plot(prob_pred_tr, prob_true_tr, marker="o", label=f"Train")
    plt.plot(prob_pred_te, prob_true_te, marker="o", label=f"Test")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(f"Calibration Curve - {name}")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{name}_calibration.png"))
    plt.close()

    # Save metrics
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# SHAP summary plot
def run_shap(name, model, X_train, out_dir, is_tree=False):
    make_dir(out_dir)
    if is_tree:
        explainer = shap.TreeExplainer(model.named_steps["clf"])
    else:
        explainer = shap.Explainer(model.named_steps["clf"], X_train)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, features=X_train, show=False)
    plt.title(f"SHAP Summary - {name}")
    plt.savefig(os.path.join(out_dir, f"{name}_shap.png"))
    plt.close()


# Running Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_csv", type=str, default="data.csv", help="CSV file (default: data.csv in root)")
    parser.add_argument("--target_col", type=str, required=True, help="Name of target column")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    out_root = "outputs"
    eda_dir = os.path.join(out_root, "eda")
    psi_dir = os.path.join(out_root, "psi")
    models_dir = os.path.join(out_root, "models")
    eval_dir = os.path.join(out_root, "evaluation")
    shap_dir = os.path.join(out_root, "shap")

    make_dir(out_root)

    # Load data
    df = pd.read_csv(args.data_csv)

    # Run EDA
    run_eda(df, args.target_col, eda_dir)

    # Train/test split
    train_df, test_df = stratified_split(df, args.target_col, args.test_size, args.random_state)

    def sanitize_filename(s):
        s = s.strip()  # remove leading/trailing spaces
        return re.sub(r'[<>:"/\\|?*]', '_', s)

    # PSI check
    make_dir(psi_dir)
    for col in train_df.columns:
        if col != args.target_col:
            psi_value = float(calculate_psi(train_df[col].values, test_df[col].values))
            safe_col = sanitize_filename(col)
            with open(os.path.join(psi_dir, f"{safe_col}_psi.txt"), "w") as f:
                f.write(str(psi_value))



    # Correlation filter
    kept_cols = correlation_filter(train_df, args.target_col, 0.9)

    X_train = train_df[kept_cols]
    y_train = train_df[args.target_col]
    X_test = test_df[kept_cols]
    y_test = test_df[args.target_col]

    # Preprocessor
    pre = build_preprocessor(kept_cols)

    # Models
    cw = get_class_weights(y_train)
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "LogReg": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(solver="liblinear", class_weight=cw))
        ]),
        "RandomForest": Pipeline([
            ("pre", pre),
            ("clf", RandomForestClassifier(class_weight=cw, n_estimators=200, random_state=args.random_state))
        ]),
        "XGBoost": Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=spw,
                n_estimators=300,
                random_state=args.random_state
            ))
        ])
    }

    results = []
    best_model = None
    best_auc = 0

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        joblib.dump(model, os.path.join(models_dir, f"{name}.joblib"))

        metrics = evaluate_model(name, model, X_train, y_train, X_test, y_test, eval_dir)
        results.append({"model": name, **metrics["test"]})

        if metrics["test"]["roc_auc"] > best_auc:
            best_auc = metrics["test"]["roc_auc"]
            best_model = (name, model)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_root, "model_comparison.csv"), index=False)

    # SHAP on best model
    best_name, best_model = best_model
    run_shap(best_name, best_model, pre.fit_transform(X_train), shap_dir, is_tree=("Forest" in best_name or "XGB" in best_name))

    print("Pipeline finished. Best model:", best_name)


if __name__ == "__main__":
    main()
