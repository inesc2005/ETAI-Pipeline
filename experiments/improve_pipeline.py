"""
"can we get a better modelling pipeline?"

Each variant changes ONE thing from the baseline (imputer, encoder, scaler, missing-value
flags, or a model hyperparameter) and is scored with the same stratified k-fold CV (same cross-validation folds) on the
development set, so every comparison is like-for-like (same folds). The locked test set is
never touched.

Run from the project root with:
    python -m experiments.improve_pipeline

A variant only counts as an improvement if its paired difference against the baseline
(fold by fold) is larger than its standard error, otherwise it is noise.
Its just "better" if the gain is clearly higher than the uncertainty, otherwise is just noise. The pipeline chosen for the final test is the best variant, not the baseline

"""

# scores each pipeline variant (one change at a time) with the same CV folds
# and compares it with the baseline, fold by fold.

import os
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.evaluate import cross_validate_pipeline
from src.model import build_pipeline
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)

LR = {"type": "logistic_regression", "params": {"max_iter": 2000}}
DT = {"type": "decision_tree", "params": {"random_state": 42}}
RF = {"type": "random_forest", "params": {"n_estimators": 300, "random_state": 42, "n_jobs": -1}}


def lr(**params):
    return {"type": "logistic_regression", "params": {"max_iter": 2000, **params}}


def dt(**params):
    return {"type": "decision_tree", "params": {"random_state": 42, **params}}


def rf(**params):
    return {"type": "random_forest", "params": {"n_estimators": 300, "random_state": 42, "n_jobs": -1, **params}}


# (name, model_config, preprocessing overrides). the first row is the baseline everything is compared to
VARIANTS = [
    ("baseline: LR, median, target, standard", LR, {}),
    # --- preprocessing changes (logistic regression) ---
    ("imputer: KNN (k=5)", LR, {"numeric_impute_strategy": "knn"}),
    ("imputer: mean", LR, {"numeric_impute_strategy": "mean"}),
    ("encoder: onehot", LR, {"encoder": "onehot"}),
    ("encoder: ordinal", LR, {"encoder": "ordinal"}),
    ("encoder: count", LR, {"encoder": "count"}),
    ("scaler: robust", LR, {"scaler": "robust"}),
    ("no missing flags", LR, {"mnar_columns": []}),
    # --- model changes ---
    ("LR: class_weight=balanced", lr(class_weight="balanced"), {}),
    ("LR: C=0.1 (stronger regularisation)", lr(C=0.1), {}),
    ("DT: default (overfits)", DT, {}),
    ("DT: max_depth=5", dt(max_depth=5), {}),
    ("DT: min_samples_leaf=50", dt(min_samples_leaf=50), {}),
    ("RF: default", RF, {}),
    ("RF: max_depth=6", rf(max_depth=6), {}),
    ("RF: min_samples_leaf=20", rf(min_samples_leaf=20), {}),
    ("RF: max_depth=6, min_samples_leaf=20", rf(max_depth=6, min_samples_leaf=20), {}),
]


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    base_prep = config["preprocessing"]

    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(
        df, target=config["data"]["target"], sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        mnar_columns=base_prep["mnar_columns"],
    )
    X_dev, _X_test, y_dev, _y_test, _e_dev, _e_test = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"], random_state=config["test_set"]["random_state"],
    )

    cv_config = config["cv"]
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=cv_config.get("shuffle", True),
                         random_state=cv_config.get("random_state"))
    scoring = cv_config.get("scoring", "accuracy")

    rows, baseline_folds = [], None
    for name, model_config, overrides in VARIANTS:
        print(f"running {name} ...")
        prep = {**base_prep, **overrides}
        fold_scores, _ = cross_validate_pipeline(build_pipeline(prep, model_config), X_dev, y_dev, cv, scoring)
        val = fold_scores["validation"].to_numpy()
        if baseline_folds is None:
            baseline_folds = val
        diff = val - baseline_folds
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        rows.append({
            "variant": name,
            "CV mean": round(val.mean(), 4),
            "CV std": round(val.std(ddof=1), 4),
            "train": round(fold_scores["train"].mean(), 3),
            "gap": round(fold_scores["gap"].mean(), 3),
            "diff vs baseline": round(diff.mean(), 4),
            "SE of diff": round(se, 4),
        })

    table = pd.DataFrame(rows)
    best = table.iloc[table["CV mean"].idxmax()]
    text = "\n".join([
        f"GOAL 3 -- pipeline variants, {cv_config['n_splits']}-fold stratified CV on the development set ({scoring})",
        "Each variant changes one thing; 'diff vs baseline' is the paired fold-by-fold difference.",
        "An improvement only counts if |diff| is clearly larger than its SE.",
        "",
        table.to_string(index=False),
        "",
        f"Best by CV mean: {best['variant']}  ({best['CV mean']:.4f})",
        "",
    ])
    print("\n" + text)

    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"improve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
