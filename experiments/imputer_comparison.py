"""
Imputer comparison (week 10, challenge A1) -- which numeric imputer should the pipeline keep?

Run from the project root with:
    python -m experiments.imputer_comparison                 # COMPAS (config.yaml)
    python -m experiments.imputer_comparison <config>        # any other config
    python -m experiments.imputer_comparison config.yaml logistic_regression   # another model, same everything else

The whole pipeline of the config (preprocessing + feature selection + model, the hyperparameters
written in the config, no tuning) is cross-validated once per numeric imputer, on the same
development rows and the same folds as main.py. Only `numeric_impute_strategy` changes.
Each imputer is then compared with the median (the one we had) with corrected_paired_test(),
and the time of the 5 folds is recorded, since a better imputer has to be worth its cost.
Saves results/<dir>/imputer_comparison.txt (imputer_comparison_<model>.txt if a model is given).
"""
import os
import sys
import time

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from main import load_config
from src.data import load_data
from src.evaluate import build_scorers, corrected_paired_test, cross_validate_pipeline
from src.model import build_pipeline
from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)

IMPUTERS = ["median", "knn", "iterative", "random_forest"]   # the four of challenge A1; the median is the baseline


def development_data(config: dict):
    """The development rows exactly as main.py builds them (the locked test set is never touched)."""
    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    if config["preprocessing"].get("engineered_features", False):
        df = add_engineered_features(df)
    X, y, extras = split_features_target(
        df, target=config["data"]["target"], sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"], mnar_columns=config["preprocessing"]["mnar_columns"],
    )
    X_dev, _, y_dev, _, _, _ = split_dev_test(X, y, extras, test_size=config["test_set"]["size"],
                                              random_state=config["test_set"]["random_state"])
    return X_dev, y_dev


def main(config_path: str = "config.yaml", model_type: str = None):
    config = load_config(config_path)
    if model_type:   # compare the imputers with another model: its default hyperparameters, no tuning
        config["model"] = {"type": model_type, "params": {"max_iter": 2000} if model_type == "logistic_regression" else {}}
    X_dev, y_dev = development_data(config)
    cv_config = config["cv"]
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=True, random_state=cv_config["random_state"])
    metric = cv_config["selection_metric"]
    scorers = build_scorers([metric])
    numeric = config["preprocessing"]["numeric_features"]
    missing = X_dev[numeric].isna().sum()

    folds, rows = {}, []
    for imputer in IMPUTERS:
        preprocessing = dict(config["preprocessing"], numeric_impute_strategy=imputer)
        pipeline = build_pipeline(preprocessing, config["model"], config.get("feature_selection"))
        start = time.time()
        scores, _ = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, metric, cv_config.get("n_jobs", 1))
        seconds = time.time() - start
        folds[imputer] = scores["validation"].to_numpy()
        rows.append({"imputer": imputer, metric: folds[imputer].mean(), "std": folds[imputer].std(ddof=1),
                     "seconds": seconds})
        print(f"{imputer:<14s} {metric} {folds[imputer].mean():.4f}   ({seconds:.0f} s)", flush=True)

    table = pd.DataFrame(rows)
    ratio = 1 / (cv_config["n_splits"] - 1)
    tests = []
    for imputer in IMPUTERS[1:]:
        test = corrected_paired_test(folds[imputer], folds["median"], ratio)
        tests.append({"imputer": imputer, "mean_diff": test["mean_diff"], "wins": test["wins"], "p_value": test["p_value"]})
    tests = pd.DataFrame(tests)
    per_fold = pd.DataFrame(folds, index=[f"fold_{i}" for i in range(1, cv_config["n_splits"] + 1)]).T

    fmt = lambda v: f"{v:.4f}"
    text = (f"Imputer comparison: {config['model']['type']} ({'default hyperparameters' if model_type else 'hyperparameters from ' + config_path}, no tuning), "
            f"feature selection {'on' if (config.get('feature_selection') or {}).get('enabled') else 'off'}, "
            f"{cv_config['n_splits']}-fold CV on {len(X_dev)} development rows, metric {metric}.\n"
            f"Only numeric_impute_strategy changes. Gaps in the numeric columns (development rows): "
            f"{', '.join(f'{c} {n}' for c, n in missing.items() if n > 0) or 'none'}.\n\n"
            + table.to_string(index=False, formatters={metric: fmt, "std": fmt, "seconds": "{:.0f}".format})
            + "\n\nCorrected paired t-test against the median (A = the imputer, B = median):\n"
            + tests.to_string(index=False, formatters={"mean_diff": lambda v: f"{v:+.4f}", "p_value": "{:.2f}".format})
            + "\n\nScore on each fold:\n" + per_fold.to_string(float_format=fmt) + "\n")
    print("\n" + text)

    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"imputer_comparison_{model_type}.txt" if model_type else "imputer_comparison.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config.yaml", sys.argv[2] if len(sys.argv) > 2 else None)
