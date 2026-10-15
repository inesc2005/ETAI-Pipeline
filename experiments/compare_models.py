"""
Compares the four models (dummy, logistic regression, decision tree, random forest) on the
same data, and writes the comparison tables to results/.

Run from the project root with:
    python -m experiments.compare_models

Table 1 -- week 2 and week 3 pipelines vs. now (week 4 preprocessing inside the Pipeline), holdout.
Table 2 -- holdout vs. cross-validation, same pipeline, same development set.

Everything is computed on the DEVELOPMENT set only (the locked test set is never touched):
the "holdout" is one stratified train/validation split of the development set, and CV is
StratifiedKFold on that same set (settings from config.yaml's `cv` section).
"""
import os
from datetime import datetime

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.data import load_data
from src.evaluate import cross_validate_pipeline
from src.model import build_pipeline
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)

MODELS = {
    "dummy": {"type": "dummy", "params": {}},
    "logistic_regression": {"type": "logistic_regression", "params": {"max_iter": 2000}},
    "decision_tree": {"type": "decision_tree", "params": {"random_state": 42}},
    "random_forest": {"type": "random_forest", "params": {"n_estimators": 300, "random_state": 42, "n_jobs": -1}},
}

HOLDOUT_SIZE = 0.25   # share of the development set held out for validation
N_SEEDS = 10          # holdout is repeated with different seeds to show how much it moves

# Earlier weeks, single 80/20 train/test split, read from the saved runs in results/:
#   week 2 (dropna + one-hot):                    week2_logistic_regression.txt, week2_decision_tree.txt
#   week 3 (impute + target encoder + scaler):    week3_logistic_regression.txt, week3_decision_tree.txt
WEEK2 = {
    "logistic_regression": {"train": 0.679, "test": 0.678},
    "decision_tree": {"train": 0.829, "test": 0.628},
}
WEEK3 = {
    "logistic_regression": {"train": 0.675, "test": 0.657},
    "decision_tree": {"train": 0.792, "test": 0.613},
}

# reads config.yaml into a dict
def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

#one train/validation split (75/25, stratified); returns train and validation accuracy.
def holdout_scores(model_config, preprocessing_config, X, y, seed):
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=HOLDOUT_SIZE, random_state=seed, stratify=y)
    pipe = build_pipeline(preprocessing_config, model_config).fit(X_tr, y_tr)
    return accuracy_score(y_tr, pipe.predict(X_tr)), accuracy_score(y_va, pipe.predict(X_va))


def main():
    config = load_config()
    prep = config["preprocessing"]

    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(
        df, target=config["data"]["target"], sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        mnar_columns=prep["mnar_columns"],
    )
    X_dev, _X_test, y_dev, _y_test, _e_dev, _e_test = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"], random_state=config["test_set"]["random_state"],
    )

    cv_config = config["cv"]
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=cv_config.get("shuffle", True),
                         random_state=cv_config.get("random_state"))

    table1, table2 = [], []
    for name, model_config in MODELS.items():
        print(f"running {name} ...")
        seeds = [holdout_scores(model_config, prep, X_dev, y_dev, s) for s in range(N_SEEDS)]
        h_train, h_val = seeds[0]                       # the single holdout split (seed 0)
        val_all = [v for _, v in seeds]

        fold_scores, _ = cross_validate_pipeline(
            build_pipeline(prep, model_config), X_dev, y_dev, cv, cv_config.get("scoring", "accuracy"),
        )

        w2, w3 = WEEK2.get(name), WEEK3.get(name)
        table1.append({
            "model": name,
            "wk2 test": w2["test"] if w2 else None,
            "wk2 gap": round(w2["train"] - w2["test"], 3) if w2 else None,
            "wk3 test": w3["test"] if w3 else None,
            "wk3 gap": round(w3["train"] - w3["test"], 3) if w3 else None,
            "now train": round(h_train, 3),
            "now valid.": round(h_val, 3),
            "now gap": round(h_train - h_val, 3),
        })
        table2.append({
            "model": name,
            "holdout (1 split)": round(h_val, 3),
            f"holdout range ({N_SEEDS} seeds)": f"{min(val_all):.3f} - {max(val_all):.3f}",
            "CV mean": round(fold_scores["validation"].mean(), 3),
            "CV std": round(fold_scores["validation"].std(ddof=1), 3),
            "CV train": round(fold_scores["train"].mean(), 3),
            "CV gap": round(fold_scores["gap"].mean(), 3),
        })

    text = "\n".join([
        "TABLE 1 -- week 2 and week 3 vs. now (week 4 preprocessing), holdout",
        "(weeks 2-3: 80/20 train/test split; now: 75/25 train/validation split of the development set;",
        " weeks 2-3 only had logistic regression and decision tree)",
        "",
        pd.DataFrame(table1).to_string(index=False),
        "",
        "TABLE 2 -- holdout vs. cross-validation (accuracy, development set)",
        "",
        pd.DataFrame(table2).to_string(index=False),
        "",
    ])
    print("\n" + text)

    # Save the tables to a timestamped file in results/ (a new file on every run).

    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
