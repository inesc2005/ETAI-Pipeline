"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> drop duplicate rows -> add missingness flags
    -> split features/target -> set the final test set aside (locked)
    -> stratified k-fold cross-validation of preprocessing + model on the development set
    -> out-of-fold classification report + fairness check
    -> refit the final model on the whole development set -> save results
"""
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    make_pipeline,
    split_dev_test,
    split_features_target,
)
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))  # before the split

    X, y, extras = split_features_target(
        df_clean,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        mnar_columns=config["preprocessing"]["mnar_columns"],
    )

    # the final test set is set aside HERE and never used again in this script: every decision
    # from now on (preprocessing, model, hyperparameters) is made on the development set only
    X_dev, X_test, y_dev, y_test, extras_dev, _extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # preprocessing lives inside the pipeline, so CV re-fits it on the training part of every fold
    pipeline = make_pipeline(config["model"], config["preprocessing"])

    # fixed random_state = the same folds for every model, so fold scores compare like-for-like
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")

    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_dev, y_dev, cv, scoring, n_jobs=cv_config.get("n_jobs", 1)
    )

    report = cv_report(fold_scores, scoring)
    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # the model we'd actually use: same pipeline refit on EVERY development row.
    # CV estimated how well this recipe does; it didn't produce a model.
    pipeline.fit(X_dev, y_dev)
    footer = (f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows.\n"
              f"Locked test set: {len(X_test)} rows set aside, not evaluated.")
    print(footer)
    report += "\n" + footer + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
