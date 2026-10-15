"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> drop duplicate rows -> add engineered features -> add missingness flags
    -> split features/target -> set the final test set aside (locked)
    -> stratified k-fold cross-validation of preprocessing + model on the development set
    -> [if tuning.enabled: nested CV of the tuning procedure, then tuning on all development rows]
    -> out-of-fold classification report + fairness check
    -> refit the final (tuned) model on the whole development set -> save results
"""
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import (
    add_engineered_features,
    clean_dataset,
    drop_duplicate_rows,
    split_dev_test,
    split_features_target,
)
from src.model import build_pipeline
from src.evaluate import (
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))  # before the split
    # week 6: new features from fixed formulas (nothing learned), so safe before the split.
    # The model only uses the ones listed in preprocessing.numeric_features.
    df_clean = add_engineered_features(df_clean)

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
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    # fixed random_state = the same folds for every model, so fold scores compare like-for-like
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # 1. the model with the hyperparameters written in config.yaml (no tuning) -- the "default" row
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs)
    header = f"Hyperparameters from config.yaml (not tuned): {config['model'].get('params')}"
    print(header)
    report = header + "\n" + cv_report(fold_scores, scoring)

    # 2.  hyperparameter tuning -- development set only, every candidate scored by CV of the whole pipeline (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning.enabled is true but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)
        optuna.logging.set_verbosity(optuna.logging.WARNING)   # no line per trial

        # the honest estimate: tuning repeated inside each outer fold (the same folds as above),
        # scored on rows it never saw. Its out-of-fold predictions replace the untuned ones in
        # the classification and fairness reports below.
        print(f"\nNested cross-validation: {cv.get_n_splits()} outer folds x {n_trials} trials x "
              f"{inner_cv.get_n_splits()} inner folds ...")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n\nNested cross-validation (tuned model, honest estimate):\n" + cv_report(nested_scores, scoring)

        # the hyperparameters we keep: the same procedure, once, on all development rows
        print()
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(
        y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # the model we'd actually use: same (tuned) pipeline refit on EVERY development row.
    # CV estimated how well this recipe does; it didn't produce a model.
    pipeline.fit(X_dev, y_dev)
    footer = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    if tuning_enabled:
        footer += f"\nTuned hyperparameters: {study.best_params}"
    footer += f"\nLocked test set: {len(X_test)} rows set aside, not evaluated."
    print(footer)
    report += "\n" + footer + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
