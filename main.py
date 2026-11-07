"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> drop duplicate rows -> add engineered features -> add missingness flags
    -> split features/target -> set the final test set aside (locked)
    -> stratified k-fold cross-validation of preprocessing + feature selection (if enabled) + model on the development set,
       on several metrics, plus the no-model floor (a dummy) on the same folds
    -> [if tuning.enabled: nested CV of the tuning procedure, then tuning on all development rows]
    -> out-of-fold confusion matrix, classification report + fairness check
    -> refit the final (tuned) model on the whole development set
    -> one row on the scoreboard (results/runs.csv) + the full report in results/
"""
import shutil
import tempfile
from datetime import datetime

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
    build_scorers,
    cross_validate_pipeline,
    cv_report,
    fairness_report,
    oof_classification_report,
)
from src.tuning import nested_cross_validate, tune_pipeline, tuning_report
from src.results import save_run, save_scores


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()
    run = datetime.now().strftime("%Y%m%d_%H%M%S")    # one id for this run: on the scoreboard and in the report's name
    model_type = config["model"]["type"]

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

    # fixed random_state = the same folds for every model, so fold scores compare like-for-like
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    # week 7: several metrics are reported, ONE (the selection metric) is optimised and ranks the runs
    metrics = cv_config["metrics"]
    selection_metric = cv_config["selection_metric"]
    scorers = build_scorers(metrics)
    n_jobs = cv_config.get("n_jobs", 1)
    # week 8: feature selection is a learned step of the pipeline, switched on in config.yaml
    selection_config = config.get("feature_selection") or {}
    selection_enabled = selection_config.get("enabled", False)
    print(f"Model: {model_type}   |   selection metric: {selection_metric}   |   {cv_config['n_splits']}-fold CV   |   "
          f"feature selection: {'min_votes ' + str(selection_config.get('min_votes', 2)) if selection_enabled else 'off'}")
    print(f"Data:  {len(X_dev)} development rows   |   {len(X_test)} test rows locked away (not used)")

    # preprocessing lives inside the pipeline, so CV re-fits it on the training part of every fold
    # the cache folder stores the fitted prep and select steps, so tuning doesn't refit them in every trial
    cache_dir = tempfile.mkdtemp(prefix="pipeline_cache_")   # a new folder for every run, deleted at the end
    pipeline = build_pipeline(config["preprocessing"], config["model"], selection_config, memory=cache_dir)

    # the no-model floor: a dummy on the same folds -- what any real model has to beat
    floor_pipeline = build_pipeline(config["preprocessing"], {"type": "dummy"})
    floor_scores, _ = cross_validate_pipeline(floor_pipeline, X_dev, y_dev, cv,
                                              {selection_metric: scorers[selection_metric]}, selection_metric, n_jobs)
    floor = floor_scores["validation"].mean()

    # the model with the hyperparameters written in config.yaml (no tuning) -- the "untuned" score
    fold_scores, oof_proba = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scorers, selection_metric, n_jobs)
    untuned = fold_scores["validation"].mean()
    print("\nDefault hyperparameters (config.yaml)")
    print(f"  {selection_metric}: {untuned:.3f} ± {fold_scores['validation'].std(ddof=1):.3f}   "
          f"train-validation gap: {fold_scores['gap'].mean():+.3f}   no-model floor: {floor:.3f}")
    report = (f"Hyperparameters from config.yaml (not tuned): {config['model'].get('params')}\n\n"
              + cv_report(fold_scores, metrics, selection_metric, "Cross-validation")
              + f"\n\nNo-model floor (dummy, same folds): {selection_metric} = {floor:.3f}")
    final_scores = fold_scores   # replaced by the nested CV scores when tuning is on

    #  hyperparameter tuning -- development set only, every candidate scored by CV of the whole pipeline on the selection metric (see src/tuning.py)
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning.enabled is true but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = dict(search_spaces[model_type])
        if selection_enabled:   # how many votes a feature needs is tuned together with the model
            search_space.update(selection_config.get("search_space") or {})
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)
        optuna.logging.set_verbosity(optuna.logging.WARNING)   # no line per trial

        # the honest estimate: tuning repeated inside each outer fold (the same folds as above),
        # scored on rows it never saw. Its out-of-fold probabilities replace the untuned ones in
        # the reports below.
        print(f"\nTuning ({n_trials} trials, nested CV) ... this takes a while", flush=True)
        final_scores, oof_proba = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scorers, selection_metric,
            search_space, n_trials, tuning_seed, n_jobs=n_jobs,
        )
        print(f"  {selection_metric}: {final_scores['validation'].mean():.3f} ± "
              f"{final_scores['validation'].std(ddof=1):.3f}   (honest estimate of the tuned model)")
        report += "\n\n" + cv_report(final_scores, metrics, selection_metric,
                                     "Nested cross-validation (tuned model, honest estimate)")

        # the hyperparameters we keep: the same procedure, once, on all development rows
        pipeline, study = tune_pipeline(pipeline, X_dev, y_dev, inner_cv, scorers[selection_metric],
                                        search_space, n_trials, tuning_seed, n_jobs=n_jobs)
        print("  Chosen hyperparameters:")
        for name, value in study.best_params.items():
            print(f"    {name.replace('model__', ''):<22s} {value:.4g}" if isinstance(value, float)
                  else f"    {name.replace('model__', ''):<22s} {value}")
        report += "\n\n" + tuning_report(study, final_scores, selection_metric)

    print(f"\nAll metrics ({'tuned model, nested CV' if tuning_enabled else 'default hyperparameters'}, mean ± std)")
    for name in metrics:
        print(f"  {name:<18s} {final_scores[name].mean():.3f} ± {final_scores[name].std(ddof=1):.3f}")
    report += "\n\n" + oof_classification_report(y_dev, oof_proba)
    report += "\n" + fairness_report(y_dev, oof_proba, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    # the model we'd actually use: same (tuned) pipeline refit on EVERY development row.
    # CV estimated how well this recipe does; it didn't produce a model.
    pipeline.fit(X_dev, y_dev)
    footer = f"Final model: {model_type} refit on all {len(X_dev)} development rows."
    if tuning_enabled:
        footer += f"\nTuned hyperparameters: {study.best_params}"
    footer += f"\nLocked test set: {len(X_test)} rows set aside, not evaluated."
    if selection_enabled:
        all_features = pipeline.named_steps["prep"].get_feature_names_out()
        kept = pipeline.named_steps["select"].get_feature_names_out(all_features)
        footer += f"\nFeatures kept: {len(kept)} of {len(all_features)}: {', '.join(kept)}"
    report += "\n" + footer + "\n"

    # week 7: the scoreboard: one row per run, with the score on each fold, so any two runs on the same data and folds can be compared with corrected_paired_test()
    row = {"run": run, "data": config["data"]["path"], "model": model_type, "tuned": tuning_enabled,
           "encoder": config["preprocessing"]["encoder"], "scaler": config["preprocessing"]["scaler"],
           "feature_selection": f"min_votes {pipeline.named_steps['select'].min_votes}" if selection_enabled else "off",
           "params": str(config["model"].get("params")),
           "tuned_params": str(study.best_params) if tuning_enabled else "",
           "metric": selection_metric, "mean": final_scores["validation"].mean(),
           "std": final_scores["validation"].std(ddof=1), "untuned": untuned, "floor": floor,
           **{f"fold_{i}": score for i, score in enumerate(final_scores["validation"], start=1)}}
    output = config.get("output", {})
    runs = save_scores(output.get("scoreboard", "results/runs.csv"), row)
    same = runs[(runs["data"] == row["data"]) & (runs["metric"] == selection_metric)]   # only comparable runs
    # runs saved before week 8 have no feature_selection value: they ran without selection
    board = (same.assign(feature_selection=same["feature_selection"].fillna("off"))
             .sort_values("mean", ascending=False)
             [["run", "model", "tuned", "feature_selection", "mean", "std", "untuned", "floor"]]
             .rename(columns={"mean": selection_metric})
             .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nScoreboard (runs on the same data and metric, best first)\n{board}")
    report += f"\nScoreboard:\n{board}\n"

    shutil.rmtree(cache_dir, ignore_errors=True)   # the cache is only useful during this run
    path = save_run(output.get("results_dir", "results"), run, config, report)
    print(f"\n{footer}\nFull report (per-fold tables, tuning, confusion matrix, fairness): {path}")


if __name__ == "__main__":
    main()
