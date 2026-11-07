"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (cross-validation scores, classification report, fairness table) to a timestamped file in `results/` instead, so you can open it again later, or compare two runs side by side after changing something in config.yaml.

Week 7: every run also adds one row to the scoreboard (`results/runs.csv`), so all runs can be
compared in one table, and any two of them fold by fold with `corrected_paired_test()`.
"""
import os

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """
    Appends one run (`row`: the model, its settings, the selection metric's mean, std and score
    on each fold, the untuned score and the no-model floor) to the CSV at `path`, creating it if
    needed, and returns all the runs saved so far.
    """
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes one run's settings and full report to `results_dir/run_<run>.txt` (the folder is
    created automatically if it doesn't exist yet) and returns the path that was written.
    `run` is the run id built in main.py, the same one saved on the scoreboard.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    if tuning.get("enabled"):
        tuning_line = (f"Optuna (TPE), {tuning['n_trials']} trials x {tuning['n_splits']} inner folds, "
                       f"nested CV  random_state={tuning['random_state']}")
    else:
        tuning_line = "off"
    selection = config.get("feature_selection") or {}
    selection_line = (f"VoteSelector, min_votes={selection.get('min_votes', 2)} (before tuning)"
                      if selection.get("enabled") else "off")
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Metrics: {cv['metrics']}  selection metric: {cv['selection_metric']}\n"
        f"Feature selection: {selection_line}\n"
        f"Tuning: {tuning_line}\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)

    return path
