"""
Hyperparameter tuning (Optuna) and nested cross-validation 

Choosing hyperparameters is a decision like any other, so it follows the same rules: 
development set only, every candidate scored by CV of the WHOLE pipeline (preprocessing re-fit in every fold of every trial), the same folds for every candidate. 
The best trial's score is the maximum of many noisy CV scores, so it's optimistic: it's used to choose, never to report. 
The honest number for "tune, then refit" comes from `nested_cross_validate`. The locked test set is never used here.
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score


def _suggest_params(trial, search_space: dict) -> dict:
    """
    Asks the trial for one value per hyperparameter in `search_space` (read from config.yaml, so the ranges are never typed in the code). 
    Keys are Pipeline parameter names (`model__max_depth`). Groups the answer in a dict

    It is called once in EACH trial (`tune_pipeline` runs `n_trials` trials, 30 in config.yaml),
    and returns a different set of values every time. Optuna chooses the values, this function only asks.
    """
    params = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
        elif kind == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown search-space type for {name}: {kind}. Options: int, float, categorical")
    return params


def tune_pipeline(pipeline, X, y, cv, scoring: str, search_space: dict, n_trials: int,
                  random_state: int, n_jobs: int = 1):
    """
    Runs an Optuna study (TPE sampler, seeded) on (X, y). 
    Each trial draws a candidate from `search_space`, sets it on a fresh, unfitted copy of `pipeline`.
    Scores it by its mean CV score over `cv` (the same folds for every trial)

    The mean is over the fold SCORES of that trial. 
    After all `n_trials` trials nothing is averaged either: the winner is the trial with the highest mean score.

    Returns (best_pipeline, study):
      - best_pipeline: an UNFITTED copy of `pipeline` with the best trial's hyperparameters. 
      - study: every trial with its values and score.
    """
    def objective(trial):
        params = _suggest_params(trial, search_space)
        #  build the candidate: clone = a new, unfitted copy of the pipeline then set_params applies this trial's values to it
        candidate = clone(pipeline).set_params(**params)

        # score the candidate by CV of the WHOLE pipeline. Returns one score per fold.
        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        trial.set_user_attr("std", float(scores.std(ddof=1)))

        return scores.mean() #returns the mean score CV

    # the study runs the loop and remembers every trial. 
    # TPE proposes the values: random for the first trials, then guided by the scores seen so far.
    study = optuna.create_study(direction="maximize",   # sklearn scorers: higher is better
                                sampler=optuna.samplers.TPESampler(seed=random_state))

    # run the loop n_trials times: propose values -> objective() scores them -> Optuna learns
    study.optimize(objective, n_trials=n_trials)

    # the winner is the trial with the highest mean CV score: ONE winner per call.
    best_pipeline = clone(pipeline).set_params(**study.best_params)
    return best_pipeline, study


def nested_cross_validate(pipeline, X, y, outer_cv, inner_cv, scoring: str, search_space: dict,
                          n_trials: int, random_state: int, n_jobs: int = 1):
    """
    "How good is the best hyperparameter choice?" The honest answer comes from nested CV: the inner loop chooses, the outer loop judges.
    The data is split into folds (`outer_cv`, 5 in config.yaml). For each fold:
      1. that fold (outer-validation) is set aside and nobody sees it;
      2. the other folds (outer-training) are used to tune: `tune_pipeline` with `inner_cv`
         runs all the trials and returns the winner, the trial with the best score (this
         function does not choose it, it only receives it and uses it);
      3. the winner (an unfitted pipeline) is refit on the outer-training rows;
      4. the winner is scored on the fold that was set aside. 
      5. the fold's scores, what the tuner believed (`inner_best`: the winner's score during the
         tuning, i.e. `study.best_value`), the hyperparameters chosen and the predictions are saved.
    In short, in every round: tune with 4 folds, evaluate on the 5th. The inner loop chooses, the
    outer loop judges. Each outer fold may pick different hyperparameters 

    """
    scorer = get_scorer(scoring)
    rows = []
    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)

    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(pipeline, X_train, y_train, inner_cv, scoring,
                                             search_space, n_trials, random_state, n_jobs)
        best_pipeline.fit(X_train, y_train)

        train_score = scorer(best_pipeline, X_train, y_train)
        val_score = scorer(best_pipeline, X_val, y_val)
        y_oof[val_idx] = best_pipeline.predict(X_val)
        rows.append({"fold": fold, "train": train_score, "validation": val_score,
                     "gap": train_score - val_score, "inner_best": study.best_value,
                     **study.best_params})

    return pd.DataFrame(rows), y_oof


def tuning_report(study, nested_scores: pd.DataFrame, scoring: str = "accuracy", top: int = 5) -> str:
    """
    The final study (best hyperparameters, top trials) plus the optimism check: the mean tuning
    score inside the outer folds vs. the mean honest score on them. As text (printed and returned).
    """
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={"number": "trial", "value": f"mean {scoring}"})
    top_trials = trials.sort_values(f"mean {scoring}", ascending=False).head(top)

    inner = nested_scores["inner_best"].mean()
    outer = nested_scores["validation"].mean()
    lines = [
        f"Tuning on the whole development set ({len(study.trials)} Optuna trials, metric: {scoring})",
        f"Best hyperparameters: {study.best_params}",
        f"Best mean CV score:   {study.best_value:.3f}  (maximum over all trials -- optimistic, not reported)",
        "",
        f"Top {top} trials:",
        top_trials.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "Optimism check (nested CV):",
        f"  tuning score inside the outer folds (inner_best) mean = {inner:.3f}",
        f"  honest score on the outer folds (validation)     mean = {outer:.3f}",
        f"  optimism = {inner - outer:+.3f}   -> the honest one is the number to report",
    ]
    text = "\n".join(lines)
    print(text)
    return text
