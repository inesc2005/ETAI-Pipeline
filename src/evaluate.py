"""
Evaluation -- stratified k-fold cross-validation on the development set.

A single train/test split gives one number that moves with the seed. Cross-validation fits the
whole Pipeline (preprocessing + model) k times, each time on k-1 folds, scoring on the fold it
did not see, and reports every fold's score plus mean AND std (the honest "how much would this
move with different data"). The locked test set is never used here.

"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    One scorer per metric name in metrics in config.yaml. 
    scikit-learn already knows most names (`accuracy`, `roc_auc`, `f1`, ...); `fbeta_<beta>` needs its beta, so it is built here
    (`fbeta_2`: recall counts twice as much as precision).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_"):
            beta = float(name.split("_", 1)[1])
            scorers[name] = make_scorer(fbeta_score, beta=beta, zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of pipeline on each fold's training part and scores it on that fold's
    validation part, on every metric in scorers. Since pipeline includes the preprocessing,
    imputation medians, encoder statistics and scaler means are re-learned inside every fold.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold, train score, validation score and gap of the
        `selection_metric` (a large, consistent gap = overfitting), then the validation score
        of every metric.
      - oof_proba: out-of-fold PROBABILITY of reoffending for every row, from the fold model
        that did NOT train on it. The reports turn it into yes/no with a threshold.
    """
    # 5 rounds: in each, a fresh copy of the pipeline trains on 4 folds and is scored on the 5th.
    # The folds are the same for every model (fixed seed), so models are compared on equal terms.
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, len(scores["indices"]["test"]) + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = np.empty(len(X))
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]   # column 1 = class "reoffends"
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table, then the mean and std of every metric, as text (saved to results/)."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        "",
        f"{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Is model A really better than model B, or is the difference just noise?
    Both models must be scored on the same folds; `scores_a` and `scores_b` are their scores
    on each fold (e.g. the fold_1 ... fold_5 columns of results/runs.csv).

    It looks at the difference between A and B in each fold. The folds share most of their
    training rows, so a normal t-test would be too confident; this version (Nadeau & Bengio)
    corrects for that with `test_train_ratio` (1 / (k - 1) for k folds: 0.25 with 5 folds).

    Returns:
      - mean_diff: how much better A is on average (A - B);
      - wins: in how many folds A beat B;
      - t: the test statistic;
      - p_value: below 0.05 -> the difference is real; above -> it could be noise.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)   # A - B, fold by fold
    k = len(diff)                                                                     # number of folds
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))   # uncertainty of the mean difference, corrected
    t = float(diff.mean() / se) if se > 0 else 0.0                 # how many "uncertainties" the difference is
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))              # chance of a difference this big by luck alone
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5) -> str:
    """
    Confusion matrix and classification report (precision, recall and f1 per class) of the
    out-of-fold predictions: probability >= `threshold` counts as "predicted to reoffend".
    """
    y_pred = (oof_proba >= threshold).astype(int)
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=["true 0", "true 1"],
                      columns=["predicted 0", "predicted 1"])
    return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
            f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5) -> str:
    """
    Deliberately simple fairness check -- not a substitute for a real audit, just enough to show
    that "accurate" and "fair" are not the same thing.

    For each group of `sensitive_attr`, the false positive rate (share of people who did NOT
    reoffend but were predicted to) for:
        - our own model (out-of-fold probability >= `threshold`)
        - COMPAS's own risk score (score_text != "Low" counts as a "high risk" prediction), for comparison
    """
    df = extras.copy()
    # people whose group is missing are reported as "unknown" instead of silently dropped by groupby
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold, threshold {threshold})",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)
