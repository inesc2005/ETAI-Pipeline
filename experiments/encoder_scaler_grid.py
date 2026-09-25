"""
One-off experiment (not part of the production pipeline): scores every
encoder x scaler combination across repeated splits, to pick config.yaml's
`preprocessing.encoder`/`preprocessing.scaler` with evidence instead of a
guess. Mirrors notebooks/02_preprocessing.ipynb's Step 4/5, but reuses this
project's real src/preprocessing.py functions instead of a notebook-only copy.

Run with:
    python experiments/encoder_scaler_grid.py
"""
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    _ENCODER_REGISTRY,
    _SCALER_REGISTRY,
    add_missing_flags,
    build_column_transformer,
    clean_dataset,
    split_features_target,
    split_train_test,
)

N_REPEATS = 15  # how many different splits to average each combination over (16 combinations)
TEST_SIZE = 0.25


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# trains and evaluates one encoder/scaler pair over N_REPEATS different splits
def score_combination(X, y, extras, preprocessing_config: dict, encoder_name: str, scaler_name: str) -> list:
    """Scores one (encoder, scaler) pair across N_REPEATS different splits, returns each split's accuracy."""
    # copy the base config, replacing only "encoder" and "scaler" with this combination's values
    combo_config = {**preprocessing_config, "encoder": encoder_name, "scaler": scaler_name}
    scores = []
    for repeat in range(N_REPEATS):
        X_train, X_test, y_train, y_test, _extras_train, _extras_test = split_train_test(
            X, y, extras, test_size=TEST_SIZE, random_state=repeat
        )
        # a 2-step pipeline: first clean/prepare the columns (this combo's encoder+scaler), then the model
        pipeline = Pipeline([
            ("prep", build_column_transformer(combo_config)),
            # max_iter: iterations allowed to converge; class_weight="balanced": correct for the class imbalance
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
        pipeline.fit(X_train, y_train)  # learn everything (medians, categories, model weights) from train only
        predictions = pipeline.predict(X_test)  # apply what it learned to the unseen test rows
        scores.append(accuracy_score(y_test, predictions))  # % of test predictions that matched the real outcome
    return scores


def main():
    config = load_config()

    # clean once -- doesn't depend on which encoder/scaler is being tested
    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = add_missing_flags(df_clean, config["preprocessing"]["mnar_columns"])
    X, y, extras = split_features_target(
        df_clean,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
    )

    per_combo_scores = {}  # keeps all 15 accuracies per combination (needed later for the paired comparison)
    rows = []               # keeps just the mean/std per combination (used to build the results table)
    for encoder_name in _ENCODER_REGISTRY:
        for scaler_name in _SCALER_REGISTRY:
            scores = score_combination(X, y, extras, config["preprocessing"], encoder_name, scaler_name)
            per_combo_scores[(encoder_name, scaler_name)] = scores
            rows.append({
                "encoder": encoder_name,
                "scaler": scaler_name,
                "mean_accuracy": np.mean(scores),
                "std_accuracy": np.std(scores),
            })

    results = pd.DataFrame(rows).sort_values("mean_accuracy", ascending=False).reset_index(drop=True)  # best first
    print(results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    winner = results.iloc[0]
    runner_up = results.iloc[1]
    winner_scores = np.array(per_combo_scores[(winner["encoder"], winner["scaler"])])
    runner_up_scores = np.array(per_combo_scores[(runner_up["encoder"], runner_up["scaler"])])

    # winner_scores[i] and runner_up_scores[i] came from the exact same split (same random_state=i),
    # so subtracting them pairwise is a fair comparison, not just "average A minus average B"
    diffs = winner_scores - runner_up_scores
    mean_diff = diffs.mean()  # average of those 15 per-split differences
    se_diff = diffs.std(ddof=1) / np.sqrt(len(diffs))  # how much that average could plausibly vary by chance

    print(f"\nWinner:    {winner['encoder']} + {winner['scaler']} (mean accuracy {winner['mean_accuracy']:.4f})")
    print(f"Runner-up: {runner_up['encoder']} + {runner_up['scaler']} (mean accuracy {runner_up['mean_accuracy']:.4f})")
    print(f"Paired mean difference: {mean_diff:+.4f} (SE {se_diff:.4f}, over {N_REPEATS} paired splits)")
    if abs(mean_diff) > 2 * se_diff:
        print("-> more than 2 standard errors from zero: a real, if modest, edge.")
    else:
        print("-> within about 2 standard errors of zero: could easily be noise. Either pair is defensible.")


if __name__ == "__main__":
    main()
