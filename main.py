"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> clean -> add missingness flags -> split
    features/target -> train/test split -> build leak-safe preprocessor
    -> train -> evaluate (train & test) -> save results
"""
import yaml
from sklearn.pipeline import Pipeline

from src.data import load_data
from src.preprocessing import (
    add_missing_flags,
    build_column_transformer,
    clean_dataset,
    split_features_target,
    split_train_test,
)
from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    df = load_data(config["data"]["path"])
    df_clean = clean_dataset(df, config["diagnostics"])
    df_clean = add_missing_flags(df_clean, config["preprocessing"]["mnar_columns"])

    X, y, extras = split_features_target(
        df_clean,
        target=config["data"]["target"],
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
    )

    # leak-safe split: everything above this line may see the whole dataset;
    # everything below (imputation, encoding, scaling, inside the Pipeline's
    # ColumnTransformer) is fit only on the training fold, via pipeline.fit
    X_train, X_test, y_train, y_test, _extras_train, extras_test = split_train_test(
        X, y, extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    preprocessor = build_column_transformer(config["preprocessing"])
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", build_model(config["model"])),
    ])
    pipeline.fit(X_train, y_train)

    # predict on both splits -- train accuracy vs. test accuracy is how we'll spot overfitting, not just how "good" the model looks
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    report = evaluate(y_train, y_train_pred, y_test, y_test_pred)
    report += "\n" + fairness_report(
        y_test, y_test_pred, extras_test, sensitive_attr=config["data"]["sensitive_attr"]
    )

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()
