"""
Final assessment on the LOCKED test set -- run this once, after the model is chosen.

Run from the project root with:
    python -m experiments.final_test

Fits the model from config.yaml on the whole development set and scores it on the test set
that was set aside in `split_dev_test`. Nothing is tuned or chosen from this number: it is
only reported. A dummy model and COMPAS's own score (on the same rows) are shown for context.
"""
    #  Final test: fits the chosen model on the dev set and scores it once on the locked test set.
    # Run it only after the model is chosen. The result is reported, never used to change the model.
    # Compare the chosen model with a dummy model and COMPAS's own score (on the same rows) for comparison.

import os
from datetime import datetime

import yaml
from sklearn.metrics import accuracy_score, classification_report

from src.data import load_data
from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    make_pipeline,
    split_dev_test,
    split_features_target,
)


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    df = load_data(config["data"]["path"])
    df = clean_dataset(df, config["diagnostics"])
    df = drop_duplicate_rows(df, config["diagnostics"].get("id_column"))
    X, y, extras = split_features_target(
        df, target=config["data"]["target"], sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        mnar_columns=config["preprocessing"]["mnar_columns"],
    )
    X_dev, X_test, y_dev, y_test, _e_dev, extras_test = split_dev_test(
        X, y, extras, test_size=config["test_set"]["size"], random_state=config["test_set"]["random_state"],
    )

    model = make_pipeline(config["model"], config["preprocessing"]).fit(X_dev, y_dev)
    dummy = make_pipeline({"type": "dummy", "params": {}}, config["preprocessing"]).fit(X_dev, y_dev)
    pred = model.predict(X_test)
    compas = (extras_test["score_text"] != "Low").astype(int)

    lines = [
        f"FINAL ASSESSMENT on the locked test set ({len(X_test)} rows; fitted on {len(X_dev)} development rows)",
        f"Model: {config['model']['type']}  params={config['model'].get('params')}",
        "",
        f"Test accuracy:        {accuracy_score(y_test, pred):.3f}",
        f"Dummy (majority):     {accuracy_score(y_test, dummy.predict(X_test)):.3f}",
        f"COMPAS's own score:   {accuracy_score(y_test, compas):.3f}",
        "",
        "Classification report (test set):",
        classification_report(y_test, pred, zero_division=0),
    ]
    text = "\n".join(lines)
    print(text)

    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"final_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
