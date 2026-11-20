"""
Week 10: predicts new people with the model saved by `main.py`, without training anything.

Run from the project root with:
    python predict.py data/compas_new_people.csv predictions.csv                        # COMPAS (config.yaml)

new CSV -> the same cleaning as training (but no `drop_duplicate_rows`: every row needs an answer)
-> the same features -> the saved pipeline (every value it learned in training: medians, scale,
encodings, kept features, model) -> one row per input row: id, probability of class 1, prediction.
Nothing is learned from the new data. `python main.py <config>` must have run first: it saves the
model to `output.model_path`.
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.data import load_data
from src.preprocessing import add_engineered_features, clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict, threshold: float = 0.5) -> pd.DataFrame:
    """
    Returns the id, the probability of class 1 and the 0/1 prediction (probability >= threshold)
    for every row of `df`, a raw file without the target.
    """
    clean = clean_dataset(df, config["diagnostics"])
    if config["preprocessing"].get("engineered_features", False):
        clean = add_engineered_features(clean)
    X, _, _ = split_features_target(
        clean,
        target=config["data"]["target"],          # not in the new file: y comes back as None
        sensitive_attr=config["data"]["sensitive_attr"],
        drop_columns=config["data"]["drop_columns"],
        mnar_columns=config["preprocessing"]["mnar_columns"],
    )
    probability = pipeline.predict_proba(X)[:, 1]
    id_column = config["diagnostics"].get("id_column", "id")
    predictions = pd.DataFrame({id_column: clean[id_column].values,
                                "probability": probability,
                                "prediction": (probability >= threshold).astype(int)})
    if len(predictions) != len(df):
        raise ValueError(f"{len(df)} rows in, {len(predictions)} predictions out: a row was dropped")
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="the config the model was trained with")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config["output"]["model_path"])     # only load model files you trust
    threshold = (config.get("predict") or {}).get("threshold", 0.5)
    predictions = predict(load_data(args.input), pipeline, config, threshold)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}  "
          f"({predictions['prediction'].mean():.0%} predicted as class 1, threshold {threshold})")
