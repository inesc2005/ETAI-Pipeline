"""
Leakage check (week 9) -- does any single column give the answer away?

Run from the project root with:
    python -m experiments.leakage_check                      # Adult (config_adult.yaml)
    python -m experiments.leakage_check config.yaml          # or any other config

For every feature listed in the config, the ROC-AUC of that feature ALONE, with no model:
  - numeric features: the value itself is the score (missing -> median). max(AUC, 1 - AUC), because
    a feature that predicts "the other way round" is just as informative;
  - categorical features: each category is replaced by its share of people with target = 1.
A feature close to 1.0 would almost give the answer away (a possible leak); close to 0.5 means
it says nothing on its own. This is a quick diagnostic on all labelled rows, not used by the model.
The category shares are computed on the same rows they are scored on, so they are slightly
optimistic, especially for features with many categories.
"""
import os
import sys

import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.data import load_data
from src.preprocessing import clean_dataset


def single_feature_auc(df: pd.DataFrame, target: str, numeric: list, categorical: list) -> pd.DataFrame:
    """ROC-AUC of each feature alone, highest first."""
    y = df[target]
    rows = []
    for col in numeric:
        score = df[col].fillna(df[col].median())
        auc = roc_auc_score(y, score)
        rows.append({"feature": col, "kind": "numeric", "single_feature_roc_auc": max(auc, 1 - auc)})
    for col in categorical:
        category = df[col].fillna("missing")
        score = category.map(y.groupby(category).mean())   # each category -> its share of target = 1
        rows.append({"feature": col, "kind": "categorical", "single_feature_roc_auc": roc_auc_score(y, score)})
    return pd.DataFrame(rows).sort_values("single_feature_roc_auc", ascending=False)


def main(config_path: str = "config_adult.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    df = clean_dataset(load_data(config["data"]["path"]), config["diagnostics"])
    table = single_feature_auc(df, config["data"]["target"],
                               config["preprocessing"]["numeric_features"],
                               config["preprocessing"]["categorical_features"])

    text = (f"Leakage check: ROC-AUC of each feature alone ({len(df)} rows of {config['data']['path']}, "
            f"after clean_dataset). A diagnostic only, not used by the model.\n"
            f"Close to 1.0 = the feature almost gives the answer away; close to 0.5 = chance.\n\n"
            + table.to_string(index=False, float_format=lambda v: f"{v:.3f}") + "\n")
    print(text)

    results_dir = config.get("output", {}).get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "leakage_check.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "config_adult.yaml")
