"""
Preprocessing -- raw data in, model-ready train/test split out.

Two stages:
  1. `clean_dataset` -- the week 3 cleanup recipe (placeholders/domain-rule
     violations -> NaN, category spelling cleanup, duplicate removal,
     redundant column drop). Still has NaNs when it's done -- it only
     flags problems, it doesn't fix them.
  2. `add_missing_flags` -> `split_features_target` -> `split_train_test` ->
     `build_column_transformer`: turns the cleaned-but-still-messy data into
     something a model can train on -- a "<col>_was_missing" flag for every
     MAR/MNAR column (see the verdict table), then a stratified split, then
     imputation/encoding/scaling that's only ever fit on the training fold
     (done via the ColumnTransformer inside main.py's sklearn Pipeline, so
     nothing about the test fold leaks into how the training fold gets
     transformed).

One thing that was NOT naive from week 2, kept as-is: `sensitive_attr`
(race) is kept out of the model's input features entirely. It's split
alongside the data so it's still available afterwards -- not to train on,
but to check whether the model treats different groups differently. See
src/evaluate.py:fairness_report.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler
from category_encoders import CountEncoder, TargetEncoder

from src.data_diagnostics import find_bucket_mismatches, flag_invalid_values, find_placeholder_rows


def canonicalize_categories(df: pd.DataFrame, canonical_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    """Normalizes spelling/casing variants of a category into one canonical label"""
    out = df.copy()
    for col, mapping in canonical_maps.items():
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Apply the Week 3 EDA notebook's verdict table to the raw data:
      1. placeholder tokens ("-", "?", ...) -> NaN
      2. domain-rule violations (impossible values) -> NaN
      3. category spelling variants -> one canonical label
      4. drop exact full-row duplicates only -- a repeated id whose other columns
         differ is NOT dropped (could be two genuinely different records that
         happen to share an id), just flagged with a warning
      5. drop perfectly/near-perfectly redundant columns
      6. cross-column mismatches (age_cat vs age, score_text vs decile_score) -> NaN
     uses find_placeholder_rows/flag_invalid_values from data_diagnostics.py, the rest is local
    """
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))
    canonical_maps = diagnostics_config.get("canonical_maps", {})
    validity_rules = diagnostics_config.get("validity_rules", {})
    id_column = diagnostics_config.get("id_column")
    columns_to_drop = diagnostics_config.get("columns_to_drop", [])

    out = df.copy()

    # numeric-looking columns that loaded as text because of placeholder tokens
    for col in out.columns:
        if out[col].dtype == object:
            mask = find_placeholder_rows(out[col], placeholder_tokens)
            out.loc[mask, col] = np.nan  # replace placeholder tokens with real NaN

    if canonical_maps:
        out = canonicalize_categories(out, canonical_maps, placeholder_tokens)  # normalize category labels

    for col in ["priors_count", "prior_offenses"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")  # force back to numeric dtype

    if validity_rules:
        out = flag_invalid_values(out, validity_rules)  # impossible values -> NaN

    # category label disagrees with its own numeric source value (e.g. age=55
    # but age_cat="Less than 25") -- trust the numeric column, null the label
    bucket_consistency = diagnostics_config.get("bucket_consistency", {})
    for bucket_col, spec in bucket_consistency.items():
        if bucket_col in out.columns and spec["value_column"] in out.columns:
            mismatch = find_bucket_mismatches(out, spec["value_column"], bucket_col, spec["buckets"])
            out.loc[mismatch, bucket_col] = np.nan

    out = out.drop_duplicates()  # exact full-row duplicates only

    if id_column and id_column in out.columns:
        # a repeated id with different data is NOT an exact duplicate -- don't
        # silently drop it, it could be two real records that share an id
        still_repeated = out[id_column].duplicated(keep=False)
        if still_repeated.any():
            n_ids = out.loc[still_repeated, id_column].nunique()
            warnings.warn(
                f"{n_ids} '{id_column}' value(s) repeat with different data across rows -- "
                "kept as-is, not dropped, since they are not exact duplicates."
            )

    out = out.drop(columns=[c for c in columns_to_drop if c in out.columns])  # drop redundant columns

    return out


def add_missing_flags(df: pd.DataFrame, mnar_columns: list) -> pd.DataFrame:
    """
    Adds a `<col>_was_missing` column (0/1) for every column the verdict table
    diagnosed as MAR/MNAR, before it gets imputed with the median/mode
    """
    out = df.copy()
    for col in mnar_columns:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, target: str, sensitive_attr: str, drop_columns: list):
    """
    Splits a cleaned dataframe into (X, y, extras), splits the columns
    """
    y = df.get(target) # target variable we want to predict

    # kept aside (like race and score_text) for later evaluation, but not used as model input features
    extras_columns = [c for c in (sensitive_attr, "score_text") if c in df.columns]
    extras = df[extras_columns].copy() if extras_columns else None

    drop_these = {target, sensitive_attr, *drop_columns}
    X = df[[c for c in df.columns if c not in drop_these]]  # everything left over, what the model actual input features

    return X, y, extras


def split_train_test(X, y, extras, test_size: float, random_state: int):
    """
    Splits rows into train/test sets, keeping X, y, and extras row-aligned.
    Random State (=42) fixes the randomness so the same split is reproduced every run.
    stratify=y: Keeps the target's class proportion balanced across train and test.
    """
    X_train, X_test, y_train, y_test, extras_train, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, extras_train, extras_test

# menu of config names -> (class, kwargs)
# rescales numeric columns to a comparable range
_SCALER_REGISTRY = {
    "none": ("passthrough", {}),
    "standard": (StandardScaler, {}),
    "minmax": (MinMaxScaler, {}),
    "robust": (RobustScaler, {}),
}

#   turns categorical columns into numbers e
_ENCODER_REGISTRY = {
    "onehot": (OneHotEncoder, {"handle_unknown": "ignore"}),
    "ordinal": (OrdinalEncoder, {"handle_unknown": "use_encoded_value", "unknown_value": -1}),
    "count": (CountEncoder, {"handle_unknown": 0, "handle_missing": 0}),
    "target": (TargetEncoder, {"handle_unknown": "value", "handle_missing": "value"}),
}

# turns a chosen config name into the actual sklearn object it stands for
def _instantiate(registry: dict, name: str):
    factory, kwargs = registry[name]
    return factory if factory == "passthrough" else factory(**kwargs)


def build_column_transformer(preprocessing_config: dict) -> ColumnTransformer:
    """
    Builds a ColumnTransformer that imputes, encodes, and scales each column group (numeric, categorical, indicators) according to the config.
    """
    numeric_features = preprocessing_config["numeric_features"]
    categorical_features = preprocessing_config["categorical_features"]
    mnar_columns = preprocessing_config.get("mnar_columns", [])
    numeric_strategy = preprocessing_config.get("numeric_impute_strategy", "median")
    categorical_strategy = preprocessing_config.get("categorical_impute_strategy", "most_frequent")

    scaler = _instantiate(_SCALER_REGISTRY, preprocessing_config.get("scaler", "standard"))
    encoder = _instantiate(_ENCODER_REGISTRY, preprocessing_config.get("encoder", "onehot"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=numeric_strategy)),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=categorical_strategy)),
        ("encode", encoder),
    ])
    indicator_columns = [f"{col}_was_missing" for col in mnar_columns]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
        ("indicators", "passthrough", indicator_columns),
    ])
