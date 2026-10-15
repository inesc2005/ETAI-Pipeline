"""
Preprocessing -- raw data in, model-ready train/test split out.

Two stages:
  1. `clean_dataset` -- the week 3 cleanup recipe (placeholders/domain-rule
     violations -> NaN, category spelling cleanup, duplicate removal,
     redundant column drop). Still has NaNs when it's done -- it only
     flags problems, it doesn't fix them.
  2. `split_features_target` (adds the missing flags) -> `split_dev_test` ->
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
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler, TargetEncoder,
)
from category_encoders import CountEncoder

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
      4. drop perfectly/near-perfectly redundant columns
      5. cross-column mismatches (age_cat vs age, score_text vs decile_score) -> NaN
     uses find_placeholder_rows/flag_invalid_values from data_diagnostics.py, the rest is local
    """
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))
    canonical_maps = diagnostics_config.get("canonical_maps", {})
    validity_rules = diagnostics_config.get("validity_rules", {})
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

    out = out.drop(columns=[c for c in columns_to_drop if c in out.columns])  # drop redundant columns

    return out


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    TRAINING DATA ONLY. Drops exact duplicate rows and repeated ids (keeping the first), so
    the same person can't be counted twice -- or land in both the development and the
    locked test set. Must run *before* `split_dev_test()`, never on data you're predicting for.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


def add_missing_flags(df: pd.DataFrame, mnar_columns: list) -> pd.DataFrame:
    """
    Adds a `<col>_was_missing` column (0/1) for every column the verdict table
    diagnosed as MAR/MNAR, before it gets imputed with the median/mode
    """
    # doing this allows the model to keep information about which rows are missing, learning with it if its missing, is connected to the target variable
    out = df.copy()
    for col in mnar_columns:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


# Week 6 (Challenge 1): fixed limits for priors_bin, chosen by hand as round numbers
# (none / a few / several / many priors) -- not computed from the data
PRIORS_BIN_EDGES = [-0.5, 0.5, 3.5, 9.5, float("inf")]   # -> 0 | 1-3 | 4-9 | 10+


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds 5 new columns, each a fixed formula (nothing learned from the data), so it is safe before the split. 
    Missing inputs give NaN, imputed later in the Pipeline. The model only uses the columns listed in `preprocessing.numeric_features` (config.yaml).
      - priors_per_adult_year: priors_count / (age - 17) 
      - has_juvenile_record:   1 if any juvenile count > 0   (attempt 1)
      - priors_bin:            priors_count in groups 0 | 1-3 | 4-9 | 10+   (attempt 1)
      - felony_with_priors:    priors_count if the charge is a felony, else 0   (attempt 2)
      - log_priors:            log(1 + priors_count)   (attempt 2)
    """
    out = df.copy()
    out["priors_per_adult_year"] = out["priors_count"] / (out["age"] - 17).clip(lower=1)

    juv = out[["juv_fel_count", "juv_misd_count", "juv_other_count"]]
    any_positive = (juv > 0).any(axis=1)        # one positive count is enough to know the answer
    all_known = juv.notna().all(axis=1)         # otherwise, all three must be known to say "no"
    out["has_juvenile_record"] = any_positive.astype(float).where(any_positive | all_known)

    out["priors_bin"] = pd.cut(out["priors_count"], bins=PRIORS_BIN_EDGES, labels=False)

    is_felony = out["c_charge_degree"].map({"F": 1.0, "M": 0.0})   # unknown charge -> NaN
    out["felony_with_priors"] = out["priors_count"] * is_felony
    out["log_priors"] = np.log1p(out["priors_count"])
    return out


def split_features_target(df: pd.DataFrame, target: str, sensitive_attr: str, drop_columns: list,
                          mnar_columns: list = ()):
    """
    Splits a cleaned dataframe into (X, y, extras), splits the columns.
    Adds the `<col>_was_missing` flags first (see `add_missing_flags`), so anything that builds
    X through this function -- training or label-free inference data -- gets them automatically.
    """
    df = add_missing_flags(df, mnar_columns)
    y = df.get(target) # target variable we want to predict

    # kept aside (like race and score_text) for later evaluation, but not used as model input features
    extras_columns = [c for c in (sensitive_attr, "score_text") if c in df.columns]
    extras = df[extras_columns].copy() if extras_columns else None

    drop_these = {target, sensitive_attr, *drop_columns}
    X = df[[c for c in df.columns if c not in drop_these]]  # everything left over, what the model actual input features

    return X, y, extras


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Sets the final test set aside (replaces `split_train_test`).

    Stratified split of X, y and extras (race/score_text, for the fairness report) together,
    so all three stay row-aligned. Returns a *development* set and a *locked test set*:
      - development set: everything we may learn from and compare models on. Cross-validation
        splits it again into train/validation folds.
      - locked test set: never used to fit, tune, compare or choose anything. Its size and
        seed live in config.yaml's `test_set` section and must not change afterwards.
    """
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = train_test_split(
        X, y, extras, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_dev, X_test, y_dev, y_test, extras_dev, extras_test


# menu of config names -> (class, kwargs)
# rescales numeric columns to a comparable range
_SCALER_REGISTRY = {
    "none": ("passthrough", {}),
    "standard": (StandardScaler, {}),
    "minmax": (MinMaxScaler, {}),
    "robust": (RobustScaler, {}),
}

# Encoders turn text categories into numbers. "target" replaces each category by its reoffending rate;
# cv=StratifiedKFold(5) cross-fits it: each training row is encoded with the OTHER 4 parts, never its own label.
_ENCODER_REGISTRY = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore"),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
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
    encoder = _ENCODER_REGISTRY[preprocessing_config.get("encoder", "onehot")](preprocessing_config.get("random_state"))

    if numeric_strategy == "knn":
        # KNN measures distances between rows, so the columns must be on a comparable scale
        # first (the scalers ignore NaNs while fitting); otherwise `age` would dominate
        numeric_pipeline = Pipeline([
            ("scale", scaler),
            ("impute", KNNImputer(n_neighbors=preprocessing_config.get("knn_neighbors", 5))),
        ])
    else:
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
