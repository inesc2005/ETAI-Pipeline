"""
Data diagnostics -- generic versions of the Week 3 EDA notebook's three
techniques (missingness mechanism, domain-rule validity, duplicate checks),
driven by `config.yaml`'s `diagnostics` section instead of hardcoded column
names.
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def cramers_v(confusion_matrix: pd.DataFrame) -> float:
    """Bias-corrected Cramer's V effect size for a chi-square test of association.
    Measures how strong the relationship between two categorical columns is (0,1)."""
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2_corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    return float(np.sqrt(phi2_corr / min(k_corr - 1, r_corr - 1)))


def test_missingness_mechanism(df: pd.DataFrame, target_col: str, candidate_predictors: list) -> pd.DataFrame:
    """
    For `target_col`'s missing-value indicator, test association against each
    column in `candidate_predictors` via chi-square + Cramer's V.
    Returns one row per predictor, sorted by association strength (strongest first).
    Tests whether a column's missing values are randomly scattered or linked to other columns (MCAR vs. MAR/MNAR).
    No verdict_from_max_v helper here on purpose -- the MCAR/MAR/MNAR call is a
    judgment made once while reading this table, not a threshold to hardcode.
    """
    indicator = df[target_col].isna()
    rows = []
    for predictor in candidate_predictors:
        if predictor == target_col or predictor not in df.columns:
            continue
        sub = pd.DataFrame({"missing": indicator, "predictor": df[predictor]}).dropna(subset=["predictor"])
        if sub["predictor"].nunique() < 2 or sub["missing"].nunique() < 2:
            continue
        table = pd.crosstab(sub["missing"], sub["predictor"])
        chi2, p, _, _ = chi2_contingency(table)
        v = cramers_v(table)
        rows.append({"predictor": predictor, "cramers_v": round(v, 3), "p_value": p, "n": len(sub)})
    result = pd.DataFrame(rows).sort_values("cramers_v", ascending=False).reset_index(drop=True)
    return result


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Apply a dict of {column: "python expression"} domain rules (the column name
    stands for its own numeric-coerced value in the expression, e.g.
    "18 <= age <= 100"). Any value that fails its rule is set to NaN --
    impossible but not missing is still missing.
    """
    out = df.copy()
    for column, rule in rules.items():
        values = pd.to_numeric(out[column], errors="coerce")
        valid = pd.eval(rule, local_dict={column: values})
        # only flag entries that were present in the first place -- NaNs are
        # already missing and belong to the missingness check, not this one
        violation = out[column].notna() & ~valid.fillna(False)
        out.loc[violation, column] = np.nan
    return out


def find_placeholder_rows(series: pd.Series, placeholder_tokens: set) -> pd.Series:
    """ which cells hold a placeholder token ( "-", "?") instead of a real value."""
    return series.astype(str).str.strip().isin(placeholder_tokens)


def find_bucket_mismatches(df: pd.DataFrame, value_col: str, bucket_col: str, buckets: list) -> pd.Series:
    """
    Flags rows where `bucket_col`'s label disagrees with the bucket that
    `value_col`'s numeric value falls into, given `buckets` (a list of
    {min?, max?, label} dicts -- min inclusive, max exclusive). Catches
    cross-column inconsistencies individual-column rules can't see, e.g.
    age_cat disagreeing with age, or score_text disagreeing with decile_score.
    """
    values = pd.to_numeric(df[value_col], errors="coerce")

    def expected_label(v):
        if pd.isna(v):
            return None
        for bucket in buckets:
            lower_ok = v >= bucket["min"] if "min" in bucket else True
            upper_ok = v < bucket["max"] if "max" in bucket else True
            if lower_ok and upper_ok:
                return bucket["label"]
        return None

    expected = values.apply(expected_label)
    actual = df[bucket_col].astype(str).str.strip()
    return expected.notna() & df[bucket_col].notna() & (actual != expected.astype(str))


def find_duplicates(df: pd.DataFrame, id_column: str) -> dict:
    """
    Reports duplicate rows two ways — exact full-row matches, and rows sharing the same id_column value
    """
    exact_dupe_mask = df.duplicated()
    id_dupe_mask = df[id_column].duplicated()
    return {
        "exact_row_duplicates": int(exact_dupe_mask.sum()),
        "repeated_ids": int(id_dupe_mask.sum()),
        "exact_duplicate_rows": df[exact_dupe_mask],
        "repeated_id_rows": df[df[id_column].isin(df.loc[id_dupe_mask, id_column])],
    }
