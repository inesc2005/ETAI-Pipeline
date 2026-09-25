**Name: Inês Calado**
**Student Number: 20260537**

## Week 3 -- diagnosis-driven preprocessing

### What was added, and where

**`src/data_diagnostics.py`** (new) holds the diagnostic functions. They only report on the data, nothing here changes a value:
- `cramers_v` -- how strongly two categorical columns are associated (0 = unrelated, 1 = perfectly related).
- `test_missingness_mechanism` -- tests a column's missing-value indicator against candidate predictors, to judge MCAR vs. MAR/MNAR.
- `flag_invalid_values` -- checks domain rules (`18 <= age <= 100`) and turns violations into `NaN`.
- `find_placeholder_rows` -- flags cells holding a placeholder token (`"-"`, `"?"`, ...) instead of a real value.
- `find_bucket_mismatches` -- flags rows where a category label disagrees with the bucket its own numeric value falls into, e.g. `age_cat` disagreeing with `age`, or `score_text` disagreeing with `decile_score`.
- `find_duplicates` -- also reporting only. Counts and returns rows two ways: exact full-row matches, and rows sharing the same id. Those can differ -- a repeated id whose other columns don't match isn't an exact duplicate, and `clean_dataset` only drops the second kind.

**`src/preprocessing.py`** now owns the whole "raw data in, model-ready split out" job:
- `canonicalize_categories` -- merges spelling/casing variants of a category into one label (`"MALE"`/`"male"`/`" Male"` all become `"Male"`).
- `clean_dataset` -- runs the full cleanup: placeholder tokens to `NaN`, domain-rule violations to `NaN`, category canonicalization, the cross-column bucket check above, duplicate removal (exact rows only -- a repeated id with differing data is kept and flagged with a warning, not silently dropped), then drops the redundant columns.
- `add_missing_flags` -- adds a `<col>_was_missing` flag for every MAR/MNAR column before imputing it, so the model can still pick up on the missingness pattern even though the fill value is just a median or mode.
- `split_features_target` -- splits into `(X, y, extras)`. `y` comes from `df.get(target)`, so it returns `None` instead of raising when there's no label column.
- `split_train_test` -- the leak-safe boundary. Everything after this point (imputation, encoding, scaling) only ever gets fit on the training fold.
- `build_column_transformer` -- a configurable `ColumnTransformer`: median/mode imputation matched to each column's missingness verdict, plus a swappable encoder/scaler pair picked from `config.yaml` (`_ENCODER_REGISTRY`/`_SCALER_REGISTRY`) instead of hardcoded.

**`config.yaml`** gained two sections: `diagnostics` (placeholder tokens, canonical spelling maps, domain-rule ranges, bucket-consistency rules, redundant columns to drop) and `preprocessing` (which columns are numeric/categorical/MNAR, imputation strategy, encoder/scaler choice).

**`experiments/encoder_scaler_grid.py`** is new and sits outside the production pipeline. It scores all 16 encoder x scaler combinations across 15 repeated splits with a `LogisticRegression`, then runs a paired comparison on the top 2 to check whether the winner's edge is real or just noise. See "Results" below for what came out of it.

### The verdict table -- what was wrong with the data, and what was done about it

| Column | Issue | Mechanism | Action |
|---|---|---|---|
| `age` | missing (~2.0%) | MCAR | median impute, no indicator |
| `juv_fel_count` | missing (~3.0%) | MCAR | median impute, no indicator |
| `priors_count` | missing (~7%, incl. placeholders) | MNAR | median impute + `priors_count_was_missing` |
| `c_charge_degree` | missing (~3.2%) | MNAR | mode impute + `c_charge_degree_was_missing` |
| `race` | missing (~1%, placeholders) | MCAR | mode impute, no indicator (excluded from features anyway) |
| `sex` | missing (~1.5%, incl. placeholders) | MCAR | mode impute, no indicator |
| `age` | invalid (age < 18 or > 100) | domain rule | -> `NaN` before imputation |
| `decile_score` | invalid (outside 1-10) | domain rule | -> `NaN` (excluded from features, fixed for the COMPAS-comparison metric) |
| `juv_fel_count` | invalid (negative) | domain rule | -> `NaN` before imputation |
| `priors_count` | invalid (> 60 or placeholder token) | domain rule | -> `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent spelling | data entry | canonicalized to one label per category |
| `age_cat` vs. `age` | 6 rows disagree (e.g. age=59 but `age_cat="Less than 25"`) | cross-column inconsistency | `age_cat` -> `NaN`, `age` trusted |
| `score_text` vs. `decile_score` | 10 rows disagree (e.g. score=9 but `score_text="Low"`) | cross-column inconsistency | `score_text` -> `NaN`, `decile_score` trusted |
| (whole row) | 72 exact-duplicate rows | data entry | dropped, first occurrence kept |
| `prior_offenses`, `age_in_months`, `juvenile_total` | perfectly/near-perfectly redundant (multicollinearity: r=1.00 with `priors_count`/`age`; VIF unsolvable for the 4 juvenile columns together) | multicollinearity | columns dropped |

### Results -- did the new preprocessing actually help?

| Model | Pipeline | Train acc. | Test acc. | Gap (overfitting) | F1 (class 1, recidivism) | FP | FN |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Week 2 (dropna + one-hot) | 0.679 | 0.678 | +0.001 | 0.63 | 176 | 227 |
| Logistic Regression | Week 3 (impute + `target` encoding + standard scaling) | 0.675 | 0.657 | +0.018 | 0.56 | 161 | 338 |
| Decision Tree | Week 2 (dropna + one-hot) | 0.829 | 0.628 | +0.201 | 0.54 | 175 | 295 |
| Decision Tree | Week 3 (impute + `target` encoding + standard scaling) | 0.792 | 0.613 | +0.178 | 0.51 | 203 | 358 |

FP/FN are relative to the positive class (`two_year_recid = 1`, actually reoffended): FP = predicted "will reoffend" but didn't, FN = predicted "won't reoffend" but did.

Test accuracy did not improve -- it dropped slightly for both models, and F1 on the recidivism class (the harder, more practically relevant class) dropped too (Logistic: 0.63 -> 0.56; Decision Tree: 0.54 -> 0.51). What did improve: the Decision Tree's train-test gap shrank (0.201 -> 0.178), a sign of somewhat less overfitting, and the pipeline itself became more structured: no rows thrown away for having a missing value (imputed instead), no target leakage into imputation/encoding/scaling (fit only on the training fold), and the encoder/scaler choice is backed by evidence instead of a guess (see `experiments/encoder_scaler_grid.py`: `target` encoding beat `onehot`/`ordinal`/`count` consistently across 15 repeated splits; the scaler choice made no statistically real difference. A paired comparison between the top 2 combinations came out as noise, SE larger than the observed gap.

In conclusion, better data cleaning did not lead to a better score. That suggests the model itself (a plain Logistic Regression / Decision Tree) is what's limiting accuracy right now, not the data. So more time spent tweaking preprocessing probably won't help much. Maybe trying a stronger model next is more likely to.

## Model Comparison - Week 2
Logistic Regression achieves a train accuracy of 0.679 and a test accuracy of 0.678, nearly identical, meaning it performs consistently on both seen and unseen data. The Decision Tree achieves a train accuracy of 0.829 but a test accuracy of only 0.628, which is a clear sign of overfitting (it memorizes the training data instead of learning generalizable patterns, and its performance collapses on new data). Since the test accuracy and the near-zero train-test gap is what actually reflects real-world performance, Logistic Regression was the better model in week 2. With week 3's preprocessing, the gap narrows for the Decision Tree but test accuracy drops slightly for both models (see the Results table above) -- Logistic Regression remains the current best model on the train/test-gap criterion, though the difference in test accuracy between the two is now smaller than the noise seen in the encoder/scaler grid.

# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── data_diagnostics.py   # missingness mechanism, domain-rule validity, duplicate + bucket-consistency checks
│   ├── preprocessing.py      # cleaning + missingness flags + leak-safe split + imputation/encoding/scaling
│   ├── model.py               # model construction
│   ├── evaluate.py           # accuracy metrics + fairness check
│   └── results.py            # saves each run's report to disk
├── experiments/
│   └── encoder_scaler_grid.py # one-off: picks encoder/scaler via an empirical grid, not a guess
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA diagnosis & leak-safe preprocessing | `src/data_diagnostics.py` (missingness mechanism test, domain-rule validity, placeholder detection, cross-column bucket-consistency check, duplicate check); `clean_dataset` replaces guesswork with the diagnosis's verdict table (placeholders/invalid values/inconsistent categories -> `NaN`, category spelling cleanup, exact-duplicate removal without blindly trusting repeated ids, redundant-column drop); `add_missing_flags` adds `_was_missing` indicators for MAR/MNAR columns; row-dropping replaced by median/mode imputation, fit only on the training fold; a configurable, leak-safe `ColumnTransformer` (`build_column_transformer`) replaces naive one-hot encoding, with the encoder/scaler pair picked via an empirical grid (`experiments/encoder_scaler_grid.py`) instead of a guess |

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
