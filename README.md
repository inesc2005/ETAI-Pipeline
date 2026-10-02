**Name: Inês Calado**
**Student Number: 20260537**

## Week 4 -- holdout vs. cross-validation, a locked test set, and a better pipeline

Before starting week 4, I corrected a few mistakes and unclear explanations I had in the week 3 section of this README.

### What was added, and where

- **`src/model.py`** -- two new models: `dummy` (always predicts the majority class: the floor every real model must beat) and `random_forest`.
- **`src/preprocessing.py`**
  - `make_pipeline(model_config, preprocessing_config)` joins the preprocessing and the model into one leak-safe `Pipeline`, so it is re-fit on whichever rows it trains on (a holdout's training part, or each CV fold).
  - The `target` encoder is now sklearn's `TargetEncoder`, which cross-fit: each training row is encoded with category means computed on the *other* internal folds, never on its own label. The previous `category_encoders` version leaked a row's own target into its own feature, which inflated train accuracy (random forest: train 0.79 -> 0.73, train-test gap +0.150 -> +0.082 after the change).
  - - `drop_duplicate_rows` is a separate step, used only on the training data (it drops exact duplicate rows and repeated ids, keeping the first). It runs before the split, so the same person cannot end up in both the development set and the test set. `clean_dataset` now keeps every row.
  - `split_dev_test` replaces `split_train_test`: it sets a locked test set aside (`test_set` in `config.yaml`). Nothing is fitted, tuned, compared or chosen on it.
  - `numeric_impute_strategy: "knn"` adds a `KNNImputer` option (the scaler runs first, since KNN uses distances).
- **`src/evaluate.py`** -- `cross_validate_pipeline` (stratified k-fold CV of the whole pipeline, returns per-fold train/validation scores and out-of-fold predictions), `cv_report` (mean and std), `oof_classification_report`. The fairness check now runs on the out-of-fold predictions of the development set.
- **`main.py`** -- set the test set aside -> CV on the development set -> out-of-fold reports -> refit on the development set. The test set is not evaluated here.
- **`config.yaml`** -- new `test_set` (size 0.2, seed 42, never to be changed) and `cv` (5 stratified folds, shuffle, seed 42, accuracy) sections; `model` now holds the chosen random forest.
- **`experiments/compare_models.py`** -- runs the four models and writes the two comparison tables below.
- **`experiments/improve_pipeline.py`** -- goal 3: one change at a time, all scored with the same CV folds (paired comparison), compares each variant with the baseline.
- **`experiments/final_test.py`** -- the one-off assessment on the locked test set.

### 1. Week 4 preprocessing with a holdout, compared with weeks 2 and 3

Holdout = one stratified split. Weeks 2-3: 80/20 train/test (week 2: `dropna` + one-hot; week 3: imputation + `target` encoder + scaler). Now: the week 4 preprocessing inside the Pipeline (cross-fitted `target` encoder, repeated ids dropped, flags built in `split_features_target`), on a 75/25 train/validation split *of the development set* (the test set stays locked, so this is not exactly the same split as weeks 2-3).

| Model | Wk 2 test accuracy | Wk 2 accuracy gap | Wk 3 test accuracy | Wk 3 accuracy gap | Now train accuracy | Now validation accuracy | Now accuracy gap |
|---|---|---|---|---|---|---|---|
| Dummy (majority) | - | - | - | - | 0.549 | 0.550 | 0.000 |
| Logistic Regression | 0.678 | +0.001 | 0.657 | +0.018 | 0.676 | 0.673 | +0.003 |
| Decision Tree | 0.628 | +0.201 | 0.613 | +0.179 | 0.685 | 0.592 | +0.093 |
| Random Forest | - | - | - | - | 0.738 | 0.656 | +0.083 |

Compared with week 3, the main change is the Decision Tree's overfitting (train-test accuracy gap 0.179 -> 0.093), which comes from the cross-fitted `target` encoder; Logistic Regression's validation accuracy is back at the week 2 level (0.673) with a small gap. The validation numbers are not strictly comparable with the week 2-3 test numbers, because they are measured different: weeks 2-3 used an 80/20 split of all the data (the 20% test part is now our locked test set, so it cannot be reused here), while now the validation part is a 25% split taken inside the development set. A holdout moves with the people it happens to contain (Logistic Regression ranges from 0.667 to 0.692 just by changing the seed, see section 2 below), so differences of 0.01-0.02 between weeks can be split noise. What can be compared safely is the gap (train minus test/validation, which measures overfitting) and the order of magnitude of the accuracy. The spread over 10 seeds can be used as a noise band: with the current pipeline, the validation accuracy is 0.677 +/- 0.008 for Logistic Regression and 0.599 +/- 0.010 for the Decision Tree (mean +/- std). Logistic Regression's week 3 test accuracy (0.657) is about 2.5 stds below that, so there is a small real improvement over week 3 (and it is back at the week 2 level, 0.678); the Decision Tree's accuracy (week 3: 0.613) did not change beyond the noise, and its gain is in the gap (0.179 -> 0.093). All models beat the dummy (0.55).

### 2. Cross-validation, before vs. after

| Model | Holdout accuracy (1 split) | Holdout accuracy range (10 seeds) | CV mean accuracy | CV accuracy std | CV train accuracy | CV accuracy gap |
|---|---|---|---|---|---|---|
| Dummy (majority) | 0.550 | 0.550 - 0.550 | 0.549 | 0.000 | 0.549 | 0.000 |
| Logistic Regression | 0.673 | 0.667 - 0.692 | 0.672 | 0.013 | 0.675 | 0.003 |
| Decision Tree | 0.592 | 0.579 - 0.611 | 0.614 | 0.014 | 0.696 | 0.082 |
| Random Forest | 0.656 | 0.637 - 0.665 | 0.649 | 0.019 | 0.732 | 0.083 |

A single holdout is one number that moves with the seed (the Decision Tree ranges from 0.579 to 0.611 just by changing the split). Cross-validation uses every development row for validation once, and gives the spread (std) that a single split hides. Logistic Regression is the best of the four here; the untuned Decision Tree and Random Forest overfit (train-validation accuracy gap ~0.08).

### 3. Can we get a better pipeline?

This comparison was run with `experiments/improve_pipeline.py`. Each variant changes one thing from the baseline (Logistic Regression, median imputation, `target` encoder, `standard` scaler); the same CV folds are used throughout, so "diff" is a paired difference and is only meaningful if it is larger than its standard error (SE).

| Variant | Mean accuracy (cross-validation) | Accuracy diff vs. baseline | SE of accuracy diff |
|---|---|---|---|
| **Baseline (LR)** | 0.672 | - | - |
| Imputer: KNN (k=5) | 0.668 | -0.004 | 0.002 |
| Imputer: mean | 0.672 | -0.000 | 0.000 |
| Encoder: onehot | 0.670 | -0.002 | 0.003 |
| Encoder: ordinal | 0.669 | -0.003 | 0.004 |
| Encoder: count | 0.674 | +0.002 | 0.002 |
| Scaler: robust | 0.672 | -0.000 | 0.000 |
| No missing-value flags | 0.677 | +0.005 | 0.001 |
| LR: `C=0.1` | 0.677 | +0.005 | 0.002 |
| LR: `class_weight=balanced` | 0.669 | -0.003 | 0.003 |
| Decision Tree: default | 0.614 | -0.058 | 0.005 |
| Decision Tree: `max_depth=5` | 0.674 | +0.002 | 0.009 |
| Decision Tree: `min_samples_leaf=50` | 0.672 | -0.001 | 0.007 |
| Random Forest: default | 0.649 | -0.023 | 0.008 |
| Random Forest: `max_depth=6` | 0.681 | +0.009 | 0.004 |
| Random Forest: `min_samples_leaf=20` | 0.681 | +0.009 | 0.005 |
| **Random Forest: `max_depth=6`, `min_samples_leaf=20`** | **0.682** | **+0.010** | 0.005 |

- **Preprocessing changes hardly matter.** The KNN imputer was slightly *worse* than the median; encoder and scaler changes are within the noise. The missing-value flags and a stronger regularisation help a little (+0.005).
- **What matters is limiting model complexity.** The untuned Decision Tree and Random Forest overfit; capping depth and leaf size removes the train-validation accuracy gap (0.083 -> 0.007) and makes the Random Forest the best model.
- **Final model:** Random Forest (`n_estimators=300`, `max_depth=6`, `min_samples_leaf=20`), mean accuracy 0.682 +/- 0.015 (cross-validation). The gain over Logistic Regression (+0.010, SE 0.005) is small: "slightly better"
- **Locked test set, evaluated once** (`experiments/final_test.py`, model fitted on the whole development set): accuracy **0.670**, vs. 0.550 for the dummy and 0.654 for COMPAS's own score on the same rows. It falls inside the CV spread (0.682 +/- 0.015), as it should if the CV estimate was honest.
- **Locked test set, evaluated once** (`experiments/final_test.py`): the final model was fitted on the whole development set and then scored on the test set, which had never been used for training, tuning or choosing anything. Accuracy: **0.670**, against 0.550 for the dummy and 0.654 for COMPAS's own score on the same people. This agrees with what cross-validation had predicted (0.682 +/- 0.015): the test result falls inside that range, which suggests the cross-validation estimate was honest and not too optimistic.

**Next steps.** Each hyperparameter was tried with a single value (Decision Tree: `max_depth=5`, `min_samples_leaf=50`; Random Forest: `max_depth=6`, `min_samples_leaf=20`), only compared with the default (no limit). A natural improvement is to try several values of each one (`max_depth` = 3, 5, 8, 12) and keep the best one.


## Week 3 -- diagnosis-driven preprocessing

### What was added, and where

**`src/data_diagnostics.py`** (new) holds the diagnostic functions. They only report on the data, nothing here changes a value:
- `cramers_v` -- how strongly two categorical columns are associated (0 = unrelated, 1 = perfectly related).
- `test_missingness_mechanism` -- tests a column's missing-value indicator against candidate predictors, (MCAR vs. MAR/MNAR).
- `flag_invalid_values` -- checks domain rules (`18 <= age <= 100`) and turns into `NaN`.
- `find_placeholder_rows` -- flags cells holding a placeholder token (`"-"`, `"?"`, ...) instead of a real value.
- `find_bucket_mismatches` -- flags rows where a category label disagrees with the bucket its own numeric value falls into, e.g. `age_cat` disagreeing with `age`, or `score_text` disagreeing with `decile_score`.
- `find_duplicates` -- also reporting only. Counts and returns rows two ways: exact full-row matches, and rows sharing the same id. Those can differ -- a repeated id whose other columns don't match isn't an exact duplicate, and `clean_dataset` only drops the second kind.

**`src/preprocessing.py`** now owns the whole "raw data in, model-ready split out" job:
- `canonicalize_categories` -- merges spelling/casing variants of a category into one label (`"MALE"`/`"male"`/`" Male"` all become `"Male"`).
- `clean_dataset` -- runs the full cleanup: placeholder tokens to `NaN`, domain-rule violations to `NaN`, category canonicalization, the cross-column bucket check above, duplicate removal, then drops the redundant columns.
- `add_missing_flags` -- adds a `<col>_was_missing` flag for every MAR/MNAR column before imputing it, so the model can still pick up on the missingness pattern even though the fill value is just a median or mode.
- `split_features_target` -- splits into `(X, y, extras)`. `y` comes from `df.get(target)`, so it returns `None` instead of raising when there's no label column.
- `split_train_test` -- the leak-safe boundary. Everything after this point (imputation, encoding, scaling) only ever gets fit on the training fold.
- `build_column_transformer` -- a configurable `ColumnTransformer`: median/mode imputation matched to each column's missingness verdict, plus a encoder/scaler pair picked from `config.yaml` (`_ENCODER_REGISTRY`/`_SCALER_REGISTRY`) instead of hardcoded.

**`config.yaml`** gained two sections: `diagnostics` (placeholder tokens, canonical spelling maps, domain-rule ranges, bucket-consistency rules, redundant columns to drop) and `preprocessing` (which columns are numeric/categorical/MNAR, imputation strategy, encoder/scaler choice).

**`experiments/encoder_scaler_grid.py`** is new and sits outside the production pipeline. It scores all 16 encoder x scaler combinations across 15 repeated splits with a `LogisticRegression`, then runs a paired comparison on the top 2 to check whether the winner's edge is real or just noise. See "Results" below for what came out of it.

### The verdict table -- what was wrong with the data, and what was done about it

| Column | Issue | Mechanism | Action |
|---|---|---|---|
| `age` | missing | MCAR | median impute, no indicator (extra column) |
| `juv_fel_count` | missing | MCAR | median impute, no indicator |
| `priors_count` | missing (incl. placeholders) | MNAR | median impute + `priors_count_was_missing` |
| `c_charge_degree` | missing | MNAR | mode impute + `c_charge_degree_was_missing` |
| `race` | missing (placeholders) | MCAR | mode impute, no indicator (excluded from features anyway) |
| `sex` | missing (incl. placeholders) | MCAR | mode impute, no indicator |
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

The positive class is "will reoffend" (`two_year_recid = 1`). TP = predicted "will reoffend" and the person did (actual 1), TN = predicted "won't reoffend" and the person didn't (actual 0), FP = predicted "will reoffend" but the person didn't (actual 0), FN = predicted "won't reoffend" but the person did (actual 1).

Test accuracy did not improve. It dropped slightly for both models, and F1 on the recidivism class (the harder, more practically relevant class) dropped too (Logistic: 0.63 -> 0.56; Decision Tree: 0.54 -> 0.51). What did improve: the Decision Tree's train-test gap (0.201 -> 0.178), a sign of somewhat less overfitting, and the pipeline itself became more structured: no rows thrown away for having a missing value (imputed instead), no target leakage into imputation/encoding/scaling (fit only on the training fold), and the encoder/scaler choice is backed by evidence instead of a guess (see `experiments/encoder_scaler_grid.py`: `target` encoding beat `onehot`/`ordinal`/`count` consistently across 15 repeated splits); the scaler choice made no statistically real difference. A paired comparison between the top 2 combinations came out as noise, SE larger than the observed gap.

In conclusion, better data cleaning did not lead to a better score. That suggests the model itself (a plain Logistic Regression / Decision Tree) is what's limiting accuracy right now, not the data. So more time spent preprocessing probably won't help much. Maybe trying a stronger model next is more likely to.
ange: with these features (age, sex, priors, charge) recidivism is inherently noisy, which is also why COMPAS itself only reaches 0.65.


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
│   ├── evaluate.py           # holdout metrics, cross-validation (week 4) + fairness check
│   └── results.py            # saves each run's report to disk
├── experiments/
│   ├── encoder_scaler_grid.py # week 3: picks encoder/scaler via an empirical grid (single splits, pre-CV)
│   ├── compare_models.py     # week 4: 4 models, week 3 vs. now + holdout vs. CV tables
│   ├── improve_pipeline.py   # week 4: one change at a time, compared with the same CV folds
│   └── final_test.py         # week 4: one-off assessment on the locked test set
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
| 4 | Holdout vs. cross-validation, locked test set | `dummy` and `random_forest` models; `make_pipeline`; `split_dev_test` replaces `split_train_test` (locked test set, `test_set` in the config); `drop_duplicate_rows` (training-only, before the split); sklearn's cross-fitting `TargetEncoder`; stratified k-fold CV of the whole pipeline (`cross_validate_pipeline`, `cv_report`, `oof_classification_report`, `cv` in the config) with out-of-fold reports; KNN imputer option; `experiments/compare_models.py`, `improve_pipeline.py` and `final_test.py`; limited-depth Random Forest chosen as the final model |

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
