**Name: Inês Calado**  
**Student Number: 20260537**

## Week 8 -- XGBoost and feature selection

The goal: add XGBoost, and test whether a model that uses fewer features is as good as one that uses all of them.

### What was added, and where

- **`requirements.txt`** -- `xgboost>=2.0`.
- **`src/model.py`**
  - `xgboost` in `_MODELS`: gradient boosting with extra brakes inside every split (`gamma`, `reg_alpha`, `reg_lambda`, `min_child_weight`).
  - New `VoteSelector`: a random forest and gradient boosting vote for the features with above-average importance, an L1 logistic regression votes for the features it does not shrink to zero, and a feature stays if it gets at least `min_votes` votes.
  - `build_pipeline()` adds the selector as a `"select"` step between preprocessing and the model when it is switched on. It also takes a cache folder (`memory`), so tuning does not refit preprocessing and selection in every trial.
- `config.yaml`: `model` is now XGBoost; new `feature_selection` section (on/off, `min_votes`, and `select__min_votes` from 1 to 3 as a search space); `xgboost` search space.
- `main.py`: passes the selection settings and a cache folder (deleted at the end of the run) to the pipeline; tunes `select__min_votes` together with the model; prints the features kept; adds a `feature_selection` column to the scoreboard.
- `src/results.py`: the report header shows the selection setting.

The selector learns from the labels, so it has to be a step of the `Pipeline`: it is refitted on the training rows of each fold and never sees the rows it is scored on. `add_engineered_features()` (week 6) is different: it uses fixed formulas, so it stays before the split.

### Feature selection

XGBoost, tuned with nested CV (30 trials, 5 inner folds, the same 5 outer folds as the scoreboard of week 7). Selection metric: ROC-AUC. Source: `results/runs.csv`.

| Run | Untuned ROC-AUC | Tuned: nested CV ROC-AUC (mean ± std) | Floor | Features kept (final model) |
|---|---|---|---|---|
| Without selection | 0.724 | 0.727 ± 0.014 | 0.500 | 10 of 10 |
| With selection | 0.720 | 0.725 ± 0.012 | 0.500 | 4 of 10: `age`, `juv_fel_count`, `juv_other_count`, `priors_count` |

| Corrected paired t-test (ROC-AUC) | Mean difference | Folds won | p-value |
|---|---|---|---|
| With selection against without | -0.002 | 2/5 | 0.57 |

The model with 4 features scores about the same as the one with all 10. The difference is only -0.002, much smaller than the variation between folds (±0.012 with selection, ±0.014 without), and p = 0.57 means it could easily be chance and noise. The model with selection also won only 2 of the 5 folds, so it is not consistently better or worse. Both are far above the floor (0.500), and tuning raised both a little (0.724 -> 0.727 without selection, 0.720 -> 0.725 with it). The 4 features kept are `age`, `juv_fel_count`, `juv_other_count` and `priors_count`; the other 6 were dropped. That doesn't prove the two models are identical, only that 5 folds show no real loss. 

---

## Week 7 -- gradient boosting, and evaluating with probabilities

The goal: add the boosting models, and judge every model on several metrics instead of just accuracy, with a scoreboard that tracks all runs for comparison.

### What was added, and where

- **`src/model.py`**: three new model types in `_MODELS`, straight from scikit-learn (no builder function needed): `gradient_boosting` (`GradientBoostingClassifier`: shallow regression trees added one at a time, each one fitted to the gradient of the log-loss left by the previous ones), `hist_gradient_boosting` (`HistGradientBoostingClassifier`: the same algorithm on binned features, faster on large data and able to handle missing values) and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline).
- **`config.yaml`**
  - `model` is now gradient boosting (`random_state: 42`); the other models' default `params` are comments, with `class_weight: balanced` for the models that support it.
  - Search spaces for the three new models. For the boosters, the number of trees, the learning rate and the tree size are searched together, because they trade off against each other 
  - In `cv`, `scoring: "accuracy"` is replaced by `metrics` (8 metrics: accuracy, balanced accuracy, precision, recall, F1, F-beta 2, ROC-AUC, average precision) and `selection_metric: "roc_auc"`.
  - `output.scoreboard: "results/runs.csv"`.
- **`src/evaluate.py`**
  - new `build_scorers()`: one scorer per metric (`fbeta_2` is built with `make_scorer`, beta = 2).
  - `cross_validate_pipeline()` scores every metric and returns out-of-fold probabilities; `cv_report()` shows the mean and std of every metric.
  - `oof_classification_report()` and `fairness_report()` take the probabilities and a threshold (0.5); the classification report now includes the confusion matrix.
  - `corrected_paired_test()` (from week 6, which I had not added yet): the corrected paired t-test (Nadeau & Bengio) to compare two models scored on the same folds. `scipy` is now listed in `requirements.txt` (it already came with scikit-learn).
- **`src/tuning.py`** : `tune_pipeline()` takes the selection metric's scorer; `nested_cross_validate()` scores every metric on each outer fold and returns probabilities; `tuning_report()` uses the selection metric.
- **`src/results.py`** new `save_scores()`: every run adds one row to `results/runs.csv` (model, settings, the selection metric's mean, std and score on each fold, the untuned score and the no-model floor - dummy model). `save_run()` names the report with the same run id and lists the metrics in its header.
- **`main.py`** : builds the scorers once; scores the no-model floor (a dummy) on the same folds; prints a short summary (all metrics, chosen hyperparameters) while the full tables go to the results file; saves the run on the scoreboard and prints it.
- `experiments/compare_models.py` and `improve_pipeline.py` were adapted to the new `cross_validate_pipeline()` and still use accuracy (they reproduce the week 4 tables).

### Selection metric

I use ROC-AUC as the selection metric. COMPAS is a risk score used to rank people, and ROC-AUC measures exactly that: the probability that a random person who reoffended gets a higher score than a random person who did not, over every threshold at once, so the choice of the yes/no threshold stays a separate decision. It is also often less noisy than accuracy, which only counts on which side of 0.5 each person falls: for gradient boosting vs. decision tree, the corrected test gives p = 0.01 on ROC-AUC and p = 0.10 on accuracy, on the same folds (not in every pair, though: against Naive Bayes both are clearly significant).

### Scoreboard

`python main.py` once per model, tuned with nested CV (30 trials, 5 inner folds, the same 5 outer folds for every model; the dummy has nothing to tune). 
Untuned = the model with its default `params` from `config.yaml`; 
floor = the dummy on the same folds. Source: `results/runs.csv`.

| Model | Untuned ROC-AUC | Tuned: nested CV ROC-AUC (mean ± std) | Floor | Chosen hyperparameters |
|---|---|---|---|---|
| Gradient boosting | 0.727 | 0.728 ± 0.015 | 0.500 | `n_estimators=208`, `learning_rate=0.012`, `max_depth=4`, `min_samples_leaf=1`, `subsample=0.55` |
| HistGradientBoosting | 0.717 | 0.728 ± 0.015 | 0.500 | `max_iter=302`, `learning_rate=0.014`, `max_leaf_nodes=10`, `min_samples_leaf=62`, `l2_regularization=0.0003` |
| Oblique random forest | 0.685 | 0.727 ± 0.015 | 0.500 | `max_depth=13`, `min_samples_leaf=151`, `n_components=10` |
| Random forest | 0.682 | 0.726 ± 0.014 | 0.500 | `max_depth=7`, `min_samples_leaf=3`, `max_features=sqrt` |
| AdaBoost | 0.723 | 0.726 ± 0.016 | 0.500 | `max_depth=4`, `n_estimators=55`, `learning_rate=0.167` |
| Logistic regression | 0.719 | 0.718 ± 0.014 | 0.500 | `C=0.44` |
| Decision tree | 0.607 | 0.718 ± 0.016 | 0.500 | `max_depth=6`, `min_samples_leaf=104`, `criterion=gini` |
| Naive Bayes | 0.685 | 0.703 ± 0.011 | 0.500 | `var_smoothing=0.091` |
| Dummy (floor) | 0.500 | -- | 0.500 | -- |

Defaults: logistic regression, decision tree, random forest and HistGradientBoosting with `class_weight: balanced`; gradient boosting, AdaBoost, oblique random forest and Naive Bayes have no `class_weight` here.

The best model against each of the others (corrected paired t-test on the 5 outer folds, ROC-AUC:

| Gradient boosting against | Mean difference | Folds won | p-value |
|---|---|---|---|
| HistGradientBoosting | +0.000 | 4/5 | 0.27 |
| Oblique random forest | +0.001 | 4/5 | 0.76 |
| Random forest | +0.002 | 4/5 | 0.48 |
| AdaBoost | +0.002 | 3/5 | 0.49 |
| Logistic regression | +0.010 | 5/5 | 0.08 |
| Decision tree | +0.010 | 5/5 | 0.01 |
| Naive Bayes | +0.025 | 5/5 | 0.01 |

Every model is far above the floor (0.500). The five tree ensembles (both boosters, both forests and AdaBoost) score almost the same, 0.726-0.728, and the p-values for comparing them (0.27-0.76) are too high to call a winner: their small differences are not reliable enough to trust. Against the single decision tree and Naive Bayes, the best ensemble does show a real, trustworthy lead (p = 0.01). Against logistic regression it is probably also better (it wins in all 5 folds), but the lead is not quite reliable enough to be sure (p = 0.08, just above the usual 0.05 cut-off). Tuning helped most where the defaults overfit (decision tree 0.607 -> 0.718, the two forests about +0.04, also table above), and changed almost nothing for gradient boosting, AdaBoost and logistic regression, which were already well regularised.

ROC-AUC says nothing about the 0.5 yes/no cut-off; for gradient boosting, at that threshold (nested CV):

| Accuracy | Precision | Recall |
|---|---|---|
| 0.683 | 0.680 | 0.560 |

With a recall of only 0.560, it misses about 44% of the people who actually reoffended, a consequence of using 0.5 as the cut-off. ROC-AUC never picks that cut-off, it only judges the ranking, so choosing (and justifying) a threshold is left as a separate decision.

---

## Week 5 and 6 -- hyperparameter tuning, and ensembles

## Part 1 -- continuing week 4: hyperparameter tuning without leakage

The goal: compare each model with its **default** hyperparameters against the **tuned** version, with an honest number for the tuned one.

### What was added, and where
 
- **`src/tuning.py`** (new)
  - `tune_pipeline` -- an Optuna study (seeded TPE sampler). Each trial reads its ranges from the search space in `config.yaml`, sets the values on a fresh, unfitted copy of the whole pipeline (`clone` + `set_params`) and scores it by its mean CV accuracy. Returns the best pipeline (unfitted) and the study.
  - `nested_cross_validate` -- the honest estimate of "tune, then refit": for each outer fold, the tuning runs on the outer-training rows only and the winner is scored on the outer-validation rows, which took no part in choosing it. Same output format as `cross_validate_pipeline`, plus `inner_best` (the tuning score inside that fold) and the hyperparameters chosen.
  - `tuning_report` -- best hyperparameters, top trials, and the optimism check (mean `inner_best` vs. mean outer validation).
- **`src/model.py`** -- `build_pipeline(preprocessing_config, model_config)` replaces `make_pipeline` (which moved out of `src/preprocessing.py`): one place builds the preprocessing + model `Pipeline` for `main.py`, tuning and the experiments.
- **`main.py`** -- after the usual CV (hyperparameters from `config.yaml`), if `tuning.enabled`: nested CV (its out-of-fold predictions now feed the classification report and fairness check), then tuning on all development rows, then the tuned pipeline is refit on all development rows.
- **`config.yaml`** -- new `tuning` section (on/off, `n_trials`, inner `n_splits`, `random_state`, one search space per model type: decision tree, logistic regression, random forest); `model` is now the decision tree with default hyperparameters.
- **`src/results.py`** -- the run header records the tuning settings.
- **`requirements.txt`** -- `optuna>=4.0`.

### Hyperparameter tuning

**How it works.** Hyperparameters are chosen with **Optuna** (TPE sampler, seed 42): 30 trials for the decision tree and logistic regression, 15 for the random forest (each trial is much slower, so the budget was halved, following the suggestion of the class notebook). `n_trials` is a single value in `config.yaml`: to reproduce a row, set `model` and `n_trials` as described in the comments of `config.yaml` (30 for the first two rows, 15 for the random forest). Every trial is scored by stratified 5-fold CV of the **whole** pipeline on the development set (inner folds, seed 42), so imputer, encoder and scaler are re-fit in every fold of every trial, and every candidate is scored on the same folds. Search spaces: decision tree `max_depth` 1-20, `min_samples_leaf` 1-200 (log), `criterion`; logistic regression `C` 0.0001-100 (log); random forest `n_estimators` 50-200, `max_depth` {3, 5, 8, 12, 20, None}, `min_samples_leaf` 1-200 (log), `max_features` {sqrt, log2, 0.5, 1.0}.

**Why it doesn't leak.** The test set stays locked (tuning sees only the development set); preprocessing is inside the `Pipeline`, so it is never fitted on validation rows; and the score reported for the tuned model is not the best trial's score (the maximum of many noisy scores, so optimistic) but the **nested CV** one, where each outer fold is scored by a model whose tuning never saw it.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Optimism | Chosen hyperparameters |
|---|---|---|---|---|---|
| Decision tree | 0.614 ± 0.014 (+0.082) | 0.678 ± 0.020 (+0.007) | 0.680 | -0.002 | `max_depth=6`, `min_samples_leaf=110`, `criterion=entropy` |
| Logistic regression | 0.672 ± 0.013 (+0.003) | 0.674 ± 0.014 (+0.004) | 0.678 | +0.003 | `C=0.079` |
| Random forest | 0.649 ± 0.019 (+0.084) | 0.680 ± 0.015 (+0.010) | 0.682 | +0.001 | `n_estimators=96`, `max_depth=12`, `min_samples_leaf=36`, `max_features=0.5` |

Defaults: sklearn defaults (decision tree `random_state=42`; logistic regression `max_iter=2000`; random forest `n_estimators=200`). Optimism = mean `inner_best` minus mean nested validation. Full runs: `results/week5_*.txt`.

The models that gained from tuning are the two that overfit with their defaults: the decision tree (+0.064) and the random forest (+0.031), both well above the fold-to-fold std (~0.015-0.020), and both lose almost all of their train-validation gap (0.08 -> ~0.01), reducing overfitting. The chosen trees are limited in size (large leaves of 36-110 people), which matches the week 4 finding that complexity was the problem. Logistic regression barely changed (+0.002, far inside the std): it has one main hyperparameter, it was not overfitting, and a linear model has little variance to remove. 
Tuning did change which model is best: by default logistic regression was clearly ahead. After tuning the three models are tied (0.674-0.680, differences smaller than the std), with the random forest slightly ahead. For the random forest I would report the **nested CV score, 0.680 ± 0.015**, not the best trial's 0.682 or week 4's hand-picked 0.682, because those were chosen and scored on the same folds and are optimistic.

**About the optimism.** In theory the tuning score (`inner_best`) should be slightly higher than the honest one (`validation`), because it is the maximum of many noisy scores. Here the optimism is between -0.002 and +0.003, which is smaller than the fold-to-fold noise (std of the nested validation score: 0.014-0.020). So it cannot be clearly told apart from that noise: for the decision tree it even came out negative (-0.002), which does not mean that the tuning underestimated the model, only that the true optimism is too small to be seen with 5 outer folds. This is expected here: the data has 5771 development rows, only 30 trials (15 for the random forest) and the candidates really differ from each other (the depth and leaf size change the score a lot), so the winner wins because it is better and not only because it was lucky. The noise itself is not an advantage: it is a limit of the data, and it is why differences of 0.002-0.006 between models (such as the three tuned models) should not be read as one model being better than another.

## Part 2 -- ensembles 

The goal: add two new ensembles of trees to the pipeline, tune all three ensembles with the Part 1 machinery, and compare them with the models so far on the same folds.

### What was added, and where

- **`src/model.py`** -- two new model types, registered in `_MODELS` (so `model.type` in `config.yaml` can be `"oblique_random_forest"` or `"adaboost"`). They are functions and not scikit-learn classes, because `config.yaml` can only hold numbers and text:
  - `adaboost()` -- AdaBoost over decision trees of depth `max_depth` (1 = stumps): trees built one after another, each focused on the rows the previous ones got wrong, combined by a weighted vote. The function builds the base tree from the number `max_depth`.
  - `oblique_random_forest()` -- a forest of oblique trees (splits on combinations of features,  diagonal lines). scikit-learn has no such model, so it is assembled: `SparseRandomProjection` (random combinations of the features) + a decision tree, inside a `BaggingClassifier` that fits many of them on bootstrap samples and averages them.
- **`config.yaml`** -- `model` is now the random forest (200 trees), with the other models' default `params` as comments; search spaces for the three ensembles. The two forests (random forest and oblique random forest) tune the size of their trees (`max_depth`, `min_samples_leaf`) and how different the trees are (`max_features` / `n_components`), but not the number of trees: in a forest more trees never overfit, they only cost time, so for the random forest and the oblique random forest `n_estimators` is fixed at 200 in `params`. AdaBoost is different: its `n_estimators` (the number of rounds, 25-400) is tuned, together with its base trees' depth (1-8) and its learning rate (0.01-1.0), because in AdaBoost more rounds or stronger trees can overfit. The random forest search space therefore changed from Part 1 (which also tuned `n_estimators`); the Part 1 one is kept as a comment in `config.yaml`.
- `src/tuning.py` and the reports did not change: they work through `build_pipeline()` and the config.
- **`src/preprocessing.py`** and **`main.py`** (Challenge 1) -- new `add_engineered_features()`, called in `main.py` after `drop_duplicate_rows()` and before `split_features_target()` (see "Challenge 1" below).

### Ensembles

Every row comes from `python main.py` with that `model.type` and tuning on (30 trials, 5 inner folds, the same 5 outer folds for every model). The logistic regression and decision tree rows are the Part 1 runs; the three ensembles are new runs (`results/week5_part2_*.txt`, 7-13 minutes each).

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | 0.674 ± 0.014 (+0.004) | 0.678 | `C=0.079` |
| Decision tree | 0.614 ± 0.014 (+0.082) | 0.678 ± 0.020 (+0.007) | 0.680 | `max_depth=6`, `min_samples_leaf=110`, `criterion=entropy` |
| Random forest | 0.649 ± 0.019 (+0.084) | 0.682 ± 0.015 (+0.008) | 0.684 | `max_depth=29`, `min_samples_leaf=61`, `max_features=0.5` |
| Oblique random forest | 0.645 ± 0.015 (+0.118) | 0.682 ± 0.016 (+0.008) | 0.683 | `max_depth=22`, `min_samples_leaf=195`, `n_components=9` |
| AdaBoost | 0.676 ± 0.013 (+0.006) | 0.681 ± 0.014 (+0.006) | 0.684 | `max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

Defaults: random forest and oblique random forest with 200 trees (oblique: 10 combinations per tree); AdaBoost with stumps, 200 rounds, `learning_rate=1.0`. The random forest row differs slightly from Part 1 (0.682 vs 0.680) because of the new search space.

After tuning, the five models are between 0.674 and 0.682: a spread of 0.008, smaller than the fold-to-fold std (0.013-0.020), so no model is clearly better than the others. The random forest and the oblique random forest share the highest mean (0.682); I would choose the random forest as the best one, because the oblique splits add nothing here (same score), and the random forest is a standard scikit-learn model and about twice as fast to tune (7 vs 13 minutes). The chosen hyperparameters match how each model works: the forests chose deep trees (depth 29 and 22) but with large leaves (61 and 195 people), because a forest averages fairly complex trees and it is the leaf size, not the depth, that stops them from memorising single people overfitting; AdaBoost chose shallow trees (depth 4) and a small learning rate (0.025), because boosting wants weak learners and small steps. 
This is also why tuning helped the forests most (random forest +0.033 = 0.682 - 0.649, oblique random forest +0.037 = 0.682 - 0.645; both overfitting by default with gaps of +0.084 and +0.118) and AdaBoost very little (+0.005 = 0.681 - 0.676; its stumps were not overfitting). After tuning, the overfitting is gone where it existed: the gap of the random forest and the oblique random forest drops to +0.008, and all five tuned models have a gap below 0.01. 
To conclude, I would say the best model is the random forest: it is tied with the others within the noise, but it has the highest mean together with the oblique random forest and is the simpler and faster of the two. The number I would report for it is its nested CV score, 0.682 ± 0.015, not its best trial (0.684), which is optimistic.

### Challenge 1 -- feature engineering

**What was done.** `add_engineered_features()` in `src/preprocessing.py` adds new columns built from the existing ones. It is called in `main.py` after `drop_duplicate_rows()` and before `split_features_target()`. Every feature is a fixed formula: nothing is learned from the data (no median, no mean, no target), so it is safe to compute before the split and on new data, like the `_was_missing` flags. (A feature such as "priors above the median" would need the median, so it would have to be a step inside the `Pipeline` instead.) A missing input gives a missing feature, which is imputed later inside the `Pipeline` like any other numeric column. A new column is only used by the model if it is listed in `preprocessing.numeric_features`; the `ColumnTransformer` drops the others.

I tried two sets of features, each together with the class notebook's example:

| Feature | Formula | Idea | Attempt |
|---|---|---|---|
| `priors_per_adult_year` | `priors_count / (age - 17)` | 2 priors at 19 and 2 priors at 45 are very different histories; a tree can't divide one column by another (class notebook example) | 1 and 2 |
| `has_juvenile_record` | 1 if any of `juv_fel_count`, `juv_misd_count`, `juv_other_count` is above 0, else 0 | having a juvenile record at all, as one yes/no answer | 1 |
| `priors_bin` | `priors_count` in fixed bins 0 / 1-3 / 4-9 / 10+, coded 0-3 | none / a few / several / many priors | 1 |
| `felony_with_priors` | `priors_count` if the current charge is a felony (`F`), 0 if a misdemeanour (`M`) | an interaction: the priors only count when the current charge is serious; the model only saw these two columns separately | 2 |
| `log_priors` | `log(1 + priors_count)` | the first priors weigh more than the twentieth; logistic regression can only use `priors_count` as a straight line | 2 |

The `priors_bin` edges are round numbers chosen by hand, after looking at the distribution of `priors_count` (on all rows, so the test set too); they were not tuned on any score.

The two features of attempt 2, with examples:
- **`felony_with_priors`** joins two pieces of information the model only saw separately: the number of priors and the type of the current charge. Two people with 5 priors each get different values: one charged with a felony (serious) gets 5, the other charged with a misdemeanour (minor) gets 0. In other words, it captures people who have many priors and are now facing a serious charge, a combination that may signal a higher risk than either fact on its own. Logistic regression cannot create this combination by itself, since it only adds up the separate effect of each column, so the feature provides it directly.
- **`log_priors`** changes the scale of `priors_count` so that the first priors count more than the later ones: moving from 0 to 1 prior (from no record to a first offence) reveals a lot about a person, whereas moving from 10 to 11 adds very little. `priors_count` treats both steps as the same (+1), while `log_priors` goes 0 -> 0.69 for the first prior but only 2.40 -> 3.04 from 10 to 20 priors. Logistic regression gives every extra prior the same weight; with the log, each extra prior weighs less and less.

**Results.** The table compares the nested CV score of each model with and without the new features. All runs use 30 trials and the same 5 outer folds, so the only difference is the features.
The 'without' scores are the ones from the tables above (the new columns are only used when listed in numeric_features, so adding the function to main.py does not change them).
Full runs: `results/week5_part2_features_v1_*.txt` (attempt 1) and `results/week5_part2_features_v2_*.txt` (attempt 2).

| Model | Without new features | Attempt 1 (`has_juvenile_record`, `priors_bin`) | Attempt 2 (`felony_with_priors`, `log_priors`) |
|---|---|---|---|
| Logistic regression | 0.674 ± 0.014 | 0.675 ± 0.010 (+0.001, better in 3/5 folds) | 0.675 ± 0.012 (+0.001, better in 2/5 folds) |
| Random forest | 0.682 ± 0.015 | 0.674 ± 0.010 (-0.008, better in 1/5 folds) | 0.672 ± 0.009 (-0.009, better in 1/5 folds) |
| AdaBoost | 0.681 ± 0.014 | 0.677 ± 0.011 (-0.004, better in 1/5 folds) | 0.676 ± 0.012 (-0.005, better in 0/5 folds) |

**Which model family gains, and by more than the fold std?** None. Logistic regression is the only one that doesn't lose, but its gain (+0.001) is far below the fold std (0.010-0.014) and the folds go both ways. The tree models (random forest and AdaBoost) got slightly worse in both attempts, in almost every fold; the losses are also smaller than the fold std, so they are not clearly significant, but the direction is consistent. A likely reason is that trees can already build these features themselves with a few splits on `priors_count`, `age` and `c_charge_degree`, so the new columns add no information, only more similar columns, i.e. more ways to fit noise. Since two different kinds of features (indicators/bins and an interaction/a transformation) failed in the same way, the limit seems to be the information in the original columns, not the way it is presented to the models.

**Which set I keep.** Both attempts give the same scores, so the choice is about the idea, not the number: I prefer attempt 2, because `felony_with_priors` combines two columns the model only saw separately, and `log_priors` tests a clear idea for logistic regression; in attempt 1, `has_juvenile_record` is almost the `juvenile_total` column I dropped in week 3 as redundant, and `priors_bin` is only `priors_count` in groups. Since neither set improves the models, none of the new features is listed in `preprocessing.numeric_features`: the final model is still the random forest without them (0.682 ± 0.015). `add_engineered_features()` still computes all five, and the comments in `config.yaml` list them and say how to switch them on.

## Week 4 -- holdout vs. cross-validation, a locked test set, and a better pipeline

Before starting week 4, I corrected a few mistakes and unclear explanations I had in the week 3 section of this README.

### What was added, and where

- **`src/model.py`** -- two new models: `dummy` (always predicts the majority class: the floor every real model must beat) and `random_forest`.
- **`src/preprocessing.py`**
  - `make_pipeline(model_config, preprocessing_config)` joins the preprocessing and the model into one leak-safe `Pipeline`, so it is re-fit on whichever rows it trains on (a holdout's training part, or each CV fold).
  - The `target` encoder is now sklearn's `TargetEncoder`, which cross-fit: each training row is encoded with category means computed on the *other* internal folds, never on its own label. The previous `category_encoders` version leaked a row's own target into its own feature, which inflated train accuracy (random forest: train 0.79 -> 0.73, train-test gap +0.150 -> +0.082 after the change).
  - `drop_duplicate_rows` is a separate step, used only on the training data (it drops exact duplicate rows and repeated ids, keeping the first). It runs before the split, so the same person cannot end up in both the development set and the test set. `clean_dataset` now keeps every row.
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

In conclusion, better data cleaning did not lead to a better score. That suggests the model itself (a plain Logistic Regression / Decision Tree) is what's limiting accuracy right now, not the data. So more time spent preprocessing probably won't help much. Maybe trying a stronger model next is more likely to. Another possible reason: with these features (age, sex, priors, charge) recidivism is inherently noisy, which is also why COMPAS itself only reaches 0.65.


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
│   ├── model.py               # model construction (incl. oblique random forest and AdaBoost, week 6), VoteSelector (week 8) + build_pipeline (preprocessing + feature selection + model)
│   ├── evaluate.py           # cross-validation (week 4), out-of-fold reports + fairness check, several metrics from out-of-fold probabilities + corrected paired test (week 7)
│   ├── tuning.py             # hyperparameter tuning with Optuna + nested cross-validation (week 5)
│   └── results.py            # saves each run's report to disk, and its scores to the scoreboard results/runs.csv (week 7)
├── experiments/
│   ├── encoder_scaler_grid.py # week 3: picks encoder/scaler via an empirical grid (single splits, pre-CV)
│   ├── compare_models.py     # week 4: 4 models, week 3 vs. now + holdout vs. CV tables
│   ├── improve_pipeline.py   # week 4: one change at a time, compared with the same CV folds
│   └── final_test.py         # week 4: one-off assessment on the locked test set
├── results/                # created automatically -- one report per run + runs.csv, the scoreboard of all runs (not tracked in git)
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
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: Optuna (seeded TPE) tunes the model on the development set only, every candidate scored by stratified 5-fold CV of the whole pipeline on the same folds (`tune_pipeline`); **nested cross-validation** (`nested_cross_validate`) gives the honest score of the tuned model, and its out-of-fold predictions feed the classification report and fairness check; `tuning_report` shows the best trial next to the nested score (optimism check); the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` replaces `make_pipeline`; `tuning` section in `config.yaml` (search spaces for decision tree, logistic regression, random forest); `optuna` in `requirements.txt`. The locked test set is still untouched |
| 6 | Ensembles: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`, registered in `_MODELS`: `oblique_random_forest` (a forest of trees that split on random linear combinations of the features, built from `SparseRandomProjection` + a decision tree inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth, stumps by default); the random forest is now the model `main.py` runs (200 trees); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size and decorrelation but not the number of trees (more trees never overfit), AdaBoost tunes its base trees' depth, number of rounds and learning rate; all five models compared with nested CV on the same folds; Challenge 1: `add_engineered_features()` in `src/preprocessing.py` (fixed-formula features, computed before the split), two sets tested with nested CV and left out of `numeric_features` because no model gained (see "Part 2 -- ensembles") |
| 7 | Gradient boosting + evaluating with probabilities | Three new model types in `src/model.py`: `gradient_boosting`, `hist_gradient_boosting` and `naive_bayes`, with search spaces in `config.yaml` (gradient boosting is now the model `main.py` runs); evaluation reports **8 metrics** from out-of-fold **probabilities** (`cv.metrics`) and one **selection metric** (`cv.selection_metric`, ROC-AUC) that tuning maximises and runs are ranked by (`build_scorers()`, new `cross_validate_pipeline()`, `cv_report()`, tuning functions); the classification report adds the confusion matrix; `corrected_paired_test()` compares two models on the same folds (`scipy` in `requirements.txt`); new `save_scores()`: every run adds a row to the **scoreboard** `results/runs.csv` (score on each fold, untuned score, no-model floor); all 9 models run and compared (see "Week 7") |
| 8 | XGBoost + feature selection | `xgboost` in `_MODELS`, now the model `main.py` runs, with its search space in `config.yaml` (`xgboost` in `requirements.txt`); new `VoteSelector` in `src/model.py`: three models vote for the features to keep; `build_pipeline()` adds it as a `"select"` step, so it is refitted inside every fold, and takes a cache folder so tuning does not refit it in every trial; new `feature_selection` section in `config.yaml`, with `select__min_votes` tuned together with the model; `main.py` prints the features kept, and the scoreboard and report header record the selection setting; XGBoost tuned with and without selection and compared with `corrected_paired_test()` (see "Week 8") |

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

This loads `config.yaml`, cleans the data, locks the test set away, cross-validates preprocessing + feature selection (if enabled) + model on the development set on several metrics (with the no-model floor on the same folds), tunes the hyperparameters if `tuning.enabled` is on, and prints a short summary:
- the **selection metric** (`cv.selection_metric`, ROC-AUC) with the hyperparameters from `config.yaml` (not tuned): mean ± std, the train-validation gap (a large gap = overfitting: the model looks much better on the data it was trained on than on data it has never seen) and the no-model floor;
- with tuning on: the honest score of the tuned model (nested cross-validation) and the hyperparameters chosen;
- the mean ± std of **every metric** in `cv.metrics`;
- the **scoreboard**: every run saved in `results/runs.csv` on the same data and metric, best first;
- the final model: the same (tuned) pipeline refit on all development rows. The locked test set is **not** evaluated.
- with feature selection on: the features the final model keeps ("Features kept: ...").

The full tables go to the results file: the per-fold tables, the tuning report (top trials, optimism check), the out-of-fold confusion matrix and classification report, and the false-positive-rate-by-race comparison between our model and COMPAS's own score (same rows).

All of this is saved to a file named after the run in `results/` (e.g. `results/run_20261031_144610.txt`), and one row per run goes to `results/runs.csv`, so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
