"""
Model construction: the model alone (`build_model`), the feature-selection step (`VoteSelector`)
and preprocessing + selection + model as one estimator (`build_pipeline`).
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_column_transformer


#  unlike the other models, these two need a function, because config.yaml only holds numbers and text:
#    adaboost needs its base tree as an object, so the function builds it from `max_depth`;
#    oblique_random_forest doesn't exist in scikit-learn, so the function assembles it.
def adaboost(max_depth: int = 1, **params):
    """
    AdaBoost over decision trees of depth `max_depth` (1 = stumps), built one after another, each focused on the rows the previous ones got wrong. 
    `params` (n_estimators, learning_rate, random_state) go to AdaBoostClassifier.
    The trees' depth is tuned as `model__estimator__max_depth`.
    """
    return AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=max_depth), **params)


def oblique_random_forest(n_estimators: int = 100, n_components: int = 10, density="auto",
                          random_state=None, n_jobs=None, **tree_params):
    """
    Each tree first mixes the features into `n_components` random combinations ("project"), then splits on them ("tree").
    BaggingClassifier fits `n_estimators` of these trees, each on a random sample of rows, and averages them.
    """
    tree_params.setdefault("max_features", "sqrt")
    oblique_tree = Pipeline([
        ("project", SparseRandomProjection(n_components=n_components, density=density)),
        ("tree", DecisionTreeClassifier(**tree_params)),
    ])
    return BaggingClassifier(estimator=oblique_tree, n_estimators=n_estimators,
                             random_state=random_state, n_jobs=n_jobs)


_MODELS = {
    "dummy": DummyClassifier,   # always predicts the most frequent class -- the floor every real model must beat
    "logistic_regression": LogisticRegression,
    "decision_tree": DecisionTreeClassifier,
    "random_forest": RandomForestClassifier,
    "oblique_random_forest": oblique_random_forest,   
    "adaboost": adaboost,
    "gradient_boosting": GradientBoostingClassifier,           # each shallow tree fits the gradient of the log-loss left by the previous ones
    "hist_gradient_boosting": HistGradientBoostingClassifier,  # the same of gradient boosting but with binned features: faster, handles missing values
    "naive_bayes": GaussianNB,                                 # fast probabilistic baseline
    "xgboost": XGBClassifier,                                  # gradient boosting with brakes inside every split (gamma, reg_alpha, reg_lambda, min_child_weight)
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


class VoteSelector(SelectorMixin, BaseEstimator):
    """
    Keeps the features chosen by at least `min_votes` of three models, fitted on the training rows only:
        a random forest and gradient boosting vote for features with above-average importance;
        an L1 logistic regression votes for the features it doesn't shrink to zero.
    It learns from the labels, so it must be a step of the Pipeline (re-fitted in every fold).
    """

    def __init__(self, min_votes: int = 2, random_state=None):
        self.min_votes = min_votes
        self.random_state = random_state

    def fit(self, X, y):
        forest = RandomForestClassifier(n_estimators=200, min_samples_leaf=20, random_state=self.random_state, n_jobs=-1).fit(X, y)
        booster = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        lasso = LogisticRegression(l1_ratio=1.0, C=0.01, solver="liblinear").fit(X, y)
        self.votes_ = ((forest.feature_importances_ > forest.feature_importances_.mean()).astype(int)
                       + (booster.feature_importances_ > booster.feature_importances_.mean())
                       + (np.abs(lasso.coef_[0]) > 1e-8))
        self.support_ = self.votes_ >= self.min_votes
        return self

    def _get_support_mask(self):   # SelectorMixin builds transform() and get_feature_names_out() from this
        return self.support_


def build_pipeline(preprocessing_config: dict, model_config: dict, selection_config: dict = None,
                   memory=None) -> Pipeline:
    """
    Builds one Pipeline, not fitted yet, with these steps, in this order:
        the preprocessing ("prep": imputation, encoding and scaling, from `preprocessing_config`)
        the feature selection ("select", only if `selection_config["enabled"]`)
        the model ("model", from `model_config`).
    `memory` (a folder) stores the fitted "prep" and "select" steps: a fit on the same rows with the
    same settings is reused, so tuning the model doesn't refit them in every trial.
    """
    steps = [("prep", build_column_transformer(preprocessing_config))]   # step 1: prepares the data
    if selection_config and selection_config.get("enabled", False):      # step 2 (optional): keeps some features
        steps.append(("select", VoteSelector(min_votes=selection_config.get("min_votes", 2),
                                             random_state=selection_config.get("random_state"))))
    steps.append(("model", build_model(model_config)))                   # last step: learns from them
    return Pipeline(steps, memory=memory)
