"""Model construction: the model alone (`build_model`) and preprocessing + model (`build_pipeline`)."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.random_projection import SparseRandomProjection
from sklearn.tree import DecisionTreeClassifier

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
}


def build_model(model_config: dict):
    model_type = model_config["type"]
    params = model_config.get("params") or {}

    if model_type not in _MODELS:
        raise ValueError(f"Unknown model type: {model_type}. Options: {list(_MODELS)}")

    return _MODELS[model_type](**params)


def build_pipeline(preprocessing_config: dict, model_config: dict) -> Pipeline:
    """
    Builds one Pipeline with two steps, in this order: 
        the preprocessing ("prep": imputation, encoding and scaling, from `preprocessing_config`)  
        the model ("model", from `model_config`). It is not fitted yet.
    """
    return Pipeline([
        ("prep", build_column_transformer(preprocessing_config)),   # step 1: prepares the data
        ("model", build_model(model_config)),                       # step 2: learns from it
    ])
