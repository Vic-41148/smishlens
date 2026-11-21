"""Model selection on the TRAIN split only.

Select on PR-AUC, never accuracy.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from smishlens.features import build_features

SCORING = {
    "ap": "average_precision",
    "f1": "f1",
    "mcc": make_scorer(matthews_corrcoef),  # a real object, not a string
}
CV = StratifiedKFold(5, shuffle=True, random_state=42)

GRIDS = {
    "mnb": (MultinomialNB(), {"clf__alpha": [0.01, 0.1, 0.5, 1.0]}),
    "cnb": (ComplementNB(), {"clf__alpha": [0.05, 0.1, 0.5, 1.0]}),
    "lr": (LogisticRegression(max_iter=2000, class_weight="balanced"), {"clf__C": [0.5, 1, 5, 10, 50]}),
    "svm": (LinearSVC(class_weight="balanced", dual="auto"), {"clf__C": [0.05, 0.1, 0.5, 1, 2, 5]}),
}


def tune(name: str, X, y, **feat_kw) -> GridSearchCV:  # noqa: N803
    """5-fold CV grid search with refit on PR-AUC."""
    clf, grid = GRIDS[name]
    pipe = Pipeline([("feats", build_features(**feat_kw)), ("clf", clf)])
    return GridSearchCV(pipe, grid, scoring=SCORING, refit="ap", cv=CV, n_jobs=-1).fit(X, y)
