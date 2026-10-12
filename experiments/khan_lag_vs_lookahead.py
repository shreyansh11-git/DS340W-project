"""
Correct-and-quantify, step 1: how much accuracy does the published notebook's
lookahead bias manufacture?

Runs all nine Khan et al. (2023) models (Table 3) with the Table 7 grids, on the
nine Table 2 features, under three versions of the "Previous" features:
    lagged (paper), same_day, lookahead (published notebook).

Data: TSLA daily bars, the repository's chronological train (2016-01-04 .. 2020-03-16)
and test (2020-03-17 .. 2021-05-26) splits. The held-out validation split is NOT loaded.
Min-Max scaler fit on train only. Hyperparameters chosen by GridSearchCV with a
5-fold TimeSeriesSplit inside the training period (no shuffling).

Usage:  python experiments/khan_lag_vs_lookahead.py
Writes: results/khan_lag_vs_lookahead.csv, results/khan_lag_vs_lookahead.md
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data import load_split  # noqa: E402
from src.khan_features import FEATURES, build  # noqa: E402

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional
    XGBClassifier = None

# Table 7 grids (paired values in the paper are the GridSearchCV candidates)
MODELS = {
    "SVM": (SVC(), {"C": [1], "kernel": ["rbf"], "degree": [3]}),
    "Decision Tree": (DecisionTreeClassifier(criterion="gini"),
                      {"random_state": [30, 20], "min_samples_split": [3, 4]}),
    "Logistic Regression": (LogisticRegression(solver="lbfgs"), {"max_iter": [100]}),
    "Naive Bayes": (GaussianNB(), {}),
    "KNN": (KNeighborsClassifier(metric="minkowski"),
            {"n_neighbors": [20, 30], "leaf_size": [30, 20]}),
    "Random Forest": (RandomForestClassifier(criterion="gini", min_samples_leaf=4),
                      {"n_estimators": [80, 100], "random_state": [30, 20]}),
    # SAMME.R was removed from recent scikit-learn; SAMME (now the only algorithm) is used
    "AdaBoost": (AdaBoostClassifier(learning_rate=1.0),
                 {"n_estimators": [40, 60]}),
    # paper used a Keras ANN (ReLU + sigmoid, 100 neurons, 20 epochs, Adam)
    "ANN": (MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                          max_iter=20, random_state=30), {}),
}
if XGBClassifier is not None:
    MODELS["XGBoost"] = (XGBClassifier(max_depth=7, eval_metric="logloss", verbosity=0),
                         {"n_estimators": [40, 60], "random_state": [30, 20]})

MODES = ["lagged", "same_day", "lookahead"]


def frames(mode):
    tr, te = load_split("train"), load_split("test")
    full = pd.concat([tr, te], ignore_index=True)  # indicators look backward only
    feat = build(full, prev_mode=mode)
    cut = te["date"].min()
    return feat[feat["date"] < cut], feat[feat["date"] >= cut]


def main():
    rows = []
    for mode in MODES:
        tr, te = frames(mode)
        Xtr, ytr, Xte, yte = tr[FEATURES], tr["target"].astype(int), te[FEATURES], te["target"].astype(int)
        for name, (est, grid) in MODELS.items():
            pipe = Pipeline([("scale", MinMaxScaler()), ("m", est)])
            gs = GridSearchCV(pipe, {f"m__{k}": v for k, v in grid.items()},
                              cv=TimeSeriesSplit(5), scoring="accuracy", n_jobs=-1)
            gs.fit(Xtr, ytr)
            pred = gs.predict(Xte)
            rows.append({"prev_features": mode, "model": name,
                         "test_accuracy": accuracy_score(yte, pred),
                         "test_f1": f1_score(yte, pred, average="weighted"),
                         "n_train": len(ytr), "n_test": len(yte)})
            print(f"{mode:10s} {name:20s} acc={rows[-1]['test_accuracy']:.4f}")
        base = max(yte.mean(), 1 - yte.mean())
        rows.append({"prev_features": mode, "model": "Majority-class baseline",
                     "test_accuracy": base, "test_f1": np.nan,
                     "n_train": len(ytr), "n_test": len(yte)})

    res = pd.DataFrame(rows)
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    res.to_csv(os.path.join(ROOT, "results", "khan_lag_vs_lookahead.csv"), index=False)
    wide = res.pivot(index="model", columns="prev_features", values="test_accuracy")[MODES]
    wide["inflation_pts"] = 100 * (wide["lookahead"] - wide["lagged"])
    with open(os.path.join(ROOT, "results", "khan_lag_vs_lookahead.md"), "w") as fh:
        fh.write("# Khan et al. features: lagged (paper) vs lookahead (published notebook)\n\n")
        fh.write("TSLA daily, test split 2020-03-17 to 2021-05-26. Test accuracy.\n\n")
        fh.write((wide * [100, 100, 100, 1]).round(2).to_markdown() + "\n")
    print(wide.round(4))


if __name__ == "__main__":
    main()
