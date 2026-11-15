"""
The nine classifiers of Khan et al. (2023), Table 3, with the Table 7 hyperparameter grids,
plus one helper that tunes and fits a model. Experiments import from here so that a model,
its grid, or the cross-validation scheme can be swapped in one place.

    from src.khan_models import MODELS, fit_predict
    pred = fit_predict("Random Forest", Xtr, ytr, Xte, cv="kfold")
"""
import warnings

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
try:
    from xgboost import XGBClassifier
except ImportError:  # xgboost is optional
    XGBClassifier = None

# Table 7 grids (paired values in the paper are the GridSearchCV candidates)
MODELS = {
    "Logistic Regression": (LogisticRegression(solver="lbfgs"), {"max_iter": [100]}),
    "Random Forest": (RandomForestClassifier(criterion="gini", min_samples_leaf=4),
                      {"n_estimators": [80, 100], "random_state": [30, 20]}),
    "AdaBoost": (AdaBoostClassifier(learning_rate=1.0), {"n_estimators": [40, 60]}),
    "ANN": (MLPClassifier(hidden_layer_sizes=(100,), activation="relu", solver="adam",
                          max_iter=20, random_state=30), {}),
    "Decision Tree": (DecisionTreeClassifier(criterion="gini"),
                      {"random_state": [30, 20], "min_samples_split": [3, 4]}),
    "SVM": (SVC(), {"C": [1], "kernel": ["rbf"], "degree": [3]}),
    "KNN": (KNeighborsClassifier(metric="minkowski"),
            {"n_neighbors": [20, 30], "leaf_size": [30, 20]}),
    "Naive Bayes": (GaussianNB(), {}),
}
if XGBClassifier is not None:
    MODELS["XGBoost"] = (XGBClassifier(max_depth=7, eval_metric="logloss", verbosity=0),
                         {"n_estimators": [40, 60], "random_state": [30, 20]})

CV = {
    # scikit-learn's GridSearchCV default for classifiers (5 folds, no shuffling): what the paper used
    "kfold": lambda: StratifiedKFold(5),
    # folds that respect time order: what our own experiments use
    "timeseries": lambda: TimeSeriesSplit(5),
}


def fit_predict(name, Xtr, ytr, Xte, cv="kfold"):
    """Tune `name` with GridSearchCV on (Xtr, ytr) and return predictions for Xte."""
    est, grid = MODELS[name]
    gs = GridSearchCV(est, grid or {}, cv=CV[cv](), scoring="accuracy", n_jobs=-1)
    gs.fit(Xtr, ytr)
    return gs.predict(Xte)
