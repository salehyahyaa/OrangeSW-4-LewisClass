"""Step 6: try different settings for each model and report which gives the
highest cross-validated classification accuracy (CA).

Uses the training sample exactly as it comes out of the saved workflow, and
the same 10-fold cross-validation the Test and Score widget runs
(random_state 42). Every setting tried here is one that can be set in the
corresponding Orange widget. Results go to ML2/results/tuning.json; the
winners are copied into TUNED in build_workflow.py.

    /Applications/Orange.app/Contents/MacOS/python ML2/tune_models.py
"""
import itertools
import json
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from Orange.evaluation import AUC, CA, CrossValidation  # noqa: E402
from Orange.widgets.model.owknn import OWKNNLearner  # noqa: E402
from Orange.widgets.model.owlogisticregression import (  # noqa: E402
    OWLogisticRegression,
)
from Orange.widgets.model.owneuralnetwork import OWNNLearner  # noqa: E402
from Orange.widgets.model.owrandomforest import OWRandomForest  # noqa: E402
from Orange.widgets.model.owsvm import OWSVM  # noqa: E402
from Orange.widgets.model.owtree import OWTreeLearner  # noqa: E402

from build_workflow import SPLIT, output  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

C_INDEX = {v: i for i, v in enumerate(OWLogisticRegression.C_s)}

# widget class, and for each setting the values to try
GRID = {
    "Logistic Regression": (OWLogisticRegression, {
        "penalty_type": [0, 1],                   # Lasso (L1), Ridge (L2)
        "C_index": [C_INDEX[c] for c in
                    (0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100, 1000)],
    }),
    "kNN": (OWKNNLearner, {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
        "metric_index": [0, 1, 2],          # euclidean, manhattan, chebyshev
        "weight_index": [0, 1],             # uniform, by distances
    }),
    "SVM": (OWSVM, {
        "kernel_type": [0, 1, 2, 3],        # linear, polynomial, RBF, sigmoid
        "C": [0.1, 0.5, 1.0, 5.0, 10.0],
        "max_iter": [100, 1000],
    }),
    "Tree": (OWTreeLearner, {
        "min_leaf": [1, 2, 5, 10],
        "min_internal": [2, 5, 10, 20],
        "max_depth": [3, 4, 5, 7, 100],
        "sufficient_majority": [95, 100],
    }),
    "Random Forest": (OWRandomForest, {
        "n_estimators": [10, 50, 100, 200],
        "use_max_features": [False, True], "max_features": [2, 3, 5],
        "use_min_samples_split": [True, False], "min_samples_split": [5, 10],
        "use_random_state": [True],
    }),
    "Neural Network": (OWNNLearner, {
        "hidden_layers_input": ["10,", "50,", "100,", "100,50", "20,10"],
        "activation_index": [1, 2, 3],      # logistic, tanh, relu
        "solver_index": [0, 2],             # L-BFGS-B, Adam
        "alpha_index": [1, 5, 10, 14],
        "max_iterations": [200, 500],
        "replicable": [True],
    }),
}


def skip(name, s):
    """Drop combinations that are the same model written two ways."""
    if name == "Random Forest":
        if not s["use_max_features"] and s["max_features"] != 2:
            return True
        if not s["use_min_samples_split"] and s["min_samples_split"] != 5:
            return True
    return False


def learner_for(cls, settings):
    """Let the real widget build the learner from its own settings."""
    w = cls()
    for key, value in settings.items():
        setattr(w, key, value)
    if cls is OWLogisticRegression:
        w.strength_C = w.C_s[w.C_index]
    learner = w.create_learner()
    w.deleteLater()
    return learner


def main():
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=3.0, timeout=600)
    split = next(n for n in scheme.nodes if n.title == SPLIT)
    train = output(scheme, split, "Data Sample")
    print("training rows:", len(train), "features:",
          len(train.domain.attributes))

    cv = CrossValidation(k=10, random_state=42)
    summary = {}
    for name, (cls, grid) in GRID.items():
        keys = list(grid)
        tried = []
        for values in itertools.product(*(grid[k] for k in keys)):
            settings = dict(zip(keys, values))
            if skip(name, settings):
                continue
            try:
                res = cv(train, [learner_for(cls, settings)])
                ca, auc = float(CA(res)[0]), float(AUC(res)[0])
            except Exception as err:   # an invalid combination
                print(name, settings, "failed:", err)
                continue
            tried.append({"settings": settings, "ca": ca, "auc": auc})
        # highest CA first; AUC breaks ties
        tried.sort(key=lambda d: (-d["ca"], -d["auc"]))
        default = cls()
        if hasattr(default, "replicable"):
            default.replicable = True
        if hasattr(default, "use_random_state"):
            default.use_random_state = True
        default_ca = float(CA(cv(train, [default.create_learner()]))[0])
        summary[name] = {"tried": len(tried), "default_ca": default_ca,
                         "best": tried[0], "worst": tried[-1],
                         "top5": tried[:5]}
        print("%-20s tried %3d  default %.4f  best %.4f  worst %.4f  %s"
              % (name, len(tried), default_ca, tried[0]["ca"],
                 tried[-1]["ca"], tried[0]["settings"]))
    os.makedirs(oh.RESULTS, exist_ok=True)
    with open(os.path.join(oh.RESULTS, "tuning.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
