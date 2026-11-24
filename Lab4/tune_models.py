"""Step 7: try different settings for AdaBoost, Random Forest, Neural
Network and kNN and report which gives the lowest cross-validated RMSE.

Uses the training set exactly as it comes out of the saved workflow, and
the same 10-fold cross-validation the Test and Score widget runs
(random_state 42). Every setting tried here is one that can be set in the
corresponding Orange widget. Results go to Lab4/results/tuning.json; the
winners are copied into TUNED in build_workflow.py.

    /Applications/Orange.app/Contents/MacOS/python Lab4/tune_models.py
"""
import itertools
import json
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

import Orange  # noqa: E402
from Orange.evaluation import RMSE, CrossValidation  # noqa: E402
from Orange.widgets.model.owadaboost import OWAdaBoost  # noqa: E402
from Orange.widgets.model.owknn import OWKNNLearner  # noqa: E402
from Orange.widgets.model.owneuralnetwork import OWNNLearner  # noqa: E402
from Orange.widgets.model.owrandomforest import OWRandomForest  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

# widget class, and for each setting the values to try
GRID = {
    "AdaBoost": (OWAdaBoost, {
        "n_estimators": [10, 25, 50, 100, 200],
        "learning_rate": [0.1, 0.5, 1.0, 2.0],
        "loss_index": [0, 1, 2],                  # linear, square, exponential
        "use_random_seed": [True], "random_seed": [0],
    }),
    "Random Forest": (OWRandomForest, {
        "n_estimators": [10, 50, 100, 200],
        "use_max_features": [False, True], "max_features": [5, 10, 20],
        "use_max_depth": [False],
        "use_min_samples_split": [True, False], "min_samples_split": [2, 5],
        "use_random_state": [True],
    }),
    "Neural Network": (OWNNLearner, {
        "hidden_layers_input": ["10,", "50,", "100,", "100,50", "50,50"],
        "activation_index": [0, 1, 2, 3],   # identity, logistic, tanh, relu
        "solver_index": [0, 2],             # L-BFGS-B, Adam
        "alpha_index": [1, 5, 10, 14],
        "max_iterations": [200, 500],
        "replicable": [True],
    }),
    "kNN": (OWKNNLearner, {
        "n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20],
        "metric_index": [0, 1, 2],          # euclidean, manhattan, chebyshev
        "weight_index": [0, 1],             # uniform, by distances
    }),
}


def skip(name, s):
    """Drop combinations that are the same model written two ways."""
    if name == "Random Forest":
        if not s["use_max_features"] and s["max_features"] != 5:
            return True
        if not s["use_min_samples_split"] and s["min_samples_split"] != 2:
            return True
        if s["use_min_samples_split"] and s["min_samples_split"] == 2:
            return True
    return False


def learner_for(cls, settings):
    """Let the real widget build the learner from its own settings."""
    w = cls()
    for key, value in settings.items():
        setattr(w, key, value)
    learner = w.create_learner()
    w.deleteLater()
    return learner


def main():
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=3.0, timeout=600)
    train = None
    for link in scheme.links:
        if link.source_node.title == "Training set (1)" and \
                link.sink_node.title == "Test and Score (1)":
            train = next(iter(
                scheme.signal_manager.link_contents(link).values()))
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
                rmse = float(RMSE(res)[0])
            except Exception as err:   # an invalid combination
                print(name, settings, "failed:", err)
                continue
            tried.append({"settings": settings, "rmse": rmse})
        tried.sort(key=lambda d: d["rmse"])
        default = cls()
        default_rmse = float(RMSE(cv(train, [default.create_learner()]))[0])
        summary[name] = {"tried": len(tried), "default_rmse": default_rmse,
                         "best": tried[0], "worst": tried[-1],
                         "top5": tried[:5]}
        print("%-15s tried %3d  default %.4f  best %.4f  worst %.4f  %s"
              % (name, len(tried), default_rmse, tried[0]["rmse"],
                 tried[-1]["rmse"], tried[0]["settings"]))
    os.makedirs(oh.RESULTS, exist_ok=True)
    with open(os.path.join(oh.RESULTS, "tuning.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
