"""Step 6: try different embedders, models and model settings and record the
10-fold cross-validated accuracy of each on the training pictures.

The same things the widgets do: Import Images reads the train folder, Image
Embedding turns each picture into numbers, and Test and Score runs 10-fold
cross-validation (random_state 42). Every setting tried here can be set in
the matching Orange widget. Only the training pictures are used, so the
test pictures stay unseen. Results go to Lab5/results/experiments.json.

    /Applications/Orange.app/Contents/MacOS/python Lab5/experiments.py
"""
import json
import logging
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from Orange.evaluation import AUC, CA, F1, CrossValidation  # noqa: E402
from Orange.widgets.model.owknn import OWKNNLearner  # noqa: E402
from Orange.widgets.model.owlogisticregression import (  # noqa: E402
    OWLogisticRegression,
)
from Orange.widgets.model.owneuralnetwork import OWNNLearner  # noqa: E402
from Orange.widgets.model.owrandomforest import OWRandomForest  # noqa: E402
from Orange.widgets.model.owsvm import OWSVM  # noqa: E402
from orangecontrib.imageanalytics.image_embedder import (  # noqa: E402
    MODELS, ImageEmbedder,
)
from orangecontrib.imageanalytics.import_images import (  # noqa: E402
    ImportImages,
)

warnings.filterwarnings("ignore")
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

TRAIN = os.path.join(oh.SUBMISSION, "classificationDataset", "train")
EMBEDDERS = ["squeezenet", "inception-v3", "vgg16", "vgg19", "painters",
             "deeploc"]

L1, L2 = 0, 1
C = {v: i for i, v in enumerate(OWLogisticRegression.C_s)}
IDENTITY, LOGISTIC, TANH, RELU = 0, 1, 2, 3
LBFGS, SGD, ADAM = 0, 1, 2
LINEAR_KERNEL, RBF_KERNEL = 0, 2
EUCLIDEAN, MANHATTAN = 0, 1
UNIFORM, BY_DISTANCE = 0, 1

# (model, what was set, widget class, widget settings); the first entry of
# each model is the widget's default.
CONFIGS = [
    ("Logistic Regression", "Ridge (L2), C=1 (default)",
     OWLogisticRegression, {"penalty_type": L2, "C_index": C[1]}),
    ("Logistic Regression", "Lasso (L1), C=1",
     OWLogisticRegression, {"penalty_type": L1, "C_index": C[1]}),
    ("Logistic Regression", "Ridge (L2), C=0.01",
     OWLogisticRegression, {"penalty_type": L2, "C_index": C[0.01]}),
    ("Logistic Regression", "Ridge (L2), C=100",
     OWLogisticRegression, {"penalty_type": L2, "C_index": C[100]}),

    ("Neural Network", "100 neurons, ReLu, Adam (default)", OWNNLearner,
     {"hidden_layers_input": "100,", "activation_index": RELU,
      "solver_index": ADAM, "replicable": True}),
    ("Neural Network", "10 neurons, ReLu, Adam", OWNNLearner,
     {"hidden_layers_input": "10,", "activation_index": RELU,
      "solver_index": ADAM, "replicable": True}),
    ("Neural Network", "100,50 neurons, ReLu, Adam", OWNNLearner,
     {"hidden_layers_input": "100,50", "activation_index": RELU,
      "solver_index": ADAM, "replicable": True}),
    ("Neural Network", "100 neurons, tanh, Adam", OWNNLearner,
     {"hidden_layers_input": "100,", "activation_index": TANH,
      "solver_index": ADAM, "replicable": True}),
    ("Neural Network", "100 neurons, ReLu, L-BFGS-B", OWNNLearner,
     {"hidden_layers_input": "100,", "activation_index": RELU,
      "solver_index": LBFGS, "replicable": True}),

    ("Random Forest", "10 trees (default)", OWRandomForest,
     {"n_estimators": 10, "use_random_state": True}),
    ("Random Forest", "50 trees", OWRandomForest,
     {"n_estimators": 50, "use_random_state": True}),
    ("Random Forest", "100 trees", OWRandomForest,
     {"n_estimators": 100, "use_random_state": True}),
    ("Random Forest", "200 trees", OWRandomForest,
     {"n_estimators": 200, "use_random_state": True}),

    ("kNN", "5 neighbors, Euclidean (default)", OWKNNLearner,
     {"n_neighbors": 5, "metric_index": EUCLIDEAN, "weight_index": UNIFORM}),
    ("kNN", "1 neighbor, Euclidean", OWKNNLearner,
     {"n_neighbors": 1, "metric_index": EUCLIDEAN, "weight_index": UNIFORM}),
    ("kNN", "3 neighbors, Euclidean", OWKNNLearner,
     {"n_neighbors": 3, "metric_index": EUCLIDEAN, "weight_index": UNIFORM}),
    ("kNN", "5 neighbors, Manhattan, by distances", OWKNNLearner,
     {"n_neighbors": 5, "metric_index": MANHATTAN,
      "weight_index": BY_DISTANCE}),

    ("SVM", "RBF kernel, C=1 (default)", OWSVM,
     {"kernel_type": RBF_KERNEL, "C": 1.0}),
    ("SVM", "Linear kernel, C=1", OWSVM,
     {"kernel_type": LINEAR_KERNEL, "C": 1.0}),
    ("SVM", "RBF kernel, C=10", OWSVM,
     {"kernel_type": RBF_KERNEL, "C": 10.0}),
]


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


def embed(images, model):
    """Same as the Image Embedding widget: returns the embedded table."""
    with ImageEmbedder(model=model) as embedder:
        embedded, skipped, n_skipped = embedder(images, col="image")
    return embedded, n_skipped


def main():
    oh.app()
    images = ImportImages()(TRAIN)[0]
    print("training pictures:", len(images),
          [str(v) for v in images.domain.class_var.values])
    cv = CrossValidation(k=10, random_state=42)
    rows, features = [], {}
    for model in EMBEDDERS:
        name = MODELS[model]["name"]
        try:
            data, n_skipped = embed(images, model)
        except Exception as err:
            print(name, "embedding failed:", repr(err))
            continue
        if data is None or n_skipped:
            print(name, "skipped pictures:", n_skipped)
            continue
        features[name] = len(data.domain.attributes)
        print("%-13s %d features" % (name, features[name]))
        for learner_name, setting, cls, settings in CONFIGS:
            res = cv(data, [learner_for(cls, settings)])
            row = {"embedder": name, "model": learner_name,
                   "setting": setting, "ca": float(CA(res)[0]),
                   "auc": float(AUC(res)[0]),
                   "f1": float(F1(res, average="weighted")[0])}
            rows.append(row)
            print("   %-20s %-38s CA %.3f  AUC %.3f"
                  % (learner_name, setting, row["ca"], row["auc"]))
    os.makedirs(oh.RESULTS, exist_ok=True)
    with open(os.path.join(oh.RESULTS, "experiments.json"), "w") as f:
        json.dump({"features": features, "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
