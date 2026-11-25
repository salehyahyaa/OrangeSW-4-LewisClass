"""Build Lab5/submission/lab5_yahya.ows - the Orange workflow for Lab 5.

The workflow is put together from the real Orange widgets, run on the
pictures in submission/classificationDataset so each widget can be set up
on the actual data, and then saved in Orange's own .ows format. The layout
follows the sample workflow in the lab handout.

    /Applications/Orange.app/Contents/MacOS/python Lab5/build_workflow.py
"""
import importlib
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from orangecanvas.scheme import (  # noqa: E402
    SchemeArrowAnnotation, SchemeNode, SchemeTextAnnotation,
)

logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

W = "Orange.widgets."
IA = "orangecontrib.imageanalytics.widgets."
IMPORT = IA + "owimageimport.OWImportImages"
EMBED = IA + "owimageembedding.OWImageEmbedding"
TABLE = W + "data.owtable.OWTable"
TEST = W + "evaluate.owtestandscore.OWTestAndScore"
CONFUSION = W + "evaluate.owconfusionmatrix.OWConfusionMatrix"
NEURAL = W + "model.owneuralnetwork.OWNNLearner"
FOREST = W + "model.owrandomforest.OWRandomForest"
LOGISTIC = W + "model.owlogisticregression.OWLogisticRegression"
KNN = W + "model.owknn.OWKNNLearner"
SVM = W + "model.owsvm.OWSVM"

DATASET = os.path.join(oh.SUBMISSION, "classificationDataset")

# Image Embedding: position of each embedder in the widget's drop-down
INCEPTION, SQUEEZENET = 0, 1
BEST_EMBEDDER = INCEPTION       # highest accuracy in Lab5/experiments.py

# Test and Score settings
CROSS_VALIDATION, TEST_ON_TEST = 0, 5
TEN_FOLDS = 3                   # index into [2, 3, 5, 10, 20]

RELU, ADAM = 3, 2
RIDGE, C_ONE = 1, 61
LINEAR_KERNEL, RBF_KERNEL = 0, 2

# Steps 4-5: the models as Orange sets them up by default (the random
# models get a fixed seed so the numbers do not change between runs).
DEFAULTS = {
    "Neural Network": {"hidden_layers_input": "100,",
                       "activation_index": RELU, "solver_index": ADAM,
                       "max_iterations": 200, "replicable": True},
    "Random Forest": {"n_estimators": 10, "use_random_state": True},
    "Logistic Regression": {"penalty_type": RIDGE, "C_index": C_ONE},
    "kNN": {"n_neighbors": 5, "metric_index": 0, "weight_index": 0},
    "SVM": {"kernel_type": RBF_KERNEL, "C": 1.0},
}
# Steps 6-7: the settings that did best in the experiments.
TUNED = dict(DEFAULTS)
TUNED["Random Forest"] = {"n_estimators": 50, "use_random_state": True}
TUNED["SVM"] = {"kernel_type": LINEAR_KERNEL, "C": 1.0}

MODEL_WIDGET = {"Neural Network": NEURAL, "Random Forest": FOREST,
                "Logistic Regression": LOGISTIC, "kNN": KNN, "SVM": SVM}
MODELS = list(MODEL_WIDGET)
MODEL_Y = [-240, -120, 120, 240, 360]      # offsets from the data line
A, B = 300, 1080                            # y of the two data lines


def named(name, settings):
    return dict(settings, learner_name=name)


# title -> (widget class, (x, y), saved settings)
NODES = {
    "Import Images": (IMPORT, (60, 690), {}),
    "Image Embedding": (EMBED, (240, A),
                        {"cb_embedder_current_id": SQUEEZENET}),
    "Training Data": (TABLE, (420, A), {}),
    "Test and Score": (TEST, (900, A), {"resampling": CROSS_VALIDATION,
                                        "n_folds": TEN_FOLDS,
                                        "cv_stratified": True}),
    "Confusion Matrix": (CONFUSION, (1100, A), {}),
    "Image Embedding (best)": (EMBED, (240, B),
                               {"cb_embedder_current_id": BEST_EMBEDDER}),
    "Training Data (1)": (TABLE, (420, B), {}),
    "Test and Score (1)": (TEST, (900, B), {"resampling": CROSS_VALIDATION,
                                            "n_folds": TEN_FOLDS,
                                            "cv_stratified": True}),
    "Confusion Matrix (1)": (CONFUSION, (1100, B), {}),
    "Import Images (test)": (IMPORT, (420, 1640), {}),
    "Image Embedding (test)": (EMBED, (640, 1640),
                               {"cb_embedder_current_id": BEST_EMBEDDER}),
}
for _name, _dy in zip(MODELS, MODEL_Y):
    NODES[_name] = (MODEL_WIDGET[_name], (660, A + _dy),
                    named(_name, DEFAULTS[_name]))
    NODES[_name + " (1)"] = (MODEL_WIDGET[_name], (660, B + _dy),
                             named(_name, TUNED[_name]))

IMPORT_FOLDER = {"Import Images": "train", "Import Images (test)": "test"}


def block(embedding, table, suffix):
    """Links of one train -> models -> Test and Score block."""
    test, matrix = "Test and Score" + suffix, "Confusion Matrix" + suffix
    return (
        [("Import Images", "Data", embedding, "Images"),
         (embedding, "Embeddings", table, "Data"),
         (table, "Selected Data", test, "Data")]
        + [(table, "Selected Data", m + suffix, "Data") for m in MODELS]
        + [(m + suffix, "Learner", test, "Learner") for m in MODELS]
        + [(test, "Evaluation Results", matrix, "Evaluation Results")])


LINKS_A = block("Image Embedding", "Training Data", "")
LINKS_B = block("Image Embedding (best)", "Training Data (1)", " (1)")
LINKS_TEST = [
    ("Import Images (test)", "Data", "Image Embedding (test)", "Images"),
    ("Image Embedding (test)", "Embeddings", "Test and Score (1)",
     "Test Data"),
]

# (x, y, width, height, text)
TEXTS = [
    (-130, -60, 420, 50,
     "**Lab 5: Classification of Images**  \nCats and dogs, 24 pictures "
     "of each"),
    (-130, 800, 230, 90,
     "**Step 2: Training images**  \nReads classificationDataset/train "
     "(12 cats, 12 dogs)."),
    (120, 70, 250, 110,
     "**Steps 2-3: SqueezeNet**  \nTurns each picture into 1000 numbers. "
     "The Data Table shows them."),
    (800, 20, 300, 110,
     "**Step 4: Models**  \nFive models with their default settings, "
     "scored with 10-fold cross-validation."),
    (1180, 180, 230, 90,
     "**Step 5: Confusion matrix**  \nRight and wrong guesses for each "
     "model."),
    (60, 1250, 300, 110,
     "**Step 6: Best embedder**  \nInception v3 gave the highest accuracy "
     "in my experiments, so this part uses it."),
    (1000, 1250, 330, 110,
     "**Step 7: Test on test data**  \nModels trained on the training "
     "pictures and scored on the test pictures."),
    (160, 1760, 430, 90,
     "**Step 7: Test images**  \nReads classificationDataset/test "
     "(12 cats, 12 dogs). Same embedder as the training data."),
]
BLUE, GREEN, RED = "#1F77B4", "#39B54A", "#C1272D"
ARROWS = [
    ((0, 800), (45, 750), RED),            # training images
    ((240, 180), (240, 250), BLUE),         # SqueezeNet embedding
    ((900, 130), (900, 250), GREEN),        # Test and Score
    ((1180, 240), (1125, 275), GREEN),      # Confusion Matrix
    ((230, 1250), (240, 1175), BLUE),       # best embedder
    ((1000, 1265), (935, 1150), RED),       # Test and Score (1)
    ((420, 1760), (420, 1695), RED),        # test images
]
# labels on the two data paths, like the sample workflow
LABELS = [
    (45, 440, 150, 30, "Training data"),
    (790, 1400, 110, 30, "Test data"),
]


def select_all_rows(table):
    """Select every row, so the table's Selected Data is the whole set."""
    data = table.input.table
    n_cols = len(data.domain.variables) + len(data.domain.metas)
    table.set_selection(list(range(len(data))), list(range(n_cols)))
    table.commit.now()


def set_resampling(test, mode):
    """Switch a Test and Score widget between its sampling options."""
    test.resampling = mode
    test._param_changed()


def build():
    os.makedirs(oh.SUBMISSION, exist_ok=True)
    reg = oh.registry()
    scheme = oh.new_scheme(
        title="Lab 5 - Classification of Images (cats and dogs)",
        description="Embeds pictures of cats and dogs with SqueezeNet and "
                    "Inception v3, compares five classification models "
                    "with cross-validation and tests them on separate "
                    "test pictures.")
    nodes = {}

    def add(title):
        qualified_name, pos, props = NODES[title]
        props = dict(props)
        if props:
            module, cls_name = qualified_name.rsplit(".", 1)
            cls = getattr(importlib.import_module(module), cls_name)
            props.setdefault("__version__", cls.settings_version)
        node = SchemeNode(reg.widget(qualified_name), title=title,
                          position=pos, properties=props)
        scheme.add_node(node)
        nodes[title] = node
        return scheme.widget_for_node(node)

    def connect(links):
        for src, out_name, dst, in_name in links:
            a, b = nodes[src], nodes[dst]
            scheme.new_link(a, a.output_channel(out_name),
                            b, b.input_channel(in_name))

    widgets = {t: add(t) for t in NODES}
    oh.settle(scheme, quiet=1.0)

    # ---- Step 2: point the two Import Images widgets at the folders ----
    for title, folder in IMPORT_FOLDER.items():
        importer = widgets[title]
        assert importer.setCurrentPath(os.path.join(DATASET, folder))
        importer.start()
    oh.settle(scheme, quiet=2.0)

    # ---- Steps 2-5: SqueezeNet, five models, scores, confusion matrix ----
    connect(LINKS_A[:2])
    connect(LINKS_B[:2])
    connect(LINKS_TEST[:1])
    oh.settle(scheme, quiet=3.0, timeout=900)
    for title in ("Training Data", "Training Data (1)"):
        select_all_rows(widgets[title])
    oh.settle(scheme, quiet=1.0)
    connect(LINKS_A[2:])
    oh.settle(scheme, quiet=3.0, timeout=900)

    # ---- Steps 6-7: best embedder, tuned models, test on test data ----
    connect(LINKS_B[2:])
    connect(LINKS_TEST[1:])
    oh.settle(scheme, quiet=3.0, timeout=900)
    set_resampling(widgets["Test and Score (1)"], TEST_ON_TEST)
    oh.settle(scheme, quiet=3.0, timeout=900)

    for x, y, w, h, text in TEXTS:
        scheme.add_annotation(SchemeTextAnnotation(
            (x, y, w, h), text, "text/markdown",
            font={"family": "Helvetica", "size": 14}))
    for x, y, w, h, text in LABELS:
        scheme.add_annotation(SchemeTextAnnotation(
            (x, y, w, h), "**%s**" % text, "text/markdown",
            font={"family": "Helvetica", "size": 14}))
    for start, end, color in ARROWS:
        scheme.add_annotation(SchemeArrowAnnotation(start, end, color))

    problems = oh.widget_problems(scheme)
    scheme.sync_node_properties()
    with open(oh.WORKFLOW, "wb") as f:
        scheme.save_to(f, pretty=True, pickle_fallback=True)
    print("saved", oh.WORKFLOW)
    print("widget problems:", problems or "none")
    for title in IMPORT_FOLDER:
        print(title, [(p.relpath, p.prefix)
                      for p in widgets[title].recent_paths])


if __name__ == "__main__":
    build()
