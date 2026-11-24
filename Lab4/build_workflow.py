"""Build Lab4/submission/lab4_yahya.ows - the Orange workflow for Lab 4.

The workflow is put together from the real Orange widgets, run against the
HDI dataset (loaded by the Datasets widget) so each widget can be set up on
the actual data, and then saved in Orange's own .ows format. The layout
follows the sample workflow in the lab handout.

    /Applications/Orange.app/Contents/MacOS/python Lab4/build_workflow.py
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
DATASETS = W + "data.owdatasets.OWDataSets"
TABLE = W + "data.owtable.OWTable"
BOX = W + "visualize.owboxplot.OWBoxPlot"
DIST = W + "visualize.owdistributions.OWDistributions"
SELECT = W + "data.owselectcolumns.OWSelectAttributes"
PREPROCESS = W + "data.owpreprocess.OWPreprocess"
SAMPLER = W + "data.owdatasampler.OWDataSampler"
LINEAR = W + "model.owlinearregression.OWLinearRegression"
RANK = W + "data.owrank.OWRank"
TEST = W + "evaluate.owtestandscore.OWTestAndScore"
ADABOOST = W + "model.owadaboost.OWAdaBoost"
FOREST = W + "model.owrandomforest.OWRandomForest"
NEURAL = W + "model.owneuralnetwork.OWNNLearner"
KNN = W + "model.owknn.OWKNNLearner"

TARGET = "HDI"

# Linear Regression widget settings (step 3)
OLS, RIDGE, LASSO, ELASTIC = 0, 1, 2, 3
ALPHA_LOWEST = 0            # slider all the way left = 0.0001
ALPHA_001 = 18              # slider position for regularization 0.01
L2_RATIO = 0.76             # shown in the widget as "0.24 : 0.76"

SORT_DESCENDING = -1        # Data Table sort order (1 = ascending)

# Test and Score settings
CROSS_VALIDATION, TEST_ON_TEST = 0, 5
TEN_FOLDS = 3               # index into [2, 3, 5, 10, 20]

# Step 7: the settings that gave the lowest cross-validated RMSE
# (found with Lab4/tune_models.py).
TUNED = {
    "AdaBoost": {"n_estimators": 25, "learning_rate": 1.0, "loss_index": 1,
                 "use_random_seed": True, "random_seed": 0},
    "Random Forest": {"n_estimators": 50, "use_random_state": True,
                      "use_max_features": True, "max_features": 20,
                      "use_max_depth": False,
                      "use_min_samples_split": True, "min_samples_split": 5},
    "Neural Network": {"hidden_layers_input": "10,", "activation_index": 1,
                       "solver_index": 0, "alpha_index": 5,
                       "max_iterations": 200, "replicable": True},
    "kNN": {"n_neighbors": 6, "metric_index": 1, "weight_index": 1},
}


def linear(name, reg_type, alpha_index=0, l2_ratio=0.5):
    return {"learner_name": name, "reg_type": reg_type,
            "alpha_index": alpha_index, "l2_ratio": l2_ratio,
            "fit_intercept": True}


def named(name, settings):
    return dict(settings, learner_name=name)


# title -> (widget class, (x, y), saved settings)
NODES = {
    "Datasets": (DATASETS, (80, 300), {"selected_id": oh.DATASET_ID}),
    "Data Table": (TABLE, (280, 110), {}),
    "Distributions": (DIST, (280, 210), {"fitted_distribution": 0}),
    "Box Plot": (BOX, (280, 310), {}),
    "Select Columns": (SELECT, (200, 480), {}),
    "Preprocess": (PREPROCESS, (350, 480), {}),
    "Data Sampler": (SAMPLER, (480, 330),
                     {"sampling_type": 0, "sampleSizePercentage": 50,
                      "use_seed": True, "replacement": False,
                      "stratify": False}),
    "Training set": (TABLE, (640, 60), {}),
    "No Regularization": (LINEAR, (920, 250),
                          linear("No Regularization", OLS)),
    "Lasso Regression (L1)": (LINEAR, (920, 450), linear(
        "Lasso Regression (L1)", LASSO, ALPHA_LOWEST)),
    "Elastic Net Regression": (LINEAR, (920, 650), linear(
        "Elastic Net Regression", ELASTIC, ALPHA_001, L2_RATIO)),
    "Coefficients (No Reg.)": (TABLE, (1030, 340), {}),
    "Coefficients (Lasso)": (TABLE, (1030, 540), {}),
    "Coefficients (Elastic Net)": (TABLE, (1030, 740), {}),
    "Test and Score": (TEST, (1700, 450), {"resampling": CROSS_VALIDATION,
                                          "n_folds": TEN_FOLDS}),
    "Rank": (RANK, (640, 920), {"selected_methods": {"RReliefF"}}),
    "Elastic Net Regression (1)": (LINEAR, (920, 920), linear(
        "Elastic Net Regression (1)", ELASTIC, ALPHA_001, L2_RATIO)),
    "Coefficients (Top feature)": (TABLE, (1030, 1010), {}),
    "Reduced Data": (TABLE, (800, 1060), {}),
    "Testing set": (TABLE, (640, -80), {}),
    "Training set (1)": (TABLE, (640, 1385), {}),
    "Testing set (1)": (TABLE, (640, 1950), {}),
    "AdaBoost": (ADABOOST, (920, 1190), named("AdaBoost", TUNED["AdaBoost"])),
    "Random Forest": (FOREST, (920, 1310),
                      named("Random Forest", TUNED["Random Forest"])),
    "Neural Network": (NEURAL, (920, 1450),
                       named("Neural Network", TUNED["Neural Network"])),
    "kNN": (KNN, (920, 1570), named("kNN", TUNED["kNN"])),
    "Test and Score (1)": (TEST, (1250, 1385),
                           {"resampling": CROSS_VALIDATION,
                            "n_folds": TEN_FOLDS}),
}
LINEAR_MODELS = ["No Regularization", "Lasso Regression (L1)",
                 "Elastic Net Regression"]
OTHER_MODELS = ["AdaBoost", "Random Forest", "Neural Network", "kNN"]
SELECT_ALL = ["Training set", "Testing set", "Training set (1)",
              "Testing set (1)"]

# (source, output channel, sink, input channel), in the order they are wired
LINKS_PREPARE = [
    ("Datasets", "Data", "Data Table", "Data"),
    ("Datasets", "Data", "Distributions", "Data"),
    ("Datasets", "Data", "Box Plot", "Data"),
    ("Datasets", "Data", "Select Columns", "Data"),
    ("Select Columns", "Data", "Preprocess", "Data"),
    ("Preprocess", "Preprocessed Data", "Data Sampler", "Data"),
    ("Data Sampler", "Data Sample", "Training set", "Data"),
    ("Data Sampler", "Remaining Data", "Testing set", "Data"),
    ("Data Sampler", "Data Sample", "Training set (1)", "Data"),
    ("Data Sampler", "Remaining Data", "Testing set (1)", "Data"),
]
LINKS_LINEAR = [
    ("Training set", "Selected Data", "Test and Score", "Data"),
    ("Training set", "Selected Data", "No Regularization", "Data"),
    ("Training set", "Selected Data", "Lasso Regression (L1)", "Data"),
    ("Training set", "Selected Data", "Elastic Net Regression", "Data"),
    ("No Regularization", "Learner", "Test and Score", "Learner"),
    ("Lasso Regression (L1)", "Learner", "Test and Score", "Learner"),
    ("Elastic Net Regression", "Learner", "Test and Score", "Learner"),
    ("No Regularization", "Coefficients",
     "Coefficients (No Reg.)", "Data"),
    ("Lasso Regression (L1)", "Coefficients", "Coefficients (Lasso)", "Data"),
    ("Elastic Net Regression", "Coefficients",
     "Coefficients (Elastic Net)", "Data"),
]
LINKS_RANK = [
    ("Training set", "Selected Data", "Rank", "Data"),
    ("Rank", "Reduced Data", "Elastic Net Regression (1)", "Data"),
    ("Rank", "Reduced Data", "Reduced Data", "Data"),
    ("Elastic Net Regression (1)", "Learner", "Test and Score", "Learner"),
    ("Elastic Net Regression (1)", "Coefficients",
     "Coefficients (Top feature)", "Data"),
]
LINKS_TEST = [("Testing set", "Selected Data", "Test and Score", "Test Data")]
LINKS_OTHER = [
    ("Training set (1)", "Selected Data", "Test and Score (1)", "Data"),
    ("Testing set (1)", "Selected Data", "Test and Score (1)", "Test Data"),
] + [("Training set (1)", "Selected Data", m, "Data") for m in OTHER_MODELS] \
  + [(m, "Learner", "Test and Score (1)", "Learner") for m in OTHER_MODELS]

# (x, y, width, height, text)
TEXTS = [
    (20, -150, 360, 50,
     "**Lab 4: Regression Analysis**  \nHDI dataset, 188 countries"),
    (360, 150, 190, 90,
     "**Step 1: Explore**  \nLoad HDI with the Datasets widget and look "
     "at it."),
    (60, 570, 330, 100,
     "**Steps 1-2: Prepare**  \nHDI set as the target, features normalized "
     "to [0, 1], then a 50% / 50% split into training and testing sets."),
    (1540, 270, 320, 70,
     "**Step 3: Linear regression**  \nThree models (no regularization, "
     "Lasso, Elastic Net) scored with 10-fold cross-validation."),
    (1200, 800, 230, 80,
     "**Step 4: Coefficients**  \nOne table of coefficients for each "
     "model."),
    (1200, 1000, 260, 100,
     "**Step 5: Rank**  \nTop 1 feature by RReliefF, then Elastic Net "
     "again on just that feature."),
    (290, -170, 250, 90,
     "**Step 6: Test data**  \nTesting set connected so the models can be "
     "scored on test data."),
    (1340, 1450, 290, 100,
     "**Step 7: Other models**  \nAdaBoost, Random Forest, Neural Network "
     "and kNN, tuned for the lowest cross-validated RMSE."),
]
BLUE, GREEN, RED = "#1F77B4", "#39B54A", "#C1272D"
ARROWS = [
    ((400, 240), (330, 250), RED),         # step 1 -> explore widgets
    ((250, 570), (275, 525), BLUE),        # prepare -> Select/Preprocess
    ((1700, 350), (1700, 400), GREEN),    # step 3 -> Test and Score
    ((1200, 825), (1110, 770), GREEN),     # step 4 -> coefficient tables
    ((1200, 1030), (1110, 1015), BLUE),      # step 5 -> top feature coefs
    ((540, -110), (595, -90), BLUE),       # step 6 -> Testing set
    ((1340, 1470), (1295, 1425), RED),     # step 7 -> Test and Score (1)
]


def output(scheme, node, channel):
    """Current value on a node's output channel."""
    sm = scheme.signal_manager
    for link in scheme.links:
        if link.source_node is node and link.source_channel.name == channel:
            contents = sm.link_contents(link)
            if contents:
                return next(iter(contents.values()))
    raise LookupError("%s / %s" % (node.title, channel))


def set_box(box, attribute, group=None):
    domain = box.dataset.domain
    box.attribute = domain[attribute]
    box.group_var = domain[group] if group else None
    box._set_list_view_selections()
    box._variables_changed(box.apply_attr_sorting)


def set_distribution(dist, var, bins=None):
    dist.var = dist.data.domain[var]
    dist.cvar = None
    dist._on_var_changed()
    dist._on_cvar_changed()
    if bins is not None:
        dist.number_of_bins = bins
        dist._on_bins_changed()


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
    from Orange.widgets.data.owpreprocess import Scale
    from Orange.widgets.model.owlinearregression import OWLinearRegression
    assert OWLinearRegression.alphas[ALPHA_001] == 0.01, \
        OWLinearRegression.alphas
    assert OWLinearRegression.alphas[ALPHA_LOWEST] == min(
        OWLinearRegression.alphas)

    reg = oh.registry()
    scheme = oh.new_scheme(
        title="Lab 4 - Regression Analysis (HDI)",
        description="Predicts the Human Development Index with linear "
                    "regression (plain, Lasso, Elastic Net), looks at the "
                    "coefficients, and compares AdaBoost, Random Forest, "
                    "Neural Network and kNN.")
    props = {t: dict(p) for t, (_, _, p) in NODES.items()}
    props["Preprocess"]["storedsettings"] = {
        "name": "", "preprocessors": [
            ("orange.preprocess.scale",
             {"method": Scale.NormalizeBySpan_ZeroBased})]}

    nodes = {}

    def add(title):
        qualified_name, pos, _ = NODES[title]
        p = props[title]
        if p:
            module, cls_name = qualified_name.rsplit(".", 1)
            cls = getattr(importlib.import_module(module), cls_name)
            p.setdefault("__version__", cls.settings_version)
        node = SchemeNode(reg.widget(qualified_name), title=title,
                          position=pos, properties=p)
        scheme.add_node(node)
        nodes[title] = node
        return scheme.widget_for_node(node)

    def connect(links):
        for src, out_name, dst, in_name in links:
            a, b = nodes[src], nodes[dst]
            scheme.new_link(a, a.output_channel(out_name),
                            b, b.input_channel(in_name))

    widgets = {t: add(t) for t in NODES}
    oh.settle(scheme, quiet=3.0, timeout=240)   # Datasets downloads HDI
    assert widgets["Datasets"].selected_dataset().title == TARGET

    # ---- Steps 1-2: load, explore, set the target, normalize, split ----
    connect(LINKS_PREPARE[:4])
    oh.settle(scheme, quiet=1.5)
    set_distribution(widgets["Distributions"], TARGET, bins=1)
    set_box(widgets["Box Plot"], TARGET)

    select = widgets["Select Columns"]
    hdi = next(v for v in select.meta_attrs if v.name == TARGET)
    select.meta_attrs[:] = [v for v in select.meta_attrs if v is not hdi]
    select.class_attrs[:] = [hdi]
    select.commit.now()
    connect(LINKS_PREPARE[4:])
    oh.settle(scheme, quiet=1.5)
    for title in SELECT_ALL:
        select_all_rows(widgets[title])
    oh.settle(scheme, quiet=1.0)

    # ---- Steps 3-4: three linear models, scores and coefficients ----
    connect(LINKS_LINEAR)
    oh.settle(scheme, quiet=2.0, timeout=240)

    # ---- Step 5: top feature by RReliefF, Elastic Net on it ----
    connect(LINKS_RANK[:1])
    oh.settle(scheme, quiet=2.0, timeout=240)
    rank = widgets["Rank"]
    rank.nSelected = 1
    rank.selectionMethod = rank.SelectNBest
    rank.autoSelection()
    rank.commit.now()
    connect(LINKS_RANK[1:])
    oh.settle(scheme, quiet=2.0, timeout=240)

    # Step 4: sort every coefficient table by value, largest first
    for title in NODES:
        if title.startswith("Coefficients"):
            widgets[title].set_sort_columns([("coef", SORT_DESCENDING)])

    # ---- Step 6: test data, and "Test on test data" ----
    connect(LINKS_TEST)
    oh.settle(scheme, quiet=1.5, timeout=240)
    set_resampling(widgets["Test and Score"], TEST_ON_TEST)
    oh.settle(scheme, quiet=1.5, timeout=240)

    # ---- Step 7: the other models ----
    connect(LINKS_OTHER)
    oh.settle(scheme, quiet=2.0, timeout=600)

    for x, y, w, h, text in TEXTS:
        scheme.add_annotation(SchemeTextAnnotation(
            (x, y, w, h), text, "text/markdown",
            font={"family": "Helvetica", "size": 14}))
    for start, end, color in ARROWS:
        scheme.add_annotation(SchemeArrowAnnotation(start, end, color))

    problems = oh.widget_problems(scheme)
    scheme.sync_node_properties()
    with open(oh.WORKFLOW, "wb") as f:
        scheme.save_to(f, pretty=True, pickle_fallback=True)
    print("saved", oh.WORKFLOW)
    print("widget problems:", problems or "none")


if __name__ == "__main__":
    build()
