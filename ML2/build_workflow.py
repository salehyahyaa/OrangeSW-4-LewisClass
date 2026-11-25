"""Build ML2/submission/mp2.ows - the Orange workflow for Machine Problem 2.

The workflow is put together from the real Orange widgets, run on the
Breast Cancer Wisconsin dataset (loaded by the Datasets widget) so each
widget can be set up on the actual data, and then saved in Orange's own
.ows format.

    /Applications/Orange.app/Contents/MacOS/python ML2/build_workflow.py
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
KMEANS = W + "unsupervised.owkmeans.OWKMeans"
DISTANCES = W + "unsupervised.owdistances.OWDistances"
HIERARCHICAL = W + "unsupervised.owhierarchicalclustering." \
    "OWHierarchicalClustering"
PIVOT = W + "data.owpivot.OWPivot"
TABLE = W + "data.owtable.OWTable"
BOX = W + "visualize.owboxplot.OWBoxPlot"
SCATTER = W + "visualize.owscatterplot.OWScatterPlot"
SAMPLER = W + "data.owdatasampler.OWDataSampler"
TEST = W + "evaluate.owtestandscore.OWTestAndScore"
LOGISTIC = W + "model.owlogisticregression.OWLogisticRegression"
KNN = W + "model.owknn.OWKNNLearner"
SVM = W + "model.owsvm.OWSVM"
TREE = W + "model.owtree.OWTreeLearner"
FOREST = W + "model.owrandomforest.OWRandomForest"
NEURAL = W + "model.owneuralnetwork.OWNNLearner"

DATASET_ID = "breast-cancer-wisconsin.tab"   # "Breast Cancer Wisconsin"
TARGET = "type"
CLUSTER = "Cluster"
CLUMP = "Clump thickness"
CELL_SIZE = "Unif_Cell_Size"
N_CLUSTERS = 2

TOP_N = 2                       # Hierarchical Clustering: selection "Top N"
ROWS = 0                        # Distances: between rows
EUCLIDEAN_NORMALIZED = 1        # Distances: "Euclidean (normalized)"
WARD = 4                        # Hierarchical Clustering: Ward linkage
FIXED_PROPORTION = 0            # Data Sampler: "Fixed proportion of data"
TRAIN_PERCENT = 80
CROSS_VALIDATION, TEST_ON_TEST = 0, 5       # Test and Score sampling
TEN_FOLDS = 3                   # index into [2, 3, 5, 10, 20]

# Step 6: the settings that gave the highest cross-validated CA
# (found with ML2/tune_models.py). Empty = the widget's defaults.
TUNED = {
    "Logistic Regression": {"penalty_type": 0, "C_index": 66},  # Lasso, C=0.5
    "kNN": {"n_neighbors": 5, "metric_index": 0,        # Euclidean
            "weight_index": 1},                         # by distances
    "SVM": {"kernel_type": 0, "C": 1.0,                 # linear kernel
            "limit_iter": True, "max_iter": 1000},
    "Tree": {"limit_min_leaf": True, "min_leaf": 10,
             "limit_min_internal": True, "min_internal": 2,
             "limit_depth": True, "max_depth": 5,
             "limit_majority": True, "sufficient_majority": 100},
    "Random Forest": {"n_estimators": 200, "use_max_features": True,
                      "max_features": 2, "use_min_samples_split": True,
                      "min_samples_split": 10, "use_random_state": True},
    "Neural Network": {"hidden_layers_input": "100,50",
                       "activation_index": 2,           # tanh
                       "solver_index": 2,               # Adam
                       "alpha_index": 14, "max_iterations": 200,
                       "replicable": True},
}
MODEL_WIDGET = {"Logistic Regression": LOGISTIC, "kNN": KNN, "SVM": SVM,
                "Tree": TREE, "Random Forest": FOREST,
                "Neural Network": NEURAL}
MODELS = list(MODEL_WIDGET)
MODEL_Y = [700, 810, 920, 1140, 1250, 1360]

KM, HC = "k-Means (2 clusters)", "Hierarchical Clustering (2 clusters)"
KM_PIVOT, HC_PIVOT = "k-Means Pivot Table", "Hierarchical Pivot Table"
BOX_TITLE = "Clump Thickness by Cluster"
SCATTER_TITLE = "Clump Thickness vs Cell Size"
SPLIT = "Train/Test Split (80/20)"
TRAIN, TEST_SAMPLE = "Training Sample (80%)", "Test Sample (20%)"
DATA = "Breast Cancer Wisconsin"

# title -> (widget class, (x, y), saved settings)
NODES = {
    DATA: (DATASETS, (60, 520), {"selected_id": DATASET_ID}),
    KM: (KMEANS, (340, 60), {"k": N_CLUSTERS, "optimize_k": False,
                             "normalize": True}),
    KM_PIVOT: (PIVOT, (640, 60), {}),
    "Distances": (DISTANCES, (340, 300),
                  {"axis": ROWS, "metric_id": EUCLIDEAN_NORMALIZED}),
    HC: (HIERARCHICAL, (600, 300), {"linkage": WARD}),
    HC_PIVOT: (PIVOT, (900, 180), {}),
    BOX_TITLE: (BOX, (900, 300), {}),
    SCATTER_TITLE: (SCATTER, (900, 420), {}),
    TRAIN: (TABLE, (600, 960), {}),
    TEST_SAMPLE: (TABLE, (600, 1120), {}),
    SPLIT: (SAMPLER, (320, 1040),
            {"sampling_type": FIXED_PROPORTION,
             "sampleSizePercentage": TRAIN_PERCENT, "use_seed": True,
             "replacement": False, "stratify": True}),
    "Test and Score": (TEST, (1040, 1040), {"resampling": CROSS_VALIDATION,
                                          "n_folds": TEN_FOLDS,
                                          "cv_stratified": True}),
}
for _name, _y in zip(MODELS, MODEL_Y):
    NODES[_name] = (MODEL_WIDGET[_name], (820, _y),
                    dict(TUNED[_name], learner_name=_name))

# (source, output channel, sink, input channel)
LINKS_CLUSTER = [
    (DATA, "Data", KM, "Data"),
    (DATA, "Data", "Distances", "Data"),
    ("Distances", "Distances", HC, "Distances"),
]
LINKS_CLUSTER_OUT = [
    (KM, "Data", KM_PIVOT, "Data"),
    (HC, "Selected Data", HC_PIVOT, "Data"),
    (HC, "Selected Data", BOX_TITLE, "Data"),
    (HC, "Selected Data", SCATTER_TITLE, "Data"),
]
LINKS_MODEL = [
    (DATA, "Data", SPLIT, "Data"),
    (SPLIT, "Data Sample", TRAIN, "Data"),
    (SPLIT, "Remaining Data", TEST_SAMPLE, "Data"),
]
LINKS_TEST = [
    (TRAIN, "Selected Data", "Test and Score", "Data"),
    (TEST_SAMPLE, "Selected Data", "Test and Score", "Test Data"),
] + [(m, "Learner", "Test and Score", "Learner") for m in MODELS]

# (x, y, width, height, text)
TEXTS = [
    (-60, 260, 300, 70,
     "**Machine Problem 2**  \nClassification and Clustering"),
    (-60, 650, 250, 90,
     "**Step 1: Load the data**  \nBreast Cancer Wisconsin, 683 samples, "
     "9 features, target = type."),
    (260, -110, 330, 90,
     "**Step 2: k-Means**  \nTwo clusters. The pivot table counts benign "
     "and malign samples in each cluster."),
    (440, 430, 300, 110,
     "**Step 2: Hierarchical clustering**  \nDistances first, then the "
     "tree is cut into the top 2 clusters."),
    (1020, 100, 250, 90,
     "**Step 2: Pivot table**  \nrows = type, columns = Cluster, count."),
    (1020, 260, 250, 90,
     "**Step 3: Box plot**  \nClump thickness for each cluster."),
    (1020, 420, 260, 110,
     "**Step 4: Scatter plot**  \nClump thickness vs Unif_Cell_Size, "
     "color = type, shape = Cluster."),
    (160, 1150, 320, 110,
     "**Step 5: Split the data**  \n80% training sample and 20% test "
     "sample, each shown in a table."),
    (660, 560, 340, 90,
     "**Step 6: Models**  \nSix models that predict type, tuned for the "
     "highest CA with 10-fold cross-validation."),
    (1160, 1090, 280, 130,
     "**Steps 6-7: Test and Score**  \nCross-validation on the training "
     "sample, then Test on test data with the 20% test sample."),
]
BLUE, GREEN, RED = "#1F77B4", "#39B54A", "#C1272D"
ARROWS = [
    ((60, 650), (60, 615), RED),            # load the data
    ((340, -30), (340, 10), BLUE),          # k-Means
    ((600, 430), (600, 395), BLUE),         # hierarchical clustering
    ((1020, 135), (950, 165), GREEN),       # pivot table
    ((1020, 295), (955, 300), GREEN),       # box plot
    ((1020, 455), (955, 430), GREEN),       # scatter plot
    ((320, 1150), (320, 1115), RED),        # split
    ((820, 620), (820, 655), BLUE),         # models
    ((1160, 1110), (1095, 1070), RED),        # Test and Score
]


def output(scheme, node, channel):
    """Current value on a node's output channel."""
    sm = scheme.signal_manager
    for link in scheme.links:
        if link.source_node is node and link.source_channel.name == channel:
            contents = sm.link_contents(link)
            if contents:
                value = next(iter(contents.values()))
                return value.get_value() if hasattr(value, "get_value") \
                    else value
    raise LookupError("%s / %s" % (node.title, channel))


def set_pivot(pivot, rows, columns, values):
    """Pivot Table: rows / columns / values (aggregation stays Count)."""
    domain = pivot.data.domain
    pivot.row_feature = domain[rows]
    pivot.col_feature = domain[columns]
    pivot.val_feature = domain[values]
    pivot._OWPivot__feature_changed()
    pivot.commit.now()


def set_box(box, attribute, group):
    domain = box.dataset.domain
    box.attribute = domain[attribute]
    box.group_var = domain[group]
    box._set_list_view_selections()
    box._variables_changed(box.apply_attr_sorting)


def set_scatter(scatter, x, y, color, shape):
    domain = scatter.data.domain
    scatter.attr_x, scatter.attr_y = domain[x], domain[y]
    scatter.attr_color, scatter.attr_shape = domain[color], domain[shape]
    scatter.attr_size = scatter.attr_label = None
    scatter.graph.show_reg_line = False     # only what the step asks for
    scatter.graph.jitter_size = 0
    scatter.graph.class_density = False
    scatter.attr_changed()
    scatter.graph.update_regression_line()
    scatter.graph.update_jittering()
    scatter.graph.update_density()
    scatter.graph.update_colors()
    scatter.graph.update_shapes()


def select_all_rows(table):
    """Select every row, so the table's Selected Data is the whole sample."""
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
        title="MP2 - Classification and Clustering (Breast Cancer "
              "Wisconsin)",
        description="Clusters the Breast Cancer Wisconsin data with "
                    "k-Means and hierarchical clustering, compares the "
                    "clusters with the real labels, and trains models "
                    "that predict the type of tumor.")
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
    oh.settle(scheme, quiet=3.0, timeout=240)     # Datasets downloads
    assert widgets[DATA].selected_dataset().title == DATA

    # ---- Step 2: k-Means and hierarchical clustering, two clusters ----
    connect(LINKS_CLUSTER)
    oh.settle(scheme, quiet=3.0, timeout=240)
    hierarchical = widgets[HC]
    hierarchical.top_n = TOP_N
    hierarchical._top_n_changed()                 # selection: "Top N"
    connect(LINKS_CLUSTER_OUT)
    oh.settle(scheme, quiet=3.0, timeout=240)
    for title in (KM_PIVOT, HC_PIVOT):
        set_pivot(widgets[title], TARGET, CLUSTER, CLUMP)

    # ---- Steps 3-4: box plot and scatter plot of the clusters ----
    set_box(widgets[BOX_TITLE], CLUMP, CLUSTER)
    set_scatter(widgets[SCATTER_TITLE], CLUMP, CELL_SIZE, TARGET, CLUSTER)
    oh.settle(scheme, quiet=2.0, timeout=240)

    # ---- Steps 5-7: split, models, Test and Score ----
    connect(LINKS_MODEL)
    oh.settle(scheme, quiet=3.0, timeout=900)
    for title in (TRAIN, TEST_SAMPLE):
        select_all_rows(widgets[title])
    connect(LINKS_TEST)
    oh.settle(scheme, quiet=3.0, timeout=900)
    set_resampling(widgets["Test and Score"], TEST_ON_TEST)
    oh.settle(scheme, quiet=3.0, timeout=900)

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
