"""Build Lab2/submission/lab2_yahya.ows - the Orange workflow for Lab 2.

The workflow is put together from the real Orange widgets, run against
auto-mpg.csv so each widget can be set up on the actual data, and then saved
in Orange's own .ows format. Widget names and layout follow the sample
workflow in the lab handout.

    /Applications/Orange.app/Contents/MacOS/python Lab2/build_workflow.py
"""
import importlib
import logging
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from AnyQt.QtCore import Qt  # noqa: E402
from orangecanvas.scheme import (  # noqa: E402
    SchemeArrowAnnotation, SchemeNode, SchemeTextAnnotation,
)
from orangewidget.utils.filedialogs import RecentPath  # noqa: E402

logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

W = "Orange.widgets."
FILE = W + "data.owfile.OWFile"
TABLE = W + "data.owtable.OWTable"
STATS = W + "data.owfeaturestatistics.OWFeatureStatistics"
BOX = W + "visualize.owboxplot.OWBoxPlot"
DIST = W + "visualize.owdistributions.OWDistributions"
SCATTER = W + "visualize.owscatterplot.OWScatterPlot"
CREATE = W + "data.owcreateclass.OWCreateClass"
IMPUTE = W + "data.owimpute.OWImpute"
INDEX = W + "data.owselectbydataindex.OWSelectByDataIndex"
OUTLIERS = W + "data.owoutliers.OWOutliers"
FEATURE = W + "data.owfeatureconstructor.OWFeatureConstructor"
SAVE = W + "data.owsave.OWSave"

# Column types from the lab handout (step 1).
CATEGORICAL = ["cylinders", "model_1", "origin"]

# Create Class rules (step 2): class name, substring of model_2.
# "chev" covers chevrolet, chevy and the misspelt "chevroelt". VW needs two
# rows because the make is written both "volkswagen" and "vw". The last row
# has no substring, so it takes every car that is left.
RULES = [["FORD", "ford"], ["CHEVY", "chev"],
         ["VW", "volkswagen"], ["VW", "vw"], ["OTHER", ""]]
EXPECTED = {"FORD": 51, "CHEVY": 47, "VW": 21, "OTHER": 279}

LPER100K = "100 * 3.785411784 / (1.609344 * mpg)"

SORT_ORDER = 0         # Data Table (1): ascending
IMPUTE_MODEL = 4       # Impute: "Model-based imputer (simple tree)"
OUTLIER_METHOD = 1     # Outliers: Covariance Estimator. The default (Local
                       # Outlier Factor) does not flag the diesel Oldsmobile.
OUTLIER_CAR = "oldsmobile cutlass ciera (diesel)"

# title -> (widget class, (x, y), saved settings); names as in the handout
NODES = {
    "File": (FILE, (110, 250), {}),
    "Data Table": (TABLE, (500, 150), {}),
    "Feature Statistics": (STATS, (540, 260), {}),
    "Box Plot": (BOX, (590, 380), {}),
    "Distributions": (DIST, (470, 440), {}),
    "Scatter Plot": (SCATTER, (370, 530), {}),
    "Create Class": (CREATE, (440, 690), {}),
    "Data Table (1)": (TABLE, (700, 500), {}),
    "Impute": (IMPUTE, (700, 680), {"_default_method_index": IMPUTE_MODEL}),
    "Select by Data Index": (INDEX, (940, 300), {}),
    "Data Table (2)": (TABLE, (1170, 340), {}),
    "Scatter Plot (1)": (SCATTER, (960, 510), {}),
    "Data Table (3)": (TABLE, (1190, 470), {}),
    "Scatter Plot (2)": (SCATTER, (1180, 600), {}),
    "Outliers": (OUTLIERS, (870, 770), {"outlier_method": OUTLIER_METHOD}),
    "Feature Constructor": (FEATURE, (1090, 800), {}),
    "Scatter Plot (3)": (SCATTER, (1270, 730), {}),
    "Save Data (1)": (SAVE, (1250, 880), {}),
    "File (1)": (FILE, (210, 730), {}),
    "Step 6": (DIST, (390, 860), {"fitted_distribution": 1}),  # 1 = Normal
}

# (source, output channel, sink, input channel), in the order they are wired
LINKS_EXPLORE = [
    ("File", "Data", "Data Table", "Data"),
    ("File", "Data", "Feature Statistics", "Data"),
    ("File", "Data", "Box Plot", "Data"),
    ("File", "Data", "Distributions", "Data"),
    ("File", "Data", "Scatter Plot", "Data"),
    ("File", "Data", "Create Class", "Data"),
]
LINKS_PROCESS = [
    ("Create Class", "Data", "Data Table (1)", "Data"),
    ("Create Class", "Data", "Impute", "Data"),
    ("Data Table (1)", "Selected Data", "Select by Data Index", "Data Subset"),
    ("Impute", "Data", "Select by Data Index", "Data"),
    ("Select by Data Index", "Matching Data", "Data Table (2)", "Data"),
    ("Impute", "Data", "Scatter Plot (1)", "Data"),
    ("Scatter Plot (1)", "Selected Data", "Data Table (3)", "Data"),
    ("Impute", "Data", "Outliers", "Data"),
    ("Outliers", "Outliers", "Scatter Plot (2)", "Data"),
    ("Outliers", "Inliers", "Feature Constructor", "Data"),
    ("Feature Constructor", "Data", "Scatter Plot (3)", "Data"),
    ("Feature Constructor", "Data", "Save Data (1)", "Data"),
]
LINKS_STEP6 = [("File (1)", "Data", "Step 6", "Data")]

# (x, y, width, height, text)
TEXTS = [
    (40, 30, 380, 60,
     "**Lab 2: Data Exploration and Processing**  \nauto-mpg.csv, 398 cars"),
    (690, 110, 260, 90,
     "**Step 1: Explore**  \nLoad the file and look at it with a table, "
     "statistics and three plots."),
    (120, 590, 230, 80,
     "**Step 2: Create Class**  \ncar_make = FORD, CHEVY, VW or OTHER, "
     "from model_2."),
    (1020, 170, 290, 100,
     "**Step 3: Impute**  \nFill in the 6 missing horsepower values, then "
     "pull out just those 6 cars to compare the methods."),
    (1330, 500, 250, 110,
     "**Step 4: Outliers**  \nFind the outlier in displacement vs mpg, "
     "look it up, then let the Outliers widget flag it."),
    (1400, 760, 240, 110,
     "**Step 5: Lper100k**  \nNew column for litres per 100 km, a scatter "
     "plot against horsepower, and save to csv."),
    (120, 950, 300, 80,
     "**Step 6: Distributions by make**  \nReload the saved csv with "
     "car_make as the target."),
]
BLUE, GREEN, RED = "#1F77B4", "#39B54A", "#C1272D"
ARROWS = [
    ((690, 190), (600, 240), BLUE),       # step 1 -> explore widgets
    ((300, 670), (390, 690), BLUE),       # step 2 -> Create Class
    ((1020, 240), (985, 275), GREEN),     # step 3 -> Select by Data Index
    ((1330, 560), (1235, 590), RED),      # step 4 -> Scatter Plot (2)
    ((1400, 800), (1320, 760), RED),      # step 5 -> Scatter Plot (3)
    ((300, 950), (350, 900), BLUE),       # step 6 -> Step 6
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


def set_columns(file_widget, types=None, roles=None):
    """Change column types / roles in a File widget's column editor."""
    model = file_widget.domain_editor.model()
    for row in range(len(model.variables)):
        name = model.data(model.index(row, 0), Qt.DisplayRole)
        if types and name in types:
            model.setData(model.index(row, 1), types[name], Qt.EditRole)
        if roles and name in roles:
            model.setData(model.index(row, 2), roles[name], Qt.EditRole)
    file_widget.apply_domain_edit()


def set_scatter(widget, x, y, color=None):
    domain = widget.data.domain
    widget.attr_x, widget.attr_y = domain[x], domain[y]
    widget.attr_color = domain[color] if color else None
    widget.attr_changed()


def build():
    os.makedirs(oh.SUBMISSION, exist_ok=True)
    csv_path = os.path.join(oh.SUBMISSION, oh.DATA_NAME)
    saved_path = os.path.join(oh.SUBMISSION, oh.SAVED_NAME)

    reg = oh.registry()
    scheme = oh.new_scheme(
        title="Lab 2 - Data Exploration and Processing (auto-mpg)",
        description="Explores auto-mpg.csv, builds a car_make class, imputes "
                    "missing horsepower, removes outliers, adds Lper100k "
                    "and compares distributions by make.")
    props = {t: dict(p) for t, (_, _, p) in NODES.items()}
    props["File"]["recent_paths"] = [
        RecentPath(csv_path, "basedir", oh.DATA_NAME)]
    props["File (1)"]["recent_paths"] = [
        RecentPath(saved_path, "basedir", oh.SAVED_NAME)]
    if os.path.exists(saved_path):
        os.remove(saved_path)   # rebuilt below by the Save Data widget

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

    widgets = {t: add(t) for t in NODES if t not in ("File (1)", "Step 6")}
    oh.settle(scheme, quiet=1.0)

    # ---- Step 1: load and explore ----
    set_columns(widgets["File"],
                types={name: "categorical" for name in CATEGORICAL})
    connect(LINKS_EXPLORE)
    oh.settle(scheme, quiet=1.0)

    data = output(scheme, nodes["File"], "Data")
    domain = data.domain
    box = widgets["Box Plot"]
    box.attribute, box.group_var = domain["mpg"], domain["origin"]
    box.reset_all_data() if hasattr(box, "reset_all_data") else None
    box.update_display() if hasattr(box, "update_display") else None
    dist = widgets["Distributions"]
    dist.var, dist.cvar = domain["mpg"], None
    dist._on_var_changed()
    dist._on_cvar_changed()
    set_scatter(widgets["Scatter Plot"], "weight", "mpg", "cylinders")

    # ---- Step 2: car_make class ----
    create = widgets["Create Class"]
    create.attribute = create.data.domain["model_2"]
    create.class_name = "car_make"
    create.rules = {"model_2": [list(r) for r in RULES]}
    create.update_rules()
    create.apply()
    connect(LINKS_PROCESS)
    oh.settle(scheme, quiet=1.0)

    classed = output(scheme, nodes["Create Class"], "Data")
    make = classed.domain.class_var
    counts = {v: int((classed.Y == i).sum()) for i, v in enumerate(make.values)}
    print("car_make counts:", counts)
    assert counts == EXPECTED, counts

    # ---- Step 3: pick the cars with unknown horsepower ----
    table1 = widgets["Data Table (1)"]
    hp = classed.get_column("horsepower")
    missing_rows = [int(i) for i in np.flatnonzero(np.isnan(hp))]
    n_cols = len(classed.domain.variables) + len(classed.domain.metas)
    # sort by horsepower so the six unknown ("?") values sit together
    table1.set_sort_columns([("horsepower", SORT_ORDER)])
    table1.set_selection(missing_rows, list(range(n_cols)))
    table1.commit.now()
    oh.settle(scheme, quiet=1.0)

    # ---- Step 4: the outlier in displacement vs mpg ----
    imputed = output(scheme, nodes["Impute"], "Data")
    names = [str(v) for v in imputed.get_column("model_2")]
    outlier_row = names.index(OUTLIER_CAR)
    sp1 = widgets["Scatter Plot (1)"]
    set_scatter(sp1, "displacement", "mpg")
    oh.settle(scheme, quiet=1.0)
    sp1.graph.select_by_indices(np.array([outlier_row]))
    oh.settle(scheme, quiet=1.0)

    set_scatter(widgets["Scatter Plot (2)"], "displacement", "mpg")

    # ---- Step 5: Lper100k ----
    from Orange.widgets.data.owfeatureconstructor import ContinuousDescriptor
    fc = widgets["Feature Constructor"]
    fc.setDescriptors([ContinuousDescriptor("Lper100k", LPER100K, 3)])
    fc.currentIndex = 0
    fc.apply()
    oh.settle(scheme, quiet=1.0)
    sp3 = widgets["Scatter Plot (3)"]
    sp3.graph.show_reg_line = True   # draws the fitted line and prints r
    set_scatter(sp3, "horsepower", "Lper100k")
    sp3.graph.update_regression_line()

    save = widgets["Save Data (1)"]
    save.filter = next(f for f in save.get_filters() if "csv" in f.lower()
                       and "compressed" not in f.lower())
    save.filename = saved_path
    save.add_type_annotations = True
    save.auto_save = True
    save.save_file()
    oh.settle(scheme, quiet=1.0)
    assert os.path.exists(saved_path), "Save Data did not write the csv"

    # ---- Step 6: reload the saved csv with car_make as the target ----
    widgets["File (1)"] = add("File (1)")
    widgets["Step 6"] = add("Step 6")
    oh.settle(scheme, quiet=1.0)
    set_columns(widgets["File (1)"], roles={"car_make": "target"})
    connect(LINKS_STEP6)
    oh.settle(scheme, quiet=1.0)
    step6 = widgets["Step 6"]
    d6 = step6.data.domain
    step6.var, step6.cvar = d6["mpg"], d6["car_make"]
    step6._on_var_changed()
    step6._on_cvar_changed()
    oh.settle(scheme, quiet=1.0)

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
