"""Build Lab3/submission/lab3_yahya.ows - the Orange workflow for Lab 3.

The workflow is put together from the real Orange widgets, run against the
Housing dataset so each widget can be set up on the actual data, and then
saved in Orange's own .ows format. The layout follows the sample workflow in
the lab handout.

    /Applications/Orange.app/Contents/MacOS/python Lab3/build_workflow.py
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
from orangewidget.utils.filedialogs import RecentPath  # noqa: E402

logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

W = "Orange.widgets."
FILE = W + "data.owfile.OWFile"
TABLE = W + "data.owtable.OWTable"
BOX = W + "visualize.owboxplot.OWBoxPlot"
DIST = W + "visualize.owdistributions.OWDistributions"
SCATTER = W + "visualize.owscatterplot.OWScatterPlot"
SELECT = W + "data.owselectcolumns.OWSelectAttributes"
CORR = W + "data.owcorrelations.OWCorrelations"
FORMULA = W + "data.owfeatureconstructor.OWFeatureConstructor"
DISCRETIZE = W + "data.owdiscretize.OWDiscretize"

TARGET = "MEDV"
TOP_POSITIVE = "RM"        # highest positive correlation with MEDV (step 3)
STRONGEST = "LSTAT"        # strongest correlation with MEDV overall (step 5)
# Formula widget columns (step 5): name, expression
FORMULAS = [("log2_LSTAT", "log2(LSTAT)"), ("log2_RM", "log2(RM)")]

# title -> (widget class, (x, y), saved settings)
NODES = {
    "File": (FILE, (100, 330), {}),
    "Data Table": (TABLE, (400, 80), {}),
    "Box Plot": (BOX, (400, 180), {}),
    "Distributions": (DIST, (400, 280), {"fitted_distribution": 0}),
    "Select Columns": (SELECT, (280, 560), {}),
    "Correlations": (CORR, (500, 560), {}),
    "Scatter Plot": (SCATTER, (700, 560), {}),
    "Formula": (FORMULA, (440, 700), {}),
    "Correlations (log2)": (CORR, (660, 700), {}),
    "Scatter Plot (log2)": (SCATTER, (880, 700), {}),
    "Discretize (CRIM)": (DISCRETIZE, (760, 400), {}),
    "Distributions (MEDV by CRIM)": (DIST, (1000, 300),
                                     {"fitted_distribution": 1,   # Normal
                                      "stacked_columns": True}),
    "Box Plot (MEDV by CRIM)": (BOX, (1000, 440), {}),
    "Discretize (RM)": (DISCRETIZE, (280, 880), {}),
    "Box Plot (by RM)": (BOX, (520, 880), {}),
}

# (source, output channel, sink, input channel), in the order they are wired
LINKS_EXPLORE = [
    ("File", "Data", "Data Table", "Data"),
    ("File", "Data", "Box Plot", "Data"),
    ("File", "Data", "Distributions", "Data"),
    ("File", "Data", "Select Columns", "Data"),
    ("File", "Data", "Discretize (CRIM)", "Data"),
    ("File", "Data", "Discretize (RM)", "Data"),
]
LINKS_CORRELATION = [
    ("Select Columns", "Data", "Correlations", "Data"),
    ("Correlations", "Data", "Scatter Plot", "Data"),
    ("Correlations", "Features", "Scatter Plot", "Features"),
    ("Select Columns", "Data", "Formula", "Data"),
    ("Formula", "Data", "Correlations (log2)", "Data"),
    ("Correlations (log2)", "Data", "Scatter Plot (log2)", "Data"),
    ("Correlations (log2)", "Features", "Scatter Plot (log2)", "Features"),
]
LINKS_TESTS = [
    ("Discretize (CRIM)", "Data", "Distributions (MEDV by CRIM)", "Data"),
    ("Discretize (CRIM)", "Data", "Box Plot (MEDV by CRIM)", "Data"),
    ("Discretize (RM)", "Data", "Box Plot (by RM)", "Data"),
]

# (x, y, width, height, text)
TEXTS = [
    (30, 10, 420, 50,
     "**Lab 3: Correlations and Hypothesis Testing**  \nHousing dataset, "
     "506 towns"),
    (540, 40, 240, 70,
     "**Step 1: Initial Data Exploration**  \nTable, box plot and "
     "distribution of the raw data."),
    (20, 610, 150, 110,
     "**Step 2: Select Columns**  \nMEDV moved from target to a feature."),
    (800, 550, 240, 80,
     "**Steps 3-4: Correlations**  \nPearson correlations with MEDV and the "
     "scatter plot of the chosen pair."),
    (800, 800, 290, 80,
     "**Step 5: Formula**  \nlog2 of the feature most correlated to MEDV, "
     "then correlations and scatter plot again."),
    (1100, 330, 240, 100,
     "**Steps 6-7: Discretize CRIM**  \nCRIM split into 2 equal-frequency "
     "groups, then MEDV compared between them."),
    (640, 900, 280, 80,
     "**Step 8: Discretize RM**  \nRM split into 2 equal-frequency groups, "
     "then every attribute tested."),
]
BLUE, GREEN, RED = "#1F77B4", "#39B54A", "#C1272D"
ARROWS = [
    ((540, 60), (455, 78), RED),           # step 1 -> Data Table
    ((540, 90), (455, 170), RED),          # step 1 -> Box Plot
    ((550, 110), (455, 265), RED),         # step 1 -> Distributions
    ((130, 610), (235, 575), BLUE),        # step 2 -> Select Columns
    ((800, 580), (750, 565), GREEN),       # steps 3-4 -> Scatter Plot
    ((800, 820), (735, 765), GREEN),       # step 5 -> Correlations (log2)
    ((1100, 390), (1055, 425), BLUE),      # steps 6-7 -> Box Plot (CRIM)
    ((640, 920), (575, 895), BLUE),        # step 8 -> Box Plot (by RM)
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
    """Pick the variable and the subgroup in a Box Plot widget."""
    domain = box.dataset.domain
    box.attribute = domain[attribute]
    box.group_var = domain[group] if group else None
    box._set_list_view_selections()
    box._variables_changed(box.apply_attr_sorting)


def set_distribution(dist, var, split=None, bins=None):
    domain = dist.data.domain
    dist.var = domain[var]
    dist.cvar = domain[split] if split else None
    dist._on_var_changed()
    dist._on_cvar_changed()
    if bins is not None:          # position of the "Bin width" slider
        dist.number_of_bins = bins
        dist._on_bins_changed()


def pick_pair(corr, a, b, text=TARGET):
    """Type `text` in the Correlations filter box and select the pair a-b."""
    corr.vizrank.filter.setText(text)
    domain = corr.actual_data.domain
    corr.selection = [domain[a], domain[b]]
    corr._vizrank_select()
    corr.commit()


def discretize_one(widget, name, intervals=2):
    """Equal-frequency discretization for one variable, keep the rest."""
    from Orange.widgets.data.owdiscretize import (
        DefaultHint, DefaultKey, Methods, VarHint, variable_key)
    var = widget.data.domain[name]
    widget.var_hints = {DefaultKey: DefaultHint,          # "Keep numeric"
                        variable_key(var): VarHint(Methods.EqualFreq,
                                                   (intervals,))}
    widget.discretized_vars = {}
    widget._update_discretizations()
    widget._update_default_model()
    widget.commit.now()


def build():
    os.makedirs(oh.SUBMISSION, exist_ok=True)
    reg = oh.registry()
    scheme = oh.new_scheme(
        title="Lab 3 - Correlations and Hypothesis Testing (Housing)",
        description="Explores the Housing dataset, finds the attributes "
                    "most correlated with MEDV, builds a log2 feature, and "
                    "uses box plots of discretized CRIM and RM for t-tests.")
    props = {t: dict(p) for t, (_, _, p) in NODES.items()}
    # Orange's own copy of the Housing data ("Browse documentation datasets"
    # in the File widget), so the workflow opens on any machine.
    props["File"]["recent_paths"] = [
        RecentPath("", "sample-datasets", oh.DATA_NAME)]

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
    oh.settle(scheme, quiet=1.0)

    # ---- Step 1: load and explore ----
    connect(LINKS_EXPLORE)
    oh.settle(scheme, quiet=1.0)
    data = output(scheme, nodes["File"], "Data")
    assert len(data) == 506 and data.domain.class_var.name == TARGET
    set_box(widgets["Box Plot"], TARGET)
    set_distribution(widgets["Distributions"], TARGET, bins=0)

    # ---- Step 2: MEDV becomes an ordinary feature ----
    select = widgets["Select Columns"]
    moved = list(select.class_attrs)
    select.class_attrs[:] = []
    select.used_attrs.extend(moved)
    select.commit.now()
    connect(LINKS_CORRELATION)
    oh.settle(scheme, quiet=1.5)
    selected = output(scheme, nodes["Select Columns"], "Data")
    assert selected.domain.class_var is None
    assert TARGET in [a.name for a in selected.domain.attributes]

    # ---- Steps 3-4: correlations with MEDV, highest positive pair ----
    pick_pair(widgets["Correlations"], TARGET, TOP_POSITIVE)
    oh.settle(scheme, quiet=1.0)

    # ---- Step 5: log2 feature ----
    from Orange.widgets.data.owfeatureconstructor import ContinuousDescriptor
    formula = widgets["Formula"]
    formula.setDescriptors(
        [ContinuousDescriptor(name, expr, 3) for name, expr in FORMULAS])
    formula.currentIndex = 0
    formula.apply()
    oh.settle(scheme, quiet=1.5)
    pick_pair(widgets["Correlations (log2)"], TARGET, "log2_" + STRONGEST)
    oh.settle(scheme, quiet=1.0)
    for name in ("Scatter Plot", "Scatter Plot (log2)"):
        sp = widgets[name]
        sp.graph.show_reg_line = True     # fitted line with r printed on it
        sp.graph.update_regression_line()

    # ---- Steps 6-8: discretize, then compare groups ----
    discretize_one(widgets["Discretize (CRIM)"], "CRIM")
    discretize_one(widgets["Discretize (RM)"], "RM")
    connect(LINKS_TESTS)
    oh.settle(scheme, quiet=1.5)
    set_distribution(widgets["Distributions (MEDV by CRIM)"], TARGET, "CRIM")
    set_box(widgets["Box Plot (MEDV by CRIM)"], TARGET, "CRIM")
    set_box(widgets["Box Plot (by RM)"], TARGET, "RM")
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
