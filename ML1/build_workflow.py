"""Build submission/mp1_yahya.ows - the Orange Canvas workflow for MP1.

The workflow is assembled from real Orange widgets, run once so that each
widget can be configured against the actual data, and then saved in Orange's
own .ows format. Open the result in Orange to see and re-run every step.

    /Applications/Orange.app/Contents/MacOS/python ML1/build_workflow.py
"""
import importlib
import logging
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from orangecanvas.scheme import (  # noqa: E402
    SchemeArrowAnnotation, SchemeNode, SchemeTextAnnotation,
)

# Orange logs a traceback each time it stores settings with pickle instead of
# as a text literal; that is normal for widgets with per-dataset settings.
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

W = "Orange.widgets."
FILE = W + "data.owcsvimport.OWCSVFileImport"
TABLE = W + "data.owtable.OWTable"
STATS = W + "data.owfeaturestatistics.OWFeatureStatistics"
DIST = W + "visualize.owdistributions.OWDistributions"
SELECT = W + "data.owselectcolumns.OWSelectAttributes"
IMPUTE = W + "data.owimpute.OWImpute"
CONT = W + "data.owcontinuize.OWContinuize"
SAMPLER = W + "data.owdatasampler.OWDataSampler"
CORR = W + "data.owcorrelations.OWCorrelations"
SCRIPT = W + "data.owpythonscript.OWPythonScript"
LINREG = W + "model.owlinearregression.OWLinearRegression"
TEST = W + "evaluate.owtestandscore.OWTestAndScore"

# Features used by the model, and the dependent variable.
FEATURES = ["Primary Type", "Domestic", "Community Area"]
TARGET = "Arrest"

# Column types for the 22 CSV columns, in file order.
CSV_COLUMN_TYPES = [
    "Numeric",      # ID
    "Text",         # Case Number
    "Time",         # Date
    "Text",         # Block
    "Categorical",  # IUCR
    "Categorical",  # Primary Type
    "Categorical",  # Description
    "Categorical",  # Location Description
    "Categorical",  # Arrest
    "Categorical",  # Domestic
    "Numeric",      # Beat
    "Numeric",      # District
    "Numeric",      # Ward
    "Categorical",  # Community Area (an area code 1-77, not a quantity)
    "Categorical",  # FBI Code
    "Numeric",      # X Coordinate
    "Numeric",      # Y Coordinate
    "Numeric",      # Year
    "Time",         # Updated On
    "Numeric",      # Latitude
    "Numeric",      # Longitude
    "Text",         # Location
]

TOP_CORR_SCRIPT = '''\
# Keep the 10 feature pairs with the strongest Pearson correlation.
# "Strongest" = largest absolute value, so strong negative
# correlations are kept as well as strong positive ones.
import numpy as np

r = in_data.get_column("Correlation")
order = np.argsort(-np.abs(r), kind="stable")[:10]
out_data = in_data[order]
out_data.name = "Top 10 correlations with Arrest"
'''

TOP_COEF_SCRIPT = '''\
# Keep the 5 most positive and the 5 most negative coefficients
# of the linear regression model (the intercept is not a feature,
# so it is left out of the ranking).
import numpy as np

coef = in_data.get_column("coef")
names = [str(n) for n in in_data.get_column("name")]
rows = np.array([i for i, n in enumerate(names) if n != "intercept"])
order = rows[np.argsort(coef[rows], kind="stable")]
top_positive = order[::-1][:5]
top_negative = order[:5]
out_data = in_data[np.concatenate([top_positive, top_negative])]
out_data.name = "Top 5 positive / top 5 negative coefficients"
'''


def script_props(name, text):
    return {
        "scriptLibrary": [{"name": name, "script": text, "filename": None}],
        "currentScriptIndex": 0,
        "scriptText": text,
    }


# title -> (widget class, (x, y), saved settings)
LOAD = "Load ChicagoCrimes2019.csv"
KEEP = "Keep Model Columns"
DROPNA = "Drop Rows with Missing Values"
DUMMY = "Create Dummy Variables"
SPLIT = "Split 80% Train / 20% Test"
CORRW = "Pearson Correlations vs Arrest"
PICK_CORR = "Pick Top 10 Correlations"
BASE = "Drop Baseline Dummy Columns"
MODEL = "Linear Regression (Arrest)"
SCORE = "Test and Score (Test Sample)"
PICK_COEF = "Pick Top 5 + / Top 5 - Coefficients"

NODES = {
    # 1) load
    # (CSV File Import rather than File: File refuses to auto-load files
    # over 10 MB, and this one is 56 MB. Configured in build() below.)
    LOAD: (FILE, (60, 430), {}),
    # 2) explore
    "Raw Crime Data": (TABLE, (250, 130), {}),
    "Feature Statistics (Missing Values)": (STATS, (250, 250), {}),
    "Arrests by Crime Type": (DIST, (250, 370), {"stacked_columns": True}),
    # 2b) clean, 3) dummy variables
    KEEP: (SELECT, (250, 570), {}),
    DROPNA: (IMPUTE, (420, 570), {"_default_method_index": 6}),  # 6 = drop
    DUMMY: (CONT, (590, 570), {
        # 1 = one-hot (a 0/1 column per value) for multi-valued categories;
        # 2 = first value as base: a TRUE/FALSE column becomes one 0/1 column
        "disc_var_hints": {"": 1, "Domestic": 2, TARGET: 2},
        "cont_var_hints": {"": 0},
    }),
    "Dummy-Coded Data": (TABLE, (720, 430), {}),
    # 4) split
    SPLIT: (SAMPLER, (850, 570), {
        "sampling_type": 0, "sampleSizePercentage": 80,
        "use_seed": True, "stratify": False, "replacement": False,
    }),
    "Train Sample (80%)": (TABLE, (1000, 700), {}),
    "Test Sample (20%)": (TABLE, (1000, 820), {}),
    # 5) correlations
    CORRW: (CORR, (1160, 250), {"correlation_type": 0}),
    "All Correlations": (TABLE, (1340, 130), {}),
    PICK_CORR: (SCRIPT, (1340, 250), script_props(
        "Top 10 correlations", TOP_CORR_SCRIPT)),
    "Top Correlations": (TABLE, (1520, 250), {}),
    # 6) model + evaluation
    BASE: (SELECT, (1160, 570), {}),
    MODEL: (LINREG, (1340, 660), {
        "reg_type": 0, "fit_intercept": True,
        "learner_name": "Linear Regression",
    }),
    SCORE: (TEST, (1540, 540), {"resampling": 5}),  # 5 = test on test data
    # 7) coefficients
    "Model Coefficients": (TABLE, (1540, 700), {}),
    PICK_COEF: (SCRIPT, (1540, 840), script_props(
        "Top coefficients", TOP_COEF_SCRIPT)),
    "Top Coefficients": (TABLE, (1730, 840), {}),
}

# (source title, output channel, sink title, input channel)
LOAD_LINKS = 4  # the first four links fan the loaded data out
LINKS = [
    (LOAD, "Data", "Raw Crime Data", "Data"),
    (LOAD, "Data", "Feature Statistics (Missing Values)", "Data"),
    (LOAD, "Data", "Arrests by Crime Type", "Data"),
    (LOAD, "Data", KEEP, "Data"),
    (KEEP, "Data", DROPNA, "Data"),
    (DROPNA, "Data", DUMMY, "Data"),
    (DUMMY, "Data", "Dummy-Coded Data", "Data"),
    (DUMMY, "Data", SPLIT, "Data"),
    (SPLIT, "Data Sample", "Train Sample (80%)", "Data"),
    (SPLIT, "Remaining Data", "Test Sample (20%)", "Data"),
    (SPLIT, "Data Sample", CORRW, "Data"),
    (CORRW, "Correlations", "All Correlations", "Data"),
    (CORRW, "Correlations", PICK_CORR, "Data"),
    (PICK_CORR, "Data", "Top Correlations", "Data"),
    (SPLIT, "Data Sample", BASE, "Data"),
    (BASE, "Data", MODEL, "Data"),
    (BASE, "Data", SCORE, "Data"),
    (SPLIT, "Remaining Data", SCORE, "Test Data"),
    (MODEL, "Learner", SCORE, "Learner"),
    (MODEL, "Coefficients", "Model Coefficients", "Data"),
    (MODEL, "Coefficients", PICK_COEF, "Data"),
    (PICK_COEF, "Data", "Top Coefficients", "Data"),
]

# (x, y, width, height, markdown text)
TEXTS = [
    (20, 15, 520, 60,
     "**Chicago Crimes 2019 - what predicts an arrest?**  \n"
     "MP1 workflow. Follow the numbered steps from left to right."),
    (15, 250, 150, 110,
     "**1. Load**  \n258,506 crime records with 22 attributes."),
    (345, 140, 215, 150,
     "**2. Explore**  \nRaw rows, missing-value counts per column, and "
     "the arrest rate for each crime type, to spot data problems."),
    (180, 655, 330, 150,
     "**2b. Clean**  \nKeep Primary Type, Domestic and Community Area as "
     "features and Arrest as the target; the other 18 columns are not "
     "used. Rows with a missing value are dropped."),
    (480, 250, 210, 110,
     "**3. Dummy variables**  \nOne 0/1 column per crime type and per "
     "community area; Arrest and Domestic become 0/1."),
    (730, 325, 190, 110,
     "**4. Split**  \n80% training sample, 20% test sample, with a fixed "
     "random seed so results repeat."),
    (1080, 15, 420, 85,
     "**5. Correlations**  \nPearson correlation of every feature with "
     "Arrest on the training sample. The 10 strongest are sent to "
     "*Top Correlations*."),
    (1650, 400, 250, 170,
     "**6. Model**  \nLinear regression with Arrest as the dependent "
     "variable, trained on the training sample and scored on the test "
     "sample. One dummy per category is dropped first as the baseline, "
     "so the coefficients are well defined."),
    (1060, 900, 400, 85,
     "**7. Coefficients**  \nEvery coefficient of the trained model, then "
     "the 5 most positive and 5 most negative in *Top Coefficients*."),
]

# (start, end, colour)
BLUE, GREEN, RED = "#1F77B4", "#39B54A", "#C1272D"
ARROWS = [
    ((75, 345), (62, 392), BLUE),          # 1 -> load
    ((345, 215), (300, 215), BLUE),        # 2 -> exploration widgets
    ((255, 655), (252, 628), BLUE),        # 2b -> Keep Model Columns
    ((650, 360), (700, 400), BLUE),        # 3 -> Dummy-coded data
    ((850, 440), (850, 528), BLUE),        # 4 -> split
    ((1165, 100), (1162, 208), GREEN),     # 5 -> Correlations
    ((1650, 500), (1592, 530), RED),       # 6 -> Test and Score
    ((1465, 920), (1500, 862), RED),       # 7 -> coefficient picker
]


def sel_out(scheme, node, channel="Data"):
    """Current value on a node's output channel."""
    sm = scheme.signal_manager
    for link in scheme.links:
        if link.source_node is node and link.source_channel.name == channel:
            contents = sm.link_contents(link)
            if contents:
                return next(iter(contents.values()))
    raise LookupError(node.title)


def build():
    os.makedirs(oh.SUBMISSION, exist_ok=True)
    src = os.path.join(oh.REPO, "data", oh.DATA_NAME)
    dst = os.path.join(oh.SUBMISSION, oh.DATA_NAME)
    if not os.path.exists(dst):
        shutil.copyfile(src, dst)  # the File widget looks next to the .ows

    reg = oh.registry()
    scheme = oh.new_scheme(
        title="MP1 - Chicago Crimes 2019: predicting arrests",
        description="Loads the 2019 Chicago crime records, cleans and "
                    "dummy-codes them, splits 80/20, ranks Pearson "
                    "correlations with Arrest and fits a linear regression.")
    nodes = {}
    for title, (qualified_name, pos, props) in NODES.items():
        props = dict(props)
        if props:
            # Stamp the settings with the widget's current settings version,
            # otherwise Orange "migrates" them as if they were from an old
            # release and shifts some option numbers.
            module, cls_name = qualified_name.rsplit(".", 1)
            cls = getattr(importlib.import_module(module), cls_name)
            props.setdefault("__version__", cls.settings_version)
        node = SchemeNode(reg.widget(qualified_name), title=title,
                          position=pos, properties=props)
        scheme.add_node(node)
        nodes[title] = node

    def connect(links):
        for src_title, out_name, dst_title, in_name in links:
            a, b = nodes[src_title], nodes[dst_title]
            scheme.new_link(a, a.output_channel(out_name),
                            b, b.input_channel(in_name))

    # Links are added in stages so each cleaning widget is configured before
    # the full 258,506 x 22 table can flow into the steps after it.
    connect(LINKS[:LOAD_LINKS])
    for x, y, w, h, text in TEXTS:
        scheme.add_annotation(SchemeTextAnnotation(
            (x, y, w, h), text, "text/markdown",
            font={"family": "Helvetica", "size": 14}))
    for start, end, color in ARROWS:
        scheme.add_annotation(SchemeArrowAnnotation(start, end, color))

    widgets = {t: scheme.widget_for_node(n) for t, n in nodes.items()}

    # --- Load: tell the importer what each CSV column is ---
    # Left on "auto", the importer reads Community Area and the coordinate
    # columns as text because some of their cells are blank. Community Area
    # is set to Categorical: it is an area code, not a quantity.
    from Orange.widgets.data.owcsvimport import (
        ColumnType, default_options_for_mime_type)
    options = default_options_for_mime_type(dst, "text/csv")
    options.columntypes = [
        (range(i, i + 1), getattr(ColumnType, kind))
        for i, kind in enumerate(CSV_COLUMN_TYPES)]
    loader = widgets[LOAD]
    loader.set_selected_file(dst, options)
    # Store the path relative to the workflow file ("basedir"), so the
    # workflow finds the CSV on any computer as long as it sits next to the
    # .ows, and no personal folder path is saved inside the submission.
    from Orange.widgets.utils.pathutils import VarPath
    loader.current_item().setVarPath(VarPath("basedir", oh.DATA_NAME))
    loader.dialog_state = {"directory": "", "filter": ""}
    oh.settle(scheme)

    # --- Keep Model Columns: 3 features + Arrest as target, drop the rest ---
    sel = widgets[KEEP]
    domain = sel.data.domain
    everything = list(domain.variables) + list(domain.metas)
    sel.used_attrs[:] = [domain[n] for n in FEATURES]
    sel.class_attrs[:] = [domain[TARGET]]
    sel.meta_attrs[:] = []
    sel.available_attrs[:] = [
        v for v in everything if v.name not in FEATURES + [TARGET]]
    sel.update_domain_role_hints()
    sel.commit.now()
    connect(LINKS[LOAD_LINKS:])
    oh.settle(scheme)

    # --- Drop Baseline Dummy Columns ---
    # With a 0/1 column for *every* crime type and *every* community area,
    # the columns of each group add up to 1 and duplicate the intercept
    # (the "dummy variable trap"), which leaves the coefficients undefined.
    # Dropping the most common level of each group makes it the baseline:
    # every coefficient is then "compared with that level".
    clean = sel_out(scheme, nodes[DROPNA])
    baselines = []
    for name in ("Primary Type", "Community Area"):
        var = clean.domain[name]
        counts = np.bincount(clean.get_column(var).astype(int),
                             minlength=len(var.values))
        baselines.append("%s=%s" % (name, var.values[int(counts.argmax())]))
    base = widgets[BASE]
    bdomain = base.data.domain
    base.used_attrs[:] = [a for a in bdomain.attributes
                          if a.name not in baselines]
    base.class_attrs[:] = list(bdomain.class_vars)
    base.meta_attrs[:] = []
    base.available_attrs[:] = [bdomain[n] for n in baselines]
    base.update_domain_role_hints()
    base.commit.now()
    oh.settle(scheme)
    print("baseline levels:", baselines)

    # --- Arrests by Crime Type: bars per crime type, split by Arrest ---
    dist = widgets["Arrests by Crime Type"]
    ddomain = dist.data.domain
    dist.var = ddomain["Primary Type"]
    dist.cvar = ddomain[TARGET]
    dist._on_var_changed()
    dist._on_cvar_changed()

    # --- Test and Score: hide the percentage-error scores ---
    # MAPE and sMAPE divide by the true value, which is 0 for every
    # non-arrest, so they are meaningless for a 0/1 target. MSE, RMSE, MAE and R2 stay visible.
    test = widgets[SCORE]
    test.score_table.show_score_hints["MAPE"] = False
    test.score_table.show_score_hints["SMAPE"] = False

    oh.settle(scheme)
    problems = oh.widget_problems(scheme)
    scheme.sync_node_properties()
    with open(oh.WORKFLOW, "wb") as f:
        scheme.save_to(f, pretty=True, pickle_fallback=True)
    print("saved", oh.WORKFLOW)
    print("widget problems:", problems or "none")
    return scheme


if __name__ == "__main__":
    build()
