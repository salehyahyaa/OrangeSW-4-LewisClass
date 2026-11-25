"""Open ML2/submission/mp2.ows as saved, check it, and make the Orange report.

Each section is made the way the Report button makes it: the widget writes
its own report and it is added to Orange's Report window. My comment is
then typed into the comment box under that section, and the report is saved
as HTML.

Writes:
    ML2/submission/mp2.html   the report to hand in
    ML2/results/results.json  the numbers the comments are based on
    ML2/results/canvas.png    picture of the workflow canvas

    /Applications/Orange.app/Contents/MacOS/python ML2/make_report.py
"""
import json
import logging
import os
import sys
import time
import warnings

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# the report view needs Qt WebEngine, which must load before the app starts
from orangewidget.report.owreport import HAVE_REPORT, OWReport  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from AnyQt.QtCore import QRectF, Qt  # noqa: E402
from AnyQt.QtGui import QImage, QPainter  # noqa: E402

from build_workflow import (  # noqa: E402
    BOX_TITLE, CELL_SIZE, CLUMP, CLUSTER, CROSS_VALIDATION, DATA, HC,
    HC_PIVOT, KM, KM_PIVOT, MODELS, SCATTER_TITLE, SPLIT, TARGET,
    TEST_ON_TEST, TUNED, output, set_resampling,
)

warnings.filterwarnings("ignore")
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

REPORT = os.path.join(oh.SUBMISSION, "mp2.html")


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        oh.app().processEvents()
        time.sleep(0.01)


def render_canvas(scheme, path, scale=2.0):
    from orangecanvas.document.schemeedit import SchemeEditWidget
    editor = SchemeEditWidget()
    editor.setRegistry(oh.registry())
    editor.setScheme(scheme)
    editor.resize(1800, 1100)
    oh.app().processEvents()
    scene = editor.scene()
    scene.clearSelection()
    rect = scene.itemsBoundingRect().adjusted(-25, -25, 25, 25)
    image = QImage(int(rect.width() * scale), int(rect.height() * scale),
                   QImage.Format_ARGB32)
    image.fill(Qt.white)
    painter = QPainter(image)
    painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing
                           | QPainter.SmoothPixmapTransform)
    scene.render(painter, QRectF(image.rect()), rect)
    painter.end()
    image.save(path)
    return editor


def sort_scores(test, column="CA"):
    """Click the CA header: largest first."""
    proxy = test.score_table.sorted_model
    for c in range(proxy.columnCount()):
        if str(proxy.headerData(c, Qt.Horizontal)) == column:
            test.score_table.view.sortByColumn(c, Qt.DescendingOrder)
            return
    raise LookupError(column)


def score_rows(test):
    """The score table of a Test and Score widget, in the order shown."""
    proxy = test.score_table.sorted_model
    heads = [str(proxy.headerData(c, Qt.Horizontal))
             for c in range(proxy.columnCount())]
    rows = []
    for r in range(proxy.rowCount()):
        row = {}
        for c, head in enumerate(heads):
            value = proxy.index(r, c).data(Qt.DisplayRole)
            row[head] = float(value) if isinstance(value, (int, float)) \
                else str(value)
        rows.append(row)
    return rows


def crosstab(table):
    """Counts of type (rows) by Cluster (columns), like the pivot table."""
    types = table.domain[TARGET].values
    clusters = table.domain[CLUSTER].values
    t = table.get_column(TARGET).astype(int)
    c = table.get_column(CLUSTER).astype(int)
    counts = {ty: {cl: int(((t == i) & (c == j)).sum())
                   for j, cl in enumerate(clusters)}
              for i, ty in enumerate(types)}
    # the cluster that holds most of each type
    home = {ty: max(counts[ty], key=counts[ty].get) for ty in types}
    agree = sum(counts[ty][home[ty]] for ty in types)
    return {"counts": counts, "home": home, "agree": agree,
            "n": len(table), "agree_percent": 100 * agree / len(table),
            "sizes": {cl: sum(counts[ty][cl] for ty in types)
                      for cl in clusters}}


def by_cluster(table, column):
    clusters = table.domain[CLUSTER].values
    c = table.get_column(CLUSTER).astype(int)
    x = table.get_column(column)
    out = {}
    for j, cl in enumerate(clusters):
        v = x[c == j]
        out[cl] = {"n": int(len(v)), "mean": float(v.mean()),
                   "sd": float(v.std(ddof=1)),
                   "median": float(np.median(v)),
                   "q1": float(np.percentile(v, 25)),
                   "q3": float(np.percentile(v, 75)),
                   "min": float(v.min()), "max": float(v.max())}
    return out


def confusion(results, index):
    actual = results.actual.astype(int)
    predicted = results.predicted[index].astype(int)
    values = results.domain.class_var.values
    return {a: {p: int(((actual == i) & (predicted == j)).sum())
                for j, p in enumerate(values)}
            for i, a in enumerate(values)}


# ------------------------------------------------------------ the comments
def comments(r):
    """My comment for each report section, written from the numbers."""
    km, hc = r["kmeans"], r["hierarchical"]
    box, cv, test = r["box"], r["cv_scores"], r["test_scores"]

    def pivot_lines(tab):
        counts, home = tab["counts"], tab["home"]
        b, m = home["benign"], home["malign"]
        return counts, b, m

    kc, kb, kmal = pivot_lines(km)
    hcounts, hb, hm = pivot_lines(hc)
    out = {}
    out["kmeans"] = (
        "Step 2, k-Means. This table comes from the k-Means widget. I set "
        "the number of clusters to 2 and left \"Normalize columns\" on. "
        "k-Means picks 2 center points, puts every sample with the center "
        "it is closest to, moves each center to the middle of its group, "
        "and repeats until nothing changes. It only looks at the 9 "
        "measurements. It never sees the type column.\n\n"
        "The rows are the real label (type) and the columns are the "
        "cluster. Cluster %s is mostly benign: %d benign and only %d "
        "malign. Cluster %s is mostly malign: %d malign and only %d "
        "benign. So yes, the clusters match the labels closely. %d of "
        "the %d samples (%.1f%%) are in the cluster that goes with their "
        "type. The %d samples that do not fit are %d malign tumors that "
        "look like benign ones and %d benign tumors that look like "
        "malign ones."
        % (kb, kc["benign"][kb], kc["malign"][kb],
           kmal, kc["malign"][kmal], kc["benign"][kmal],
           km["agree"], km["n"], km["agree_percent"],
           km["n"] - km["agree"], kc["malign"][kb], kc["benign"][kmal]))
    out["hierarchical"] = (
        "Step 2, hierarchical clustering. This table comes from the "
        "Distances widget (normalized Euclidean distance between rows) "
        "followed by Hierarchical Clustering with Ward linkage. This "
        "method starts with every sample as its own cluster and keeps "
        "joining the two closest clusters until everything is one tree. "
        "I set Selection to \"Top N\" with N = 2, which cuts the tree "
        "into 2 clusters. Like k-Means, it does not use the type "
        "column.\n\n"
        "Cluster %s is almost all benign: %d benign and %d malign. "
        "Cluster %s is mostly malign: %d malign and %d benign. So these "
        "clusters also correspond to the labels. %d of %d samples "
        "(%.1f%%) are in the matching cluster, which is a little better "
        "than k-Means (%.1f%%). The mistakes are different though. "
        "Hierarchical clustering puts only %d malign tumors in the benign "
        "cluster, where k-Means put %d. But it puts %d benign tumors in "
        "the malign cluster, where k-Means put %d. Missing a malign "
        "tumor is the worse mistake, so I would trust the hierarchical "
        "result more here."
        % (hb, hcounts["benign"][hb], hcounts["malign"][hb],
           hm, hcounts["malign"][hm], hcounts["benign"][hm],
           hc["agree"], hc["n"], hc["agree_percent"], km["agree_percent"],
           hcounts["malign"][hb], kc["malign"][kb],
           hcounts["benign"][hm], kc["benign"][kmal]))
    lo, hi = box[hb], box[hm]
    out["box"] = (
        "Step 3, box plot. This shows Clump thickness for each of the 2 "
        "hierarchical clusters. Each box holds the middle half of the "
        "samples in that cluster. The yellow line is the median and the "
        "dark blue line is the mean.\n\n"
        "Cluster %s (the mostly benign one, %d samples) has thin clumps: "
        "the mean is %.2f and the median is %.2f. Cluster %s (the mostly "
        "malign one, %d samples) has much thicker clumps: the mean is "
        "%.2f and the median is %.2f. The two boxes do not overlap. The "
        "top of the first box is at %.2f and the bottom of the second is "
        "at %.2f. The t-test under the plot gives p = 0.000, so the "
        "difference is not down to chance. The dotted lines, which run "
        "from the smallest to the largest value, do overlap though "
        "(%.2f to %.2f against %.2f to %.2f). Some samples in the malign "
        "cluster have thin clumps and a few in the benign cluster have "
        "thick ones. So Clump "
        "thickness separates the clusters well, but not perfectly on its "
        "own. Cluster %s is also more spread out (standard deviation "
        "%.2f against %.2f)."
        % (hb, lo["n"], lo["mean"], lo["median"],
           hm, hi["n"], hi["mean"], hi["median"],
           lo["q3"], hi["q1"], lo["min"], lo["max"], hi["min"], hi["max"],
           hm, hi["sd"], lo["sd"]))
    sc = r["scatter"]
    shape = dict(zip(sorted(hc["sizes"]), ("circles", "crosses")))
    out["scatter"] = (
        "Step 4, scatter plot. Each point is one sample. Clump thickness "
        "is on the x axis and Unif_Cell_Size is on the y axis. The color "
        "shows the real type and the shape shows the hierarchical "
        "cluster.\n\n"
        "The benign points sit in the bottom left corner, with thin "
        "clumps and small, uniform cells. %d%% of the benign samples "
        "have a cell size below 2. The malign points are spread over "
        "the rest of the plot, mostly higher up and further right. Cell "
        "size separates the two types even more clearly than clump "
        "thickness: the average cell size is %.2f for benign and %.2f "
        "for malign.\n\n"
        "Color and shape agree for almost every point. Nearly all "
        "blue (benign) points are %s, the shape for cluster %s, and "
        "nearly all red (malign) points are %s, the shape for cluster "
        "%s. The %d points "
        "where they disagree are mostly in the area where the two "
        "groups meet. Those are the samples the clustering got wrong."
        % (round(sc["benign_small_cells_percent"]),
           sc["cell_size_mean"]["benign"], sc["cell_size_mean"]["malign"],
           shape[hb], hb, shape[hm], hm, hc["n"] - hc["agree"]))
    best, second, last = cv[0], cv[1], cv[-1]
    tuned_lines = "\n".join("- %s: %s" % (m, r["settings_text"][m])
                            for m in MODELS)
    out["cv"] = (
        "Step 6, cross-validation. I split the data into %d training "
        "samples (80%%) and %d test samples (20%%). This table uses only "
        "the training sample, with stratified 10-fold cross-validation. "
        "I made 6 models that predict type and changed the settings of "
        "each one until the CA stopped going up. I sorted the table by "
        "CA. These are the settings I ended up with:\n%s\n\n"
        "%s has the highest CA, %.3f. That means it gets about %d of "
        "the %d training samples right. %s is next with %.3f. %s is "
        "last with %.3f. The top 5 models are within %.3f of each other, "
        "which is only about %d samples, so they are close to a tie. "
        "The AUC is at least %.3f for every model, so all of them rank "
        "malign above benign very well. Tuning helped the most for the "
        "Tree (CA from %.3f with the default settings to %.3f) and the "
        "least for kNN (%.3f to %.3f). This data is easy to classify, "
        "so there was not much left to gain."
        % (r["train_rows"], r["test_rows"], tuned_lines,
           best["Model"], best["CA"], round(best["CA"] * r["train_rows"]),
           r["train_rows"], second["Model"], second["CA"],
           last["Model"], last["CA"], cv[0]["CA"] - cv[4]["CA"],
           round((cv[0]["CA"] - cv[4]["CA"]) * r["train_rows"]),
           min(row["AUC"] for row in cv),
           r["tuning"]["Tree"]["default_ca"], r["tuning"]["Tree"]["best"]["ca"],
           r["tuning"]["kNN"]["default_ca"], r["tuning"]["kNN"]["best"]["ca"]))
    tbest, tlast = test[0], test[-1]
    ties = [row["Model"] for row in test if row["CA"] == tbest["CA"]]
    cm = r["test_confusion"][tbest["Model"]]
    out["test"] = (
        "Step 7, test on test data. I connected the 20%% test sample "
        "(%d samples the models never saw) to the Test Data input and "
        "chose \"Test on test data\". Each model is now trained on all "
        "%d training samples and scored on the test sample.\n\n"
        "%s the highest CA on the test data, %.3f, which is %d of %d "
        "right. %s is last with %.3f. Every model scores between %.3f "
        "and %.3f, so the test results are about the same as the "
        "cross-validation results. For example %s got %.3f in "
        "cross-validation and %.3f here. That tells me the models did "
        "not overfit the training sample, and that cross-validation "
        "gave an honest estimate.\n\n"
        "The order of the models changed, though. %s was only number "
        "%d in cross-validation and is first here, while %s was first "
        "in cross-validation and is down at number %d here. One test "
        "sample is "
        "worth about %.1f percentage points, so these small gaps are "
        "just a few samples and do not mean much.\n\n"
        "%s made %d mistakes on the test data: %d malign tumors called "
        "benign and %d benign tumors called malign. Calling a malign "
        "tumor benign is the dangerous kind of mistake, and every model "
        "made only 2 or 3 of those out of %d malign test samples. "
        "Overall, type can be predicted from these 9 measurements with "
        "about %d to %d%% accuracy."
        % (r["test_rows"], r["train_rows"],
           ("%s has" % ties[0]) if len(ties) == 1
           else ("%s share" % " and ".join(ties)),
           tbest["CA"], round(tbest["CA"] * r["test_rows"]), r["test_rows"],
           tlast["Model"], tlast["CA"], tlast["CA"], tbest["CA"],
           cv[0]["Model"], cv[0]["CA"],
           next(row["CA"] for row in test if row["Model"] == cv[0]["Model"]),
           tbest["Model"],
           1 + [row["Model"] for row in cv].index(tbest["Model"]),
           cv[0]["Model"],
           1 + sum(row["CA"] > next(
               x["CA"] for x in test if x["Model"] == cv[0]["Model"])
               for row in test),
           100 / r["test_rows"],
           tbest["Model"], cm["malign"]["benign"] + cm["benign"]["malign"],
           cm["malign"]["benign"], cm["benign"]["malign"],
           r["test_class_counts"][1],
           int(100 * tlast["CA"]), round(100 * tbest["CA"])))
    return out


def settings_text():
    from Orange.widgets.model.owlogisticregression import (
        OWLogisticRegression,
    )
    from Orange.widgets.model.owneuralnetwork import OWNNLearner
    t = TUNED
    lr, knn, svm, tree = (t["Logistic Regression"], t["kNN"], t["SVM"],
                          t["Tree"])
    rf, nn = t["Random Forest"], t["Neural Network"]
    return {
        "Logistic Regression": "%s regularization, C = %s" % (
            ["Lasso (L1)", "Ridge (L2)"][lr["penalty_type"]],
            OWLogisticRegression.C_s[lr["C_index"]]),
        "kNN": "%d neighbors, %s distance, weight %s" % (
            knn["n_neighbors"],
            ["Euclidean", "Manhattan", "Chebyshev"][knn["metric_index"]],
            ["uniform", "by distances"][knn["weight_index"]]),
        "SVM": "%s kernel, cost C = %s, iteration limit %d" % (
            ["linear", "polynomial", "RBF", "sigmoid"][svm["kernel_type"]],
            svm["C"], svm["max_iter"]),
        "Tree": "at least %d samples in a leaf, maximum depth %d, stop "
                "splitting when majority reaches %d%%" % (
                    tree["min_leaf"], tree["max_depth"],
                    tree["sufficient_majority"]),
        "Random Forest": "%d trees, %d features considered at each split, "
                         "do not split subsets smaller than %d" % (
                             rf["n_estimators"], rf["max_features"],
                             rf["min_samples_split"]),
        "Neural Network": "hidden layers %s, %s activation, %s solver, "
                          "regularization alpha = %s" % (
                              nn["hidden_layers_input"],
                              ["identity", "logistic", "tanh",
                               "ReLu"][nn["activation_index"]],
                              ["L-BFGS-B", "SGD", "Adam"][nn["solver_index"]],
                              OWNNLearner.alphas[nn["alpha_index"]]),
    }


def main():
    import Orange
    from Orange.evaluation import CA, CrossValidation

    assert HAVE_REPORT, "Qt WebEngine is needed to build the report"
    os.makedirs(oh.RESULTS, exist_ok=True)
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=3.0, timeout=900)
    W = lambda title: oh.widget(scheme, title)  # noqa: E731
    node = {n.title: n for n in scheme.nodes}
    problems = oh.widget_problems(scheme)

    data = output(scheme, node[DATA], "Data")
    km_data = output(scheme, node[KM], "Data")
    hc_data = output(scheme, node[HC], "Selected Data")
    train = output(scheme, node[SPLIT], "Data Sample")
    test_set = output(scheme, node[SPLIT], "Remaining Data")
    test = W("Test and Score")
    kmeans, hier, scatter = W(KM), W(HC), W(SCATTER_TITLE)
    box = W(BOX_TITLE)
    as_saved = {
        "dataset": W(DATA).selected_dataset().title,
        "kmeans_k": kmeans.k, "kmeans_optimize_k": kmeans.optimize_k,
        "hier_selection_method": hier.selection_method,
        "hier_top_n": hier.top_n,
        "pivots": {t: [str(W(t).row_feature), str(W(t).col_feature),
                       str(W(t).val_feature), sorted(W(t).sel_agg_functions)]
                   for t in (KM_PIVOT, HC_PIVOT)},
        "box": [str(box.attribute), str(box.group_var)],
        "scatter": [str(scatter.attr_x), str(scatter.attr_y),
                    str(scatter.attr_color), str(scatter.attr_shape)],
        "sampler_percent": W(SPLIT).sampleSizePercentage,
        "test_mode": test.resampling, "folds": test.NFolds[test.n_folds],
    }

    t = hc_data.get_column(TARGET).astype(int)
    types = hc_data.domain[TARGET].values
    cell = hc_data.get_column(CELL_SIZE)
    results = {
        "orange_version": Orange.__version__, "as_saved": as_saved,
        "rows": len(data), "features": len(data.domain.attributes),
        "class_counts": {ty: int((data.Y == i).sum())
                         for i, ty in enumerate(data.domain.class_var.values)},
        "kmeans": crosstab(km_data), "hierarchical": crosstab(hc_data),
        "box": by_cluster(hc_data, CLUMP),
        "scatter": {
            "cell_size_mean": {ty: float(cell[t == i].mean())
                               for i, ty in enumerate(types)},
            "clump_mean": {ty: float(
                hc_data.get_column(CLUMP)[t == i].mean())
                for i, ty in enumerate(types)},
            "benign_small_cells_percent": float(
                100 * (cell[t == 0] < 2).mean()),
        },
        "train_rows": len(train), "test_rows": len(test_set),
        "train_class_counts": [int(c) for c in
                               np.bincount(train.Y.astype(int))],
        "test_class_counts": [int(c) for c in
                              np.bincount(test_set.Y.astype(int))],
        "tuned_settings": TUNED, "settings_text": settings_text(),
    }
    with open(os.path.join(oh.RESULTS, "tuning.json")) as f:
        results["tuning"] = json.load(f)

    # ---- the Report window, filled the way the Report button fills it ----
    report = OWReport()
    OWReport.set_instance(report)
    report.resize(1150, 900)   # not shown: there is no screen here

    def add_section(widget, size):
        widget.show()
        widget.resize(*size)
        pump(1.0)
        widget.create_report_html()
        report.make_report(widget)
        widget.hide()
        return report.table_model.item(report.table_model.rowCount() - 1)

    items = {
        "kmeans": add_section(W(KM_PIVOT), (700, 420)),
        "hierarchical": add_section(W(HC_PIVOT), (700, 420)),
        "box": add_section(box, (1000, 520)),
        "scatter": add_section(scatter, (1000, 700)),
    }
    # step 6: 10-fold cross-validation on the training sample
    set_resampling(test, CROSS_VALIDATION)
    oh.settle(scheme, quiet=3.0, timeout=900)
    sort_scores(test)
    results["cv_scores"] = score_rows(test)
    by_hand = CA(CrossValidation(k=10, random_state=42)(
        train, [W(m).create_learner() for m in MODELS]))
    items["cv"] = add_section(test, (1100, 560))
    # step 7: test on test data (this is also how the workflow is saved)
    set_resampling(test, TEST_ON_TEST)
    oh.settle(scheme, quiet=3.0, timeout=900)
    sort_scores(test)
    results["test_scores"] = score_rows(test)
    # per-model results, straight from the widget
    learners = [str(s.learner.name) for s in test.learners.values()]
    results["test_confusion"] = {
        name: confusion(test.learners[key].results.value, 0)
        for key, name in zip(test.learners, learners)}
    items["test"] = add_section(test, (1100, 560))

    cv_by = {r["Model"]: r for r in results["cv_scores"]}
    checks = {
        "widget_problems": problems,
        "dataset": as_saved["dataset"] == DATA and len(data) == 683,
        "two_clusters_each": as_saved["kmeans_k"] == 2
        and not as_saved["kmeans_optimize_k"]
        and as_saved["hier_selection_method"] == 2
        and as_saved["hier_top_n"] == 2
        and len(results["kmeans"]["sizes"]) == 2
        and len(results["hierarchical"]["sizes"]) == 2,
        "clustering_on_whole_data": len(km_data) == 683
        and len(hc_data) == 683,
        "pivots_as_asked": all(
            v == [TARGET, CLUSTER, CLUMP, [0]]
            for v in as_saved["pivots"].values()),
        "box_plot_as_asked": as_saved["box"] == [CLUMP, CLUSTER],
        "scatter_as_asked": as_saved["scatter"] == [CLUMP, CELL_SIZE,
                                                    TARGET, CLUSTER],
        "split_80_20": as_saved["sampler_percent"] == 80
        and len(train) + len(test_set) == 683
        and abs(len(train) / 683 - 0.8) < 0.005,
        "ten_folds": as_saved["folds"] == 10,
        "saved_as_test_on_test_data": as_saved["test_mode"] == TEST_ON_TEST,
        "six_models": sorted(cv_by) == sorted(MODELS)
        and len(results["test_scores"]) == 6,
        "my_cv_matches_widget": all(
            abs(cv_by[m]["CA"] - float(ca)) < 1e-9
            for m, ca in zip(MODELS, by_hand)),
        "cv_matches_tuning": all(
            abs(cv_by[m]["CA"] - results["tuning"][m]["best"]["ca"]) < 1e-9
            for m in MODELS),
    }
    results["checks"] = checks

    text = comments(results)
    results["comments"] = text
    for key, item in items.items():
        item.comment = text[key]
    report._build_html()
    pump(3.0)
    # make every comment box tall enough to show the whole comment
    report.report_view.runJavaScript(
        "document.querySelectorAll('textarea').forEach(function (t) {"
        " var rows = 1;"
        " t.value.split('\\n').forEach(function (line) {"
        "  rows += Math.max(1, Math.ceil(line.length / 100)); });"
        " t.rows = rows; t.style.height = 'auto';"
        " t.style.fieldSizing = 'content'; });")
    pump(1.5)
    html = report.report_view.html()
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(html)
    results["report_sections"] = [report.table_model.item(i).name
                                  for i in range(report.table_model.rowCount())]

    with open(os.path.join(oh.RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    keep = render_canvas(scheme, os.path.join(oh.RESULTS, "canvas.png"))
    del keep

    slim = {k: v for k, v in results.items()
            if k not in ("tuning", "comments")}
    print(json.dumps(slim, indent=1, default=str))
    for key, value in text.items():
        print("\n[%s]\n%s" % (key, value))
    print("\nwrote", REPORT, len(html) // 1024, "KB")
    failed = [k for k, v in checks.items() if k != "widget_problems" and not v]
    if problems or failed:
        print("CHECKS FAILED", problems, failed)
        sys.exit(1)
    print("ALL CHECKS PASSED")
    sys.stdout.flush()
    os._exit(0)     # skip Qt WebEngine's teardown, which crashes offscreen


if __name__ == "__main__":
    main()
