"""Open Lab3/submission/lab3_yahya.ows as saved, run it, check it, and
capture what each widget shows.

Writes to Lab3/results/:
    results.json   the numbers quoted in the report
    canvas.png     picture of the workflow canvas
    *.png          pictures of the Orange widgets used in each step

    /Applications/Orange.app/Contents/MacOS/python Lab3/run_workflow.py
"""
import json
import logging
import os
import re
import sys

import numpy as np
from scipy import stats as sps

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from AnyQt.QtCore import QRectF, Qt  # noqa: E402
from AnyQt.QtGui import QImage, QPainter  # noqa: E402

from build_workflow import (  # noqa: E402
    FORMULAS, STRONGEST, TARGET, TOP_POSITIVE, pick_pair, set_box,
    set_distribution,
)

logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

ALPHA = 0.05


def outputs(scheme):
    sm = scheme.signal_manager
    out = {}
    for link in scheme.links:
        contents = sm.link_contents(link)
        if contents:
            out[(link.source_node.title, link.source_channel.name)] = \
                next(iter(contents.values()))
    return out


def grab(w, name, size):
    w.show()
    w.resize(*size)
    for _ in range(40):
        oh.app().processEvents()
    w.grab().save(os.path.join(oh.RESULTS, name + ".png"))
    w.hide()


def render_canvas(scheme, path, scale=2.0):
    from orangecanvas.document.schemeedit import SchemeEditWidget
    editor = SchemeEditWidget()
    editor.setRegistry(oh.registry())
    editor.setScheme(scheme)
    editor.resize(1800, 1100)
    oh.app().processEvents()
    scene = editor.scene()
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


def rank_rows(corr):
    """The rows the Correlations widget is listing right now, as text."""
    model = corr.vizrank.rank_table.model()
    return [[str(model.data(model.index(r, c), Qt.DisplayRole))
             for c in range(model.columnCount())]
            for r in range(model.rowCount())]


def pearson_with(table, target):
    y = table.get_column(target)
    out = {}
    for var in table.domain.attributes:
        if var.name != target and var.is_continuous:
            r, p = sps.pearsonr(table.get_column(var), y)
            out[var.name] = {"r": float(r), "p": float(p)}
    return out


def group_stats(table, value, group):
    gvar = table.domain[group]
    g, v = table.get_column(gvar), table.get_column(value)
    return {label: {"n": int((g == i).sum()),
                    "mean": float(v[g == i].mean()),
                    "sd": float(v[g == i].std(ddof=1)),
                    "median": float(np.median(v[g == i]))}
            for i, label in enumerate(gvar.values)}


def welch(table, value, group):
    gvar = table.domain[group]
    g, v = table.get_column(gvar), table.get_column(value)
    t, p = sps.ttest_ind(v[g == 0], v[g == 1], equal_var=False)
    return float(t), float(p)


def shown(text):
    """(t, p) as printed under the plot by the Box Plot widget."""
    found = re.search(r"([\d.]+) \(p=([\d.]+)", text)
    return (float(found.group(1)), float(found.group(2))) if found \
        else (None, None)


def scatter_axes(sp):
    return [sp.attr_x.name, sp.attr_y.name]


def main():
    os.makedirs(oh.RESULTS, exist_ok=True)
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=1.5)
    W = lambda title: oh.widget(scheme, title)  # noqa: E731
    problems = oh.widget_problems(scheme)
    out = outputs(scheme)

    raw = out[("File", "Data")]
    selected = out[("Select Columns", "Data")]
    built = out[("Formula", "Data")]
    by_crim = out[("Discretize (CRIM)", "Data")]
    by_rm = out[("Discretize (RM)", "Data")]

    corr, corr2 = W("Correlations"), W("Correlations (log2)")
    sp, sp2 = W("Scatter Plot"), W("Scatter Plot (log2)")
    as_saved = {"correlations_selection": [v.name for v in corr.selection],
                "correlations_log2_selection": [v.name for v in corr2.selection],
                "scatter_axes": scatter_axes(sp),
                "scatter_log2_axes": scatter_axes(sp2)}

    # ---------------- step 1 ----------------
    medv = raw.get_column(TARGET)
    explore = {
        "rows": len(raw), "columns": len(raw.domain.variables),
        "target_in_file": raw.domain.class_var.name,
        "missing_values": int(np.isnan(raw.X).sum() + np.isnan(raw.Y).sum()),
        "medv": {"min": float(medv.min()), "max": float(medv.max()),
                 "mean": float(medv.mean()), "median": float(np.median(medv)),
                 "at_50": int((medv == 50).sum())},
    }

    # ---------------- steps 3-4 ----------------
    pick_pair(corr, TARGET, TOP_POSITIVE)
    oh.settle(scheme, quiet=1.0)
    listed = rank_rows(corr)
    pearson = pearson_with(selected, TARGET)
    ordered = sorted(pearson.items(), key=lambda kv: -kv[1]["r"])
    spearman = {n: float(sps.spearmanr(selected.get_column(n),
                                       selected.get_column(TARGET))[0])
                for n in (TOP_POSITIVE, STRONGEST)}

    # ---------------- step 5 ----------------
    pick_pair(corr2, TARGET, "log2_" + STRONGEST)
    oh.settle(scheme, quiet=1.0)
    listed2 = rank_rows(corr2)
    pearson2 = pearson_with(built, TARGET)
    formula_error = max(
        float(np.abs(built.get_column(name) - np.log2(
            built.get_column(name.replace("log2_", "")))).max())
        for name, _ in FORMULAS)

    # ---------------- steps 6-7 ----------------
    box_crim = W("Box Plot (MEDV by CRIM)")
    set_box(box_crim, TARGET, "CRIM")
    crim_var = by_crim.domain["CRIM"]
    t_crim, p_crim = welch(by_crim, TARGET, "CRIM")
    step7 = {
        "groups": group_stats(by_crim, TARGET, "CRIM"),
        "labels": list(crim_var.values),
        "crim_median": float(np.median(raw.get_column("CRIM"))),
        "widget_text": box_crim.stat_test,
        "t_widget": shown(box_crim.stat_test)[0],
        "p_widget": shown(box_crim.stat_test)[1],
        # cross-check with scipy's Welch test (sample variances; Orange's
        # own figure differs in the third digit)
        "t": abs(t_crim), "p": p_crim,
        "discretized": [v.name for v in by_crim.domain.variables
                        if v.is_discrete],
    }

    # ---------------- step 8 ----------------
    box_rm = W("Box Plot (by RM)")
    rm_var = by_rm.domain["RM"]
    tests = []
    for var in by_rm.domain.variables:
        if var.name == "RM":
            continue
        set_box(box_rm, var.name, "RM")
        t, p = welch(by_rm, var.name, "RM")
        g = group_stats(by_rm, var.name, "RM")
        tests.append({"attribute": var.name, "widget_text": box_rm.stat_test,
                      "t_widget": shown(box_rm.stat_test)[0],
                      "p_widget": shown(box_rm.stat_test)[1],
                      "t": abs(t), "p": p,
                      "means": [g[label]["mean"] for label in rm_var.values]})
    tests.sort(key=lambda d: d["p"])
    m = len(tests)
    step8 = {
        "labels": list(rm_var.values),
        "counts": [int((by_rm.get_column(rm_var) == i).sum())
                   for i in range(len(rm_var.values))],
        "tests": tests, "n_tests": m,
        "significant": [d["attribute"] for d in tests if d["p"] < ALPHA],
        "not_significant": [d["attribute"] for d in tests if d["p"] >= ALPHA],
        "bonferroni_alpha": ALPHA / m,
        "significant_bonferroni": [d["attribute"] for d in tests
                                   if d["p"] < ALPHA / m],
        "expected_false_positives": ALPHA * m,
        "chance_of_any_false_positive": 1 - (1 - ALPHA) ** m,
        "discretized": [v.name for v in by_rm.domain.variables
                        if v.is_discrete],
    }

    checks = {
        "widget_problems": problems,
        "housing_has_506_rows_14_columns": explore["rows"] == 506
        and explore["columns"] == 14,
        "medv_is_a_feature_after_select_columns":
            selected.domain.class_var is None
            and TARGET in [a.name for a in selected.domain.attributes]
            and len(selected.domain.attributes) == 14,
        "top_positive_is_rm": ordered[0][0] == TOP_POSITIVE,
        "strongest_is_lstat": max(
            pearson, key=lambda n: abs(pearson[n]["r"])) == STRONGEST,
        "saved_selection_is_medv_rm": sorted(
            as_saved["correlations_selection"]) == sorted([TARGET, TOP_POSITIVE]),
        "saved_scatter_shows_medv_rm": sorted(
            as_saved["scatter_axes"]) == sorted([TARGET, TOP_POSITIVE]),
        "saved_log2_selection_is_medv_log2_lstat": sorted(
            as_saved["correlations_log2_selection"])
            == sorted([TARGET, "log2_" + STRONGEST]),
        "saved_log2_scatter_shows_that_pair": sorted(
            as_saved["scatter_log2_axes"])
            == sorted([TARGET, "log2_" + STRONGEST]),
        "log2_formula_correct": formula_error < 1e-9,
        "log2_lstat_is_top_pair_with_medv": max(
            pearson2, key=lambda n: abs(pearson2[n]["r"]))
            == "log2_" + STRONGEST,
        "only_crim_discretized_into_2": step7["discretized"] == ["CRIM"]
        and len(step7["labels"]) == 2,
        "only_rm_discretized_into_2": step8["discretized"] == ["RM"]
        and len(step8["labels"]) == 2,
        "t_agrees_with_scipy": abs(step7["t_widget"] / step7["t"] - 1) < 0.01,
        "all_step8_t_agree_with_scipy": all(
            abs(d["t_widget"] / d["t"] - 1) < 0.01 for d in tests),
        "step8_significance_same_in_widget": all(
            (d["p_widget"] < ALPHA) == (d["p"] < ALPHA) for d in tests),
    }

    results = {
        "orange_version": __import__("Orange").__version__,
        "explore": explore, "as_saved": as_saved,
        "pearson_with_medv": pearson, "listed_in_widget": listed,
        "spearman_with_medv": spearman,
        "pearson_with_medv_after_formula": pearson2,
        "listed_in_widget_after_formula": listed2,
        "step7": step7, "step8": step8, "checks": checks,
    }
    with open(os.path.join(oh.RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- pictures ----------------
    keep = render_canvas(scheme, os.path.join(oh.RESULTS, "canvas.png"))
    grab(W("File"), "file", (760, 640))
    grab(W("Data Table"), "data_table", (1250, 480))
    grab(W("Box Plot"), "explore_box", (1050, 440))
    grab(W("Distributions"), "explore_dist", (1050, 560))
    grab(W("Select Columns"), "select_columns", (900, 620))

    grab(corr, "correlations", (560, 620))
    grab(sp, "scatter_rm_medv", (1100, 640))
    pick_pair(corr, TARGET, STRONGEST)
    oh.settle(scheme, quiet=1.0)
    sp.graph.update_regression_line()
    grab(sp, "scatter_lstat_medv", (1100, 640))
    pick_pair(corr, TARGET, TOP_POSITIVE)
    oh.settle(scheme, quiet=1.0)

    grab(W("Formula"), "formula", (760, 420))
    sp2.graph.update_regression_line()
    grab(corr2, "correlations_log2", (560, 660))
    grab(sp2, "scatter_log2", (1100, 640))

    grab(W("Discretize (CRIM)"), "discretize_crim", (900, 620))
    dist = W("Distributions (MEDV by CRIM)")
    set_distribution(dist, TARGET, "CRIM")
    grab(dist, "dist_medv_by_crim", (1050, 560))
    set_box(box_crim, TARGET, "CRIM")
    grab(box_crim, "box_medv_by_crim", (1050, 460))

    grab(W("Discretize (RM)"), "discretize_rm", (900, 620))
    pictured = [tests[0]["attribute"], TARGET] + step8["not_significant"]
    for name in dict.fromkeys(pictured):
        set_box(box_rm, name, "RM")
        grab(box_rm, "box_rm_" + name, (1050, 460))
    set_box(box_rm, TARGET, "RM")
    del keep

    slim = {k: v for k, v in results.items() if k != "listed_in_widget"
            and k != "listed_in_widget_after_formula"}
    print(json.dumps(slim, indent=1, default=str))
    print("LISTED", listed)
    print("LISTED2", listed2)
    failed = [k for k, v in checks.items() if k != "widget_problems" and not v]
    if problems or failed:
        print("CHECKS FAILED", problems, failed)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
