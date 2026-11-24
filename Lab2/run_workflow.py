"""Open Lab2/submission/lab2_yahya.ows as saved, run it, check it, and
capture what each widget shows.

Writes to Lab2/results/:
    results.json   the numbers quoted in the report
    canvas.png     picture of the workflow canvas
    *.png          pictures of the Orange widgets used in each step

    /Applications/Orange.app/Contents/MacOS/python Lab2/run_workflow.py
"""
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from AnyQt.QtCore import QRectF, Qt  # noqa: E402
from AnyQt.QtGui import QImage, QPainter  # noqa: E402

logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

OUTLIER_CAR = "oldsmobile cutlass ciera (diesel)"
NUMERIC = ["displacement", "horsepower", "weight", "acceleration", "mpg"]
IMPUTE_METHODS = {"Average/Most frequent": 2,
                  "Model-based imputer (simple tree)": 4,
                  "Random values": 5}
OUTLIER_METHODS = ["One Class SVM", "Covariance Estimator",
                   "Local Outlier Factor", "Isolation Forest"]


def outputs(scheme):
    sm = scheme.signal_manager
    out = {}
    for link in scheme.links:
        contents = sm.link_contents(link)
        if contents:
            out[(link.source_node.title, link.source_channel.name)] = \
                next(iter(contents.values()))
    return out


def rows(table, columns=None):
    domain = table.domain
    cols = [domain[c] for c in columns] if columns else \
        list(domain.variables) + list(domain.metas)
    result = []
    for inst in table:
        row = {}
        for var in cols:
            val = inst[var]
            if var.is_continuous:
                row[var.name] = None if np.isnan(val) else round(float(val), 3)
            else:
                row[var.name] = str(val)
        result.append(row)
    return result


def names_of(table):
    return [str(v) for v in table.get_column("model_2")]


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


def set_scatter(widget, x, y, color=None):
    domain = widget.data.domain
    widget.attr_x, widget.attr_y = domain[x], domain[y]
    widget.attr_color = domain[color] if color else None
    widget.attr_changed()
    widget.graph.update_regression_line()


def main():
    os.makedirs(oh.RESULTS, exist_ok=True)
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=1.5)
    W = lambda title: oh.widget(scheme, title)  # noqa: E731
    problems = oh.widget_problems(scheme)
    out = outputs(scheme)

    raw = out[("File", "Data")]
    classed = out[("Create Class", "Data")]
    imputed = out[("Impute", "Data")]
    picked = out[("Data Table (1)", "Selected Data")]
    matching = out[("Select by Data Index", "Matching Data")]
    clicked = out[("Scatter Plot (1)", "Selected Data")]
    flagged = out[("Outliers", "Outliers")]
    inliers = out[("Outliers", "Inliers")]
    built = out[("Feature Constructor", "Data")]
    reloaded = out[("File (1)", "Data")]

    # ---------------- step 1: exploration facts ----------------
    car = names_of(raw)
    first_words = sorted({c.split()[0] for c in car})
    dup_names = len(car) - len(set(car))
    stats = {}
    for name in NUMERIC:
        col = raw.get_column(name)
        ok = col[~np.isnan(col)]
        stats[name] = {"min": float(ok.min()), "max": float(ok.max()),
                       "mean": float(ok.mean()),
                       "median": float(np.median(ok)),
                       "missing": int(np.isnan(col).sum())}
    cat_counts = {}
    for name in ("cylinders", "origin", "model_1"):
        var = raw.domain[name]
        col = raw.get_column(var)
        cat_counts[name] = {v: int((col == i).sum())
                            for i, v in enumerate(var.values)}
    origin_var = raw.domain["origin"]
    mpg = raw.get_column("mpg")
    origin_col = raw.get_column(origin_var)
    mpg_by_origin = {v: float(np.median(mpg[origin_col == i]))
                     for i, v in enumerate(origin_var.values)}
    cyl_var = raw.domain["cylinders"]
    cyl_col = raw.get_column(cyl_var)
    mpg_by_cyl = {v: float(mpg[cyl_col == i].mean())
                  for i, v in enumerate(cyl_var.values)}
    low_mpg = [r for r in rows(raw, ["model_2", "mpg", "displacement"])
               if r["mpg"] <= 10]
    explore = {
        "rows": len(raw),
        "columns": len(raw.domain.variables) + len(raw.domain.metas),
        "types": {v.name: type(v).__name__.replace("Variable", "")
                  for v in list(raw.domain.variables) + list(raw.domain.metas)},
        "stats": stats, "categories": cat_counts,
        "first_words": first_words, "duplicate_car_names": dup_names,
        "median_mpg_by_origin": mpg_by_origin, "mean_mpg_by_cylinders": mpg_by_cyl,
        "corr_weight_mpg": float(np.corrcoef(raw.get_column("weight"), mpg)[0, 1]),
        "lowest_mpg": low_mpg,
    }

    # ---------------- step 2 ----------------
    make = classed.domain.class_var
    make_counts = {v: int((classed.Y == i).sum())
                   for i, v in enumerate(make.values)}

    # ---------------- step 3: try the three imputers ----------------
    show = ["model_2", "cylinders", "displacement", "weight", "horsepower"]
    impute = W("Impute")
    saved_method = impute.default_method_index
    trials = {}
    for label, index in IMPUTE_METHODS.items():
        impute.set_default_method(index)
        oh.settle(scheme, quiet=1.0)
        trial = outputs(scheme)[("Select by Data Index", "Matching Data")]
        trials[label] = rows(trial, show)
        tag = {2: "average", 4: "model", 5: "random"}[index]
        grab(impute, "impute_" + tag, (700, 520))
        grab(W("Data Table (2)"), "table2_" + tag, (1100, 330))
    impute.set_default_method(saved_method)
    oh.settle(scheme, quiet=1.0)
    hp_all = raw.get_column("horsepower")
    cyl4 = cyl_col == cyl_var.values.index("4")
    reference = {
        "overall_mean_hp": float(np.nanmean(hp_all)),
        "mean_hp_4cyl": float(np.nanmean(hp_all[cyl4])),
        "mean_hp_6cyl": float(np.nanmean(
            hp_all[cyl_col == cyl_var.values.index("6")])),
    }

    # ---------------- step 4: outliers ----------------
    out = outputs(scheme)
    flagged_names = names_of(flagged)
    outliers_w = W("Outliers")
    saved_outlier = outliers_w.outlier_method
    by_method = {}
    for i, label in enumerate(OUTLIER_METHODS):
        outliers_w.outlier_method = i
        outliers_w.set_current_editor()
        outliers_w.commit.now()
        oh.settle(scheme, quiet=1.0)
        o = outputs(scheme)
        f = o.get(("Outliers", "Outliers"))
        by_method[label] = {
            "flagged": len(f) if f is not None else 0,
            "catches_the_diesel": f is not None and OUTLIER_CAR in names_of(f)}
    outliers_w.outlier_method = saved_outlier
    outliers_w.set_current_editor()
    outliers_w.commit.now()
    oh.settle(scheme, quiet=1.0)

    # ---------------- step 5 ----------------
    hp, mpg_b = built.get_column("horsepower"), built.get_column("mpg")
    l100 = built.get_column("Lper100k")
    expected = 100 * 3.785411784 / (1.609344 * mpg_b)
    step5 = {
        "rows": len(built),
        "r_hp_mpg": float(np.corrcoef(hp, mpg_b)[0, 1]),
        "r_hp_lper100k": float(np.corrcoef(hp, l100)[0, 1]),
        "formula_max_error": float(np.abs(l100 - expected).max()),
        "example": rows(built, ["model_2", "mpg", "Lper100k"])[:3],
        "lper100k_min": float(l100.min()), "lper100k_max": float(l100.max()),
    }

    # ---------------- step 6: by make ----------------
    mk = reloaded.domain.class_var
    by_make = {}
    for i, v in enumerate(mk.values):
        mask = reloaded.Y == i
        by_make[v] = {"count": int(mask.sum())}
        for name in NUMERIC + ["Lper100k"]:
            col = reloaded.get_column(name)[mask]
            by_make[v][name] = {"mean": float(np.nanmean(col)),
                                "sd": float(np.nanstd(col, ddof=1))}

    checks = {
        "widget_problems": problems,
        "make_counts_match_handout": make_counts == {
            "FORD": 51, "CHEVY": 47, "VW": 21, "OTHER": 279},
        "six_unknown_horsepower_selected": len(picked) == 6 and bool(
            np.isnan(picked.get_column("horsepower")).all()),
        "select_by_index_returns_those_six": sorted(names_of(matching))
        == sorted(names_of(picked)),
        "imputed_has_no_missing_hp": not bool(
            np.isnan(imputed.get_column("horsepower")).any()),
        "scatter_selection_is_the_diesel": names_of(clicked) == [OUTLIER_CAR],
        "diesel_flagged_as_outlier": OUTLIER_CAR in flagged_names,
        "diesel_not_in_inliers": OUTLIER_CAR not in names_of(inliers),
        "lper100k_formula_correct": step5["formula_max_error"] < 1e-9,
        "saved_csv_reloaded": len(reloaded) == len(built),
        "car_make_is_target_in_file1": mk is not None and mk.name == "car_make",
    }

    results = {
        "orange_version": __import__("Orange").__version__,
        "explore": explore,
        "make_counts": make_counts,
        "missing_hp_cars": rows(picked, show),
        "impute_trials": trials, "impute_reference": reference,
        "outlier_car": rows(clicked)[0] if len(clicked) else None,
        "outliers_flagged": len(flagged), "inliers": len(inliers),
        "flagged_cars": rows(flagged, ["model_2", "displacement", "mpg",
                                       "horsepower", "weight"]),
        "outlier_methods": by_method,
        "step5": step5, "by_make": by_make, "checks": checks,
    }
    with open(os.path.join(oh.RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- pictures ----------------
    keep = render_canvas(scheme, os.path.join(oh.RESULTS, "canvas.png"))
    grab(W("File"), "file", (760, 620))
    grab(W("Data Table"), "data_table", (1250, 520))
    grab(W("Feature Statistics"), "feature_statistics", (1250, 520))
    box = W("Box Plot")
    grab(box, "box_mpg_origin", (1050, 520))
    dist = W("Distributions")
    grab(dist, "dist_mpg", (1050, 560))
    grab(W("Scatter Plot"), "scatter_weight_mpg", (1100, 640))
    grab(W("Create Class"), "create_class", (760, 420))
    grab(W("Data Table (1)"), "table1_selected", (1250, 420))
    grab(W("Scatter Plot (1)"), "scatter1_outlier", (1100, 640))
    grab(W("Data Table (3)"), "table3_outlier", (1250, 260))
    grab(W("Outliers"), "outliers_widget", (520, 420))
    grab(W("Scatter Plot (2)"), "scatter2_outliers", (1100, 640))
    grab(W("Feature Constructor"), "feature_constructor", (760, 420))
    sp3 = W("Scatter Plot (3)")
    set_scatter(sp3, "horsepower", "mpg")
    grab(sp3, "scatter3_hp_mpg", (1100, 640))
    set_scatter(sp3, "displacement", "mpg")
    grab(sp3, "scatter3_inliers_disp_mpg", (1100, 640))
    set_scatter(sp3, "horsepower", "Lper100k")
    grab(sp3, "scatter3_hp_lper100k", (1100, 640))
    grab(W("Save Data (1)"), "save_data", (520, 260))
    grab(W("File (1)"), "file1", (760, 660))
    step6 = W("Step 6")
    d6 = step6.data.domain
    for name in NUMERIC + ["Lper100k"]:
        step6.var = d6[name]
        step6._on_var_changed()
        oh.settle(scheme, quiet=0.5)
        grab(step6, "step6_" + name, (1050, 560))
    del keep

    slim = dict(results)
    for key in ("explore", "flagged_cars"):
        slim.pop(key)
    print(json.dumps(slim, indent=1, default=str))
    ex = dict(explore)
    print("EXPLORE", json.dumps(ex, indent=1, default=str))
    failed = [k for k, v in checks.items() if k != "widget_problems" and not v]
    if problems or failed:
        print("CHECKS FAILED", problems, failed)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
