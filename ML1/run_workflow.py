"""Open submission/mp1_yahya.ows exactly as saved, run it, and record what
every widget produced.

Writes to ML1/results/:
    results.json        all numbers used in the report
    *.csv               the output tables (correlations, coefficients, ...)
    canvas.png          picture of the workflow canvas
    widget_*.png        pictures of the key Orange widgets with their output

It also recomputes the correlations and the regression directly with
numpy / scikit-learn and fails if they disagree with the Orange widgets.

    /Applications/Orange.app/Contents/MacOS/python ML1/run_workflow.py
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


def outputs(scheme):
    """{(source widget title, output channel name): value}"""
    sm = scheme.signal_manager
    out = {}
    for link in scheme.links:
        contents = sm.link_contents(link)
        if contents:
            out[(link.source_node.title, link.source_channel.name)] = \
                next(iter(contents.values()))
    return out


def table_rows(table):
    """Orange table -> list of dicts (attributes, class and metas)."""
    domain = table.domain
    cols = list(domain.variables) + list(domain.metas)
    rows = []
    for inst in table:
        row = {}
        for var in cols:
            val = inst[var]
            if var.is_continuous:
                row[var.name] = None if np.isnan(val) else float(val)
            else:
                row[var.name] = str(val)
        rows.append(row)
    return rows


def missing_counts(table):
    """Missing-value count per column of an Orange table."""
    counts = {}
    for var in table.domain.variables:
        counts[var.name] = int(np.isnan(table.get_column(var)).sum())
    for var in table.domain.metas:
        col = table.get_column(var)
        if var.is_string:
            counts[var.name] = int(sum(1 for v in col if v in ("", None)))
        else:
            counts[var.name] = int(np.isnan(col.astype(float)).sum())
    return counts


def render_canvas(scheme, path, scale=2.0):
    from orangecanvas.document.schemeedit import SchemeEditWidget
    editor = SchemeEditWidget()
    editor.setRegistry(oh.registry())
    editor.setScheme(scheme)
    editor.resize(1800, 1000)
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
    return editor  # keep alive until the caller is done


def grab_widget(w, path, size):
    w.resize(*size)
    w.show()
    for _ in range(20):
        oh.app().processEvents()
    w.grab().save(path)
    w.hide()


def main():
    os.makedirs(oh.RESULTS, exist_ok=True)
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme)

    problems = oh.widget_problems(scheme)
    print("INFO MESSAGES", oh.widget_problems(scheme, ("Information",)))
    out = outputs(scheme)

    raw = out[("Load ChicagoCrimes2019.csv", "Data")]
    kept = out[("Keep Model Columns", "Data")]
    model_train = out[("Drop Baseline Dummy Columns", "Data")]
    clean = out[("Drop Rows with Missing Values", "Data")]
    dummy = out[("Create Dummy Variables", "Data")]
    train = out[("Split 80% Train / 20% Test", "Data Sample")]
    test = out[("Split 80% Train / 20% Test", "Remaining Data")]
    corr = out[("Pearson Correlations vs Arrest", "Correlations")]
    top_corr = out[("Pick Top 10 Correlations", "Data")]
    coef = out[("Linear Regression (Arrest)", "Coefficients")]
    top_coef = out[("Pick Top 5 + / Top 5 - Coefficients", "Data")]

    # ---- Test and Score: read the numbers straight from the widget ----
    ts = oh.widget(scheme, "Test and Score (Test Sample)")
    model = ts.score_table.model
    header = [model.horizontalHeaderItem(c).text()
              for c in range(model.columnCount())]
    scores = {}
    for c in range(1, model.columnCount()):
        value = model.item(0, c).data(Qt.DisplayRole) if model.item(0, c) \
            else None
        if isinstance(value, (int, float)):
            scores[header[c]] = float(value)

    # ---- exploration facts from the raw table ----
    arrest = raw.get_column("Arrest")
    ptype_var = raw.domain["Primary Type"]
    ptype = raw.get_column(ptype_var)
    by_type = []
    for i, name in enumerate(ptype_var.values):
        mask = ptype == i
        n = int(mask.sum())
        by_type.append({"type": name, "count": n,
                        "arrest_rate": float(arrest[mask].mean()) if n else None})
    by_type.sort(key=lambda r: -r["count"])
    case = [str(v) for v in raw.get_column("Case Number")]
    area_var = raw.domain["Community Area"]
    area = raw.get_column(area_var)
    domestic = raw.get_column("Domestic")
    by_area = {}
    for i, name in enumerate(area_var.values):
        mask = area == i
        n = int(mask.sum())
        by_area[name] = {"count": n, "arrest_rate":
                         float(arrest[mask].mean()) if n else None}
    explore = {
        "by_area": by_area,
        "rows": len(raw),
        "columns": len(raw.domain.variables) + len(raw.domain.metas),
        "column_types": {
            v.name: type(v).__name__.replace("Variable", "")
            for v in list(raw.domain.variables) + list(raw.domain.metas)},
        "missing": missing_counts(raw),
        "arrest_true": int(arrest.sum()),
        "arrest_rate": float(arrest.mean()),
        "domestic_true": int(domestic.sum()),
        "arrest_rate_domestic": float(arrest[domestic == 1].mean()),
        "arrest_rate_not_domestic": float(arrest[domestic == 0].mean()),
        "duplicate_case_numbers": len(case) - len(set(case)),
        "community_area_zero": int(
            (area == area_var.values.index("0")).sum())
        if "0" in area_var.values else 0,
        "community_area_distinct": len(area_var.values),
        "year_values": sorted(set(raw.get_column("Year").tolist())),
        "primary_types": len(ptype_var.values),
        "by_type": by_type,
        "rare_types": [r for r in by_type if r["count"] < 50],
    }

    # ---- independent check with numpy / scikit-learn ----
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                                 r2_score)
    names = [a.name for a in train.domain.attributes]
    target = train.domain.class_var.name
    check_r = np.array([np.corrcoef(train.X[:, j], train.Y)[0, 1]
                        if train.X[:, j].std() > 0 else np.nan
                        for j in range(train.X.shape[1])])
    widget_r = {}
    for row in table_rows(corr):
        other = row["Feature 2"] if row["Feature 1"] == target \
            else row["Feature 1"]
        widget_r[other] = row["Correlation"]
    corr_diff = max(abs(widget_r[n] - check_r[j])
                    for j, n in enumerate(names) if n in widget_r)
    expected_top = sorted(widget_r, key=lambda n: -abs(widget_r[n]))[:10]
    got_top = [r["Feature 2"] if r["Feature 1"] == target else r["Feature 1"]
               for r in table_rows(top_corr)]

    # the model sees the training sample minus the two baseline dummies
    model_test = test.transform(model_train.domain)
    lr = LinearRegression().fit(model_train.X, model_train.Y)
    pred = lr.predict(model_test.X)
    kept_names = {a.name for a in model_train.domain.attributes}
    baselines = [n for n in names if n not in kept_names]
    check_scores = {
        "MSE": float(mean_squared_error(test.Y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(test.Y, pred))),
        "MAE": float(mean_absolute_error(test.Y, pred)),
        "R2": float(r2_score(test.Y, pred)),
    }
    score_diff = max(abs(scores[k] - check_scores[k]) for k in check_scores)
    # how the regression does if its output is read as arrest / no arrest
    as_class = pred >= 0.5
    actual = test.Y == 1
    classification = {
        "threshold": 0.5,
        "accuracy": float((as_class == actual).mean()),
        "majority_baseline": float(max(actual.mean(), 1 - actual.mean())),
        "precision": float((as_class & actual).sum() / max(as_class.sum(), 1)),
        "recall": float((as_class & actual).sum() / max(actual.sum(), 1)),
        "predicted_arrests": int(as_class.sum()),
        "actual_arrests": int(actual.sum()),
        "prediction_min": float(pred.min()),
        "prediction_max": float(pred.max()),
    }
    baseline_pred = np.full_like(test.Y, train.Y.mean())
    baseline = {
        "RMSE": float(np.sqrt(mean_squared_error(test.Y, baseline_pred))),
        "MAE": float(mean_absolute_error(test.Y, baseline_pred)),
    }

    coef_rows = table_rows(coef)
    widget_coef = {r["name"]: r["coef"] for r in coef_rows}
    coef_diff = max(
        [abs(widget_coef[a.name] - c)
         for a, c in zip(model_train.domain.attributes, lr.coef_)]
        + [abs(widget_coef["intercept"] - lr.intercept_)])
    feature_coefs = sorted((r for r in coef_rows if r["name"] != "intercept"),
                           key=lambda r: r["coef"])
    expected_coef = [r["name"] for r in feature_coefs[::-1][:5]] + \
                    [r["name"] for r in feature_coefs[:5]]
    got_coef = [r["name"] for r in table_rows(top_coef)]

    checks = {
        "widget_problems": problems,
        "max_correlation_difference_vs_numpy": float(corr_diff),
        "top10_matches_ranking": got_top == expected_top,
        "max_score_difference_vs_sklearn": float(score_diff),
        "max_coefficient_difference_vs_sklearn": float(coef_diff),
        "largest_abs_coefficient": float(max(
            abs(r["coef"]) for r in coef_rows)),
        "top_coefficients_match_ranking": got_coef == expected_coef,
        "train_plus_test_equals_clean": len(train) + len(test) == len(clean),
    }

    results = {
        "orange_version": __import__("Orange").__version__,
        "explore": explore,
        "pipeline": {
            "raw_rows": len(raw),
            "kept_columns": [v.name for v in kept.domain.variables],
            "community_area_levels": len(
                clean.domain["Community Area"].values),
            "rows_dropped_missing": len(kept) - len(clean),
            "baseline_levels": baselines,
            "model_columns": len(model_train.domain.attributes),
            "clean_rows": len(clean),
            "dummy_columns": len(dummy.domain.attributes),
            "dummy_column_names": [a.name for a in dummy.domain.attributes],
            "target": dummy.domain.class_var.name,
            "train_rows": len(train),
            "test_rows": len(test),
            "train_arrest_rate": float(train.Y.mean()),
            "test_arrest_rate": float(test.Y.mean()),
        },
        "correlations_total": len(corr),
        "top_correlations": table_rows(top_corr),
        "scores": scores,
        "baseline_scores": baseline,
        "classification_view": classification,
        "intercept": next(r["coef"] for r in coef_rows
                          if r["name"] == "intercept"),
        "coefficients_total": len(coef_rows),
        "top_coefficients": table_rows(top_coef),
        "checks": checks,
    }
    with open(os.path.join(oh.RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    for name, table in (("all_correlations", corr),
                        ("top_correlations", top_corr),
                        ("model_coefficients", coef),
                        ("top_coefficients", top_coef)):
        table.save(os.path.join(oh.RESULTS, name + ".csv"))

    # ---- pictures ----
    keep = render_canvas(scheme, os.path.join(oh.RESULTS, "canvas.png"))
    for title, fname, size in (
            ("Feature Statistics (Missing Values)", "widget_feature_statistics",
             (1100, 640)),
            ("Arrests by Crime Type", "widget_distribution", (1200, 620)),
            ("Top Correlations", "widget_top_correlations", (900, 360)),
            ("Test and Score (Test Sample)", "widget_test_and_score",
             (900, 420)),
            ("Top Coefficients", "widget_top_coefficients", (760, 360))):
        try:
            grab_widget(oh.widget(scheme, title),
                        os.path.join(oh.RESULTS, fname + ".png"), size)
        except Exception as err:  # pictures are a nice-to-have
            print("could not capture", title, err)
    del keep

    print(json.dumps({k: results[k] for k in (
        "pipeline", "scores", "baseline_scores", "classification_view",
        "intercept", "checks")}, indent=1, default=str)[:6000])
    print("TOP CORR", json.dumps(results["top_correlations"], indent=1))
    print("TOP COEF", json.dumps(results["top_coefficients"], indent=1))
    ex = dict(explore)
    ex.pop("by_type")
    ex.pop("by_area")
    print("EXPLORE", json.dumps(ex, indent=1))

    bad = [k for k, v in checks.items()
           if (k.endswith("ranking") or k.endswith("clean")) and not v]
    if problems or bad or corr_diff > 1e-6 or score_diff > 1e-6 \
            or coef_diff > 1e-6:
        print("CHECKS FAILED", problems, bad, corr_diff, score_diff, coef_diff)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
