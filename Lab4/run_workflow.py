"""Open Lab4/submission/lab4_yahya.ows as saved, run it, check it, and
capture what each widget shows.

Writes to Lab4/results/:
    results.json   the numbers quoted in the report
    canvas.png     picture of the workflow canvas
    *.png          pictures of the Orange widgets used in each step

    /Applications/Orange.app/Contents/MacOS/python Lab4/run_workflow.py
"""
import json
import logging
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from AnyQt.QtCore import QRectF, Qt  # noqa: E402
from AnyQt.QtGui import QImage, QPainter  # noqa: E402

from build_workflow import (  # noqa: E402
    CROSS_VALIDATION, LINEAR_MODELS, OTHER_MODELS, TARGET, TEST_ON_TEST,
    TUNED, set_resampling,
)

warnings.filterwarnings("ignore")
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

COEF_TABLES = {
    "No Regularization": "Coefficients (No Reg.)",
    "Lasso Regression (L1)": "Coefficients (Lasso)",
    "Elastic Net Regression": "Coefficients (Elastic Net)",
    "Elastic Net Regression (1)": "Coefficients (Top feature)",
}


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


def sort_scores(test, column="RMSE"):
    """Click the RMSE header: smallest to largest."""
    proxy = test.score_table.sorted_model
    for c in range(proxy.columnCount()):
        if str(proxy.headerData(c, Qt.Horizontal)) == column:
            test.score_table.view.sortByColumn(c, Qt.AscendingOrder)
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


def coefficients(table):
    names = [str(v) for v in table.metas[:, 0]]
    return [{"name": n, "coef": float(c)}
            for n, c in zip(names, table.X[:, 0])]


def summarise(coefs):
    feats = [c for c in coefs if c["name"] != "intercept"]
    ranked = sorted(feats, key=lambda c: -c["coef"])
    return {
        "intercept": next(c["coef"] for c in coefs
                          if c["name"] == "intercept"),
        "n_features": len(feats),
        "n_zero": sum(1 for c in feats if c["coef"] == 0),
        "n_nonzero": sum(1 for c in feats if c["coef"] != 0),
        "top_positive": ranked[:5], "top_negative": ranked[::-1][:5],
        "largest_abs": max(abs(c["coef"]) for c in feats),
        "all": ranked,
    }


def main():
    import Orange
    from Orange.evaluation import MAE, R2, RMSE, CrossValidation

    os.makedirs(oh.RESULTS, exist_ok=True)
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=3.0, timeout=600)
    W = lambda title: oh.widget(scheme, title)  # noqa: E731
    problems = oh.widget_problems(scheme)
    out = outputs(scheme)

    raw = out[("Datasets", "Data")]
    targeted = out[("Select Columns", "Data")]
    normal = out[("Preprocess", "Preprocessed Data")]
    train = out[("Training set", "Selected Data")]
    test_set = out[("Testing set", "Selected Data")]
    sample = out[("Data Sampler", "Data Sample")]
    remaining = out[("Data Sampler", "Remaining Data")]
    reduced = out[("Rank", "Reduced Data")]
    test, test1 = W("Test and Score"), W("Test and Score (1)")
    as_saved = {"test_and_score_mode": test.resampling,
                "test_and_score_1_mode": test1.resampling,
                "folds": [test.NFolds[test.n_folds],
                          test1.NFolds[test1.n_folds]]}

    # ---------------- step 1: the data ----------------
    info = W("Datasets").selected_dataset()
    hdi = raw.get_column(TARGET)
    countries = [str(c) for c in raw.get_column("Country")]
    order = np.argsort(-hdi)
    missing_by_col = np.isnan(raw.X).sum(axis=0)
    explore = {
        "dataset_title": info.title, "dataset_file": info.file_path[-1],
        "rows": len(raw), "features": len(raw.domain.attributes),
        "metas": [m.name for m in raw.domain.metas],
        "target_as_loaded": raw.domain.class_var.name
        if raw.domain.class_var else None,
        "all_numeric": all(a.is_continuous for a in raw.domain.attributes),
        "missing_cells": int(missing_by_col.sum()),
        "missing_percent": float(100 * missing_by_col.sum() / raw.X.size),
        "columns_with_missing": int((missing_by_col > 0).sum()),
        "most_missing": [(raw.domain.attributes[i].name,
                          int(missing_by_col[i]))
                         for i in np.argsort(-missing_by_col)[:5]],
        "hdi": {"min": float(hdi.min()), "max": float(hdi.max()),
                "mean": float(hdi.mean()), "median": float(np.median(hdi))},
        "top5": [(countries[i], float(hdi[i])) for i in order[:5]],
        "bottom5": [(countries[i], float(hdi[i])) for i in order[-5:][::-1]],
        "feature_names": [a.name for a in raw.domain.attributes],
    }

    # ---------------- step 2 ----------------
    train_names = {str(c) for c in train.get_column("Country")}
    test_names = {str(c) for c in test_set.get_column("Country")}
    step2 = {
        "target_after_select_columns": targeted.domain.class_var.name,
        "feature_min": float(np.nanmin(normal.X)),
        "feature_max": float(np.nanmax(normal.X)),
        "every_feature_spans_0_to_1": bool(
            np.allclose(np.nanmin(normal.X, axis=0), 0)
            and np.allclose(np.nanmax(normal.X, axis=0), 1)),
        "target_unchanged": bool(np.allclose(normal.Y, targeted.Y)),
        "train_rows": len(train), "test_rows": len(test_set),
        "overlap": len(train_names & test_names),
        "train_hdi_mean": float(train.Y.mean()),
        "test_hdi_mean": float(test_set.Y.mean()),
    }

    # ---------------- step 3 / 5: cross-validation ----------------
    set_resampling(test, CROSS_VALIDATION)
    oh.settle(scheme, quiet=2.0, timeout=600)
    sort_scores(test)
    cv_rows = score_rows(test)
    grab(test, "test_and_score_cv", (1100, 470))

    # ---------------- step 4: coefficients ----------------
    out = outputs(scheme)
    coefs = {model: summarise(coefficients(out[(model, "Coefficients")]))
             for model in COEF_TABLES}

    # ---------------- step 5: rank ----------------
    rank = W("Rank")
    top_feature = reduced.domain.attributes[0].name
    rmodel = rank.ranksModel
    rank_rows = []
    for r in range(min(8, rmodel.rowCount())):
        rank_rows.append([rmodel.index(r, c).data(Qt.DisplayRole)
                          for c in range(rmodel.columnCount())])
    rank_head = [str(rmodel.headerData(c, Qt.Horizontal))
                 for c in range(rmodel.columnCount())]
    # Extra check, outside the saved workflow: the same Elastic Net and the
    # same 10-fold cross-validation, but given only the one top feature.
    en = W("Elastic Net Regression (1)").create_learner()
    cv = CrossValidation(k=10, random_state=42)
    res_one = cv(reduced, [en])
    res_all = cv(train, [en])
    # Why the ranking looks the way it does: compare each feature's RReliefF
    # score with how many of its values are missing in the training set, and
    # see what RReliefF picks once the gaps are filled in with the mean.
    from Orange.preprocess import Impute, score as oscore
    scores = oscore.RReliefF()(train)
    n_missing = np.isnan(train.X).sum(axis=0)
    by_score = np.argsort(-scores)
    filled = Impute()(train)
    scores_filled = oscore.RReliefF()(filled)
    by_filled = np.argsort(-scores_filled)
    attrs = train.domain.attributes
    missing_effect = {
        "widget_top_matches": attrs[by_score[0]].name == top_feature,
        "top8": [{"name": attrs[i].name, "score": float(scores[i]),
                  "missing_in_training": int(n_missing[i])}
                 for i in by_score[:8]],
        "corr_score_vs_missing": float(
            np.corrcoef(scores, n_missing)[0, 1]),
        "top5_after_filling_gaps": [
            {"name": attrs[i].name, "score": float(scores_filled[i])}
            for i in by_filled[:5]],
        "training_rows": len(train),
    }
    step5 = {
        "missing_effect": missing_effect,
        "top_feature": top_feature, "reduced_features": len(
            reduced.domain.attributes), "rank_header": rank_head,
        "rank_top": [[str(x) if not isinstance(x, float) else round(x, 4)
                      for x in row] for row in rank_rows],
        "one_feature_cv": {"RMSE": float(RMSE(res_one)[0]),
                           "MAE": float(MAE(res_one)[0]),
                           "R2": float(R2(res_one)[0])},
        "all_features_cv_check": {"RMSE": float(RMSE(res_all)[0]),
                                  "R2": float(R2(res_all)[0])},
        "top_feature_missing_in_training": int(np.isnan(
            train.get_column(top_feature)).sum()),
    }

    # ---------------- step 6: test on test data ----------------
    set_resampling(test, TEST_ON_TEST)
    oh.settle(scheme, quiet=2.0, timeout=600)
    sort_scores(test)
    test_rows = score_rows(test)
    grab(test, "test_and_score_test", (1100, 470))

    # ---------------- step 7: other models ----------------
    sort_scores(test1)
    other_cv = score_rows(test1)
    grab(test1, "test_and_score_1_cv", (1100, 470))
    set_resampling(test1, TEST_ON_TEST)
    oh.settle(scheme, quiet=2.0, timeout=600)
    sort_scores(test1)
    other_test = score_rows(test1)
    set_resampling(test1, CROSS_VALIDATION)
    oh.settle(scheme, quiet=2.0, timeout=600)
    sort_scores(test1)
    tuning = None
    tuning_path = os.path.join(oh.RESULTS, "tuning.json")
    if os.path.exists(tuning_path):
        with open(tuning_path) as f:
            tuning = json.load(f)

    by_model = {r["Model"]: r for r in cv_rows}
    en_same = abs(by_model["Elastic Net Regression"]["RMSE"]
                  - by_model["Elastic Net Regression (1)"]["RMSE"]) < 1e-12
    checks = {
        "widget_problems": problems,
        "dataset_is_HDI": explore["dataset_title"] == "HDI"
        and explore["dataset_file"] == "HDI.tab" and explore["rows"] == 188,
        "hdi_is_target": step2["target_after_select_columns"] == TARGET,
        "features_normalized_0_1": step2["every_feature_spans_0_to_1"],
        "target_not_normalized": step2["target_unchanged"],
        "split_is_50_50": step2["train_rows"] == 94
        and step2["test_rows"] == 94 and step2["overlap"] == 0,
        "tables_pass_on_every_row": len(train) == len(sample)
        and len(test_set) == len(remaining),
        "cv_has_four_linear_models": sorted(by_model) == sorted(COEF_TABLES),
        "cv_sorted_by_rmse": [r["RMSE"] for r in cv_rows]
        == sorted(r["RMSE"] for r in cv_rows),
        "ten_folds": as_saved["folds"] == [10, 10],
        "rank_selects_one_feature": step5["reduced_features"] == 1,
        "one_feature_model_has_one_coefficient":
            coefs["Elastic Net Regression (1)"]["n_features"] == 1,
        "saved_as_test_on_test_data":
            as_saved["test_and_score_mode"] == TEST_ON_TEST,
        "other_models_scored_by_cv":
            as_saved["test_and_score_1_mode"] == CROSS_VALIDATION,
        "four_other_models": sorted(r["Model"] for r in other_cv)
        == sorted(OTHER_MODELS),
        "my_cv_matches_widget": abs(
            step5["all_features_cv_check"]["RMSE"]
            - by_model["Elastic Net Regression"]["RMSE"]) < 1e-9,
    }

    results = {
        "orange_version": Orange.__version__,
        "explore": explore, "step2": step2, "as_saved": as_saved,
        "cv_scores": cv_rows, "coefficients": coefs, "step5": step5,
        "elastic_net_1_same_score_as_elastic_net": en_same,
        "test_scores": test_rows,
        "other_cv_scores": other_cv, "other_test_scores": other_test,
        "tuned_settings": TUNED, "tuning": tuning, "checks": checks,
    }
    with open(os.path.join(oh.RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- pictures ----------------
    keep = render_canvas(scheme, os.path.join(oh.RESULTS, "canvas.png"))
    grab(W("Datasets"), "datasets", (1100, 640))
    grab(W("Data Table"), "data_table", (1250, 480))
    grab(W("Distributions"), "explore_dist", (1050, 520))
    grab(W("Box Plot"), "explore_box", (1050, 420))
    grab(W("Select Columns"), "select_columns", (900, 620))
    grab(W("Preprocess"), "preprocess", (900, 460))
    grab(W("Data Sampler"), "data_sampler", (360, 560))
    grab(W("Training set"), "training_set", (1250, 420))
    grab(W("Testing set"), "testing_set", (1250, 420))
    for model in LINEAR_MODELS + ["Elastic Net Regression (1)"]:
        tag = model.lower().replace(" ", "_").replace("(", "").replace(")", "")
        grab(W(model), "model_" + tag, (460, 420))
        table = W(COEF_TABLES[model])
        view, tmodel = table.view, table.view.model()
        for c in range(tmodel.columnCount()):   # make the names readable
            head = str(tmodel.headerData(c, Qt.Horizontal, Qt.DisplayRole))
            view.setColumnWidth(c, 430 if head == "name" else 110)
        grab(table, "coef_" + tag, (860, 640 if "(1)" not in model else 260))
    grab(rank, "rank", (900, 520))
    grab(W("Reduced Data"), "reduced_data", (900, 420))
    for model in OTHER_MODELS:
        grab(W(model), "model_" + model.lower().replace(" ", "_"), (460, 460))
    del keep

    slim = {k: v for k, v in results.items()
            if k not in ("explore", "coefficients", "tuning")}
    print(json.dumps(slim, indent=1, default=str))
    ex = dict(explore)
    ex.pop("feature_names")
    print("EXPLORE", json.dumps(ex, indent=1))
    for model, c in coefs.items():
        print("COEF", model, "intercept %.4f zero %d nonzero %d"
              % (c["intercept"], c["n_zero"], c["n_nonzero"]))
        print("   +", [(d["name"][:38], round(d["coef"], 4))
                       for d in c["top_positive"]])
        print("   -", [(d["name"][:38], round(d["coef"], 4))
                       for d in c["top_negative"]])
    failed = [k for k, v in checks.items() if k != "widget_problems" and not v]
    if problems or failed:
        print("CHECKS FAILED", problems, failed)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
