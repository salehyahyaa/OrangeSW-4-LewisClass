"""Open Lab5/submission/lab5_yahya.ows as saved, run it, check it, and
capture what each widget shows.

Writes to Lab5/results/:
    results.json   the numbers quoted in the report
    canvas.png     picture of the workflow canvas
    *.png          pictures of the Orange widgets used in each step

    /Applications/Orange.app/Contents/MacOS/python Lab5/run_workflow.py
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
from orangewidget.utils.overlay import MessageOverlayWidget  # noqa: E402

from build_workflow import (  # noqa: E402
    BEST_EMBEDDER, CROSS_VALIDATION, DATASET, MODELS, SQUEEZENET,
    TEST_ON_TEST, TUNED, set_resampling,
)

warnings.filterwarnings("ignore")
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)


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
    for tip in w.findChildren(MessageOverlayWidget):   # "Ok, got it" tips
        tip.hide()
    for _ in range(10):
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
    """Click the CA (accuracy) header: largest first."""
    proxy = test.score_table.sorted_model
    for c in range(proxy.columnCount()):
        if str(proxy.headerData(c, Qt.Horizontal)) == column:
            test.score_table.view.sortByColumn(c, Qt.DescendingOrder)
            for i in range(proxy.columnCount()):   # room for the numbers
                test.score_table.view.setColumnWidth(i, 150 if i == 0 else 78)
            test.score_table.view.resizeRowsToContents()
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


def matrices(results, tag, matrix_widget):
    """Confusion matrix of every model, the pictures the widget shows and
    the names of the pictures each model got wrong."""
    classes = [str(v) for v in results.domain.class_var.values]
    names = [str(n) for n in results.data.get_column("image name")]
    out = {}
    for i, learner in enumerate(matrix_widget.learners):
        actual = results.actual.astype(int)
        predicted = results.predicted[i].astype(int)
        counts = np.zeros((len(classes), len(classes)), dtype=int)
        for a, p in zip(actual, predicted):
            counts[a, p] += 1
        wrong = [{"picture": names[row], "actual": classes[a],
                  "predicted": classes[p]}
                 for row, a, p in zip(results.row_indices, actual, predicted)
                 if a != p]
        out[str(learner)] = {"classes": classes, "counts": counts.tolist(),
                             "wrong": wrong}
        matrix_widget.selected_learner[:] = [i]
        matrix_widget._learner_changed()
        grab(matrix_widget, "confusion_%s_%s" % (
            tag, str(learner).lower().replace(" ", "_")), (640, 330))
    matrix_widget.selected_learner[:] = [0]
    matrix_widget._learner_changed()
    return out


def describe(table):
    """What the Image Embedding widget produced."""
    x = table.X
    return {
        "rows": len(table), "features": len(table.domain.attributes),
        "first_features": [a.name for a in table.domain.attributes[:3]],
        "last_feature": table.domain.attributes[-1].name,
        "all_numeric": all(a.is_continuous for a in table.domain.attributes),
        "metas": [m.name for m in table.domain.metas],
        "target": table.domain.class_var.name,
        "classes": [str(v) for v in table.domain.class_var.values],
        "class_counts": [int(c) for c in np.bincount(table.Y.astype(int))],
        "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()),
        "share_zero": float((x == 0).mean()),
        "first_row": {"picture": str(table.get_column("image name")[0]),
                      "values": [round(float(v), 4) for v in x[0, :6]]},
    }


def contact_sheet(path, per_class=6, cell=200):
    """A small sheet of example training pictures for the report."""
    from PIL import Image
    train = os.path.join(DATASET, "train")
    classes = sorted(d for d in os.listdir(train)
                     if os.path.isdir(os.path.join(train, d)))
    sheet = Image.new("RGB", (per_class * cell, len(classes) * cell),
                      "white")
    for r, cls in enumerate(classes):
        files = sorted(os.listdir(os.path.join(train, cls)))[:per_class]
        for c, name in enumerate(files):
            im = Image.open(os.path.join(train, cls, name)).convert("RGB")
            side = min(im.size)
            left, top = (im.width - side) // 2, (im.height - side) // 2
            im = im.crop((left, top, left + side, top + side))
            im = im.resize((cell - 8, cell - 8))
            sheet.paste(im, (c * cell + 4, r * cell + 4))
    sheet.save(path, quality=88)


def main():
    import Orange
    from Orange.evaluation import CA, CrossValidation

    os.makedirs(oh.RESULTS, exist_ok=True)
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=3.0, timeout=900)
    W = lambda title: oh.widget(scheme, title)  # noqa: E731
    problems = oh.widget_problems(scheme)
    out = outputs(scheme)

    imported = out[("Import Images", "Data")]
    imported_test = out[("Import Images (test)", "Data")]
    squeeze = out[("Training Data", "Selected Data")]
    best_train = out[("Training Data (1)", "Selected Data")]
    best_test = out[("Image Embedding (test)", "Embeddings")]
    test, test1 = W("Test and Score"), W("Test and Score (1)")
    embed, embed_best = W("Image Embedding"), W("Image Embedding (best)")
    embed_test = W("Image Embedding (test)")
    as_saved = {
        "mode": test.resampling, "mode_1": test1.resampling,
        "folds": [test.NFolds[test.n_folds], test1.NFolds[test1.n_folds]],
        "embedder": embed.embedders[embed.cb_embedder_current_id],
        "embedder_best": embed_best.embedders[
            embed_best.cb_embedder_current_id],
        "embedder_test": embed_test.embedders[
            embed_test.cb_embedder_current_id],
        "import_paths": [W(t).recent_paths[0].relpath
                         for t in ("Import Images", "Import Images (test)")],
    }
    folders = {}
    for split in ("train", "test"):
        for cls in sorted(os.listdir(os.path.join(DATASET, split))):
            d = os.path.join(DATASET, split, cls)
            if os.path.isdir(d):
                folders["%s/%s" % (split, cls)] = len(
                    [f for f in os.listdir(d) if f.endswith(".jpg")])
    train_names = {str(n) for n in imported.get_column("image name")}
    test_names = {str(n) for n in imported_test.get_column("image name")}

    # ---- steps 4-5: SqueezeNet, default models, cross-validation ----
    sort_scores(test)
    cv_rows = score_rows(test)
    grab(test, "test_and_score_cv", (1100, 560))
    cm_cv = matrices(out[("Test and Score", "Evaluation Results")],
                     "squeezenet_cv", W("Confusion Matrix"))
    # the same cross-validation done by hand, to check the widget
    learners = [W(m).create_learner() for m in MODELS]
    by_hand = CA(CrossValidation(k=10, random_state=42)(squeeze, learners))

    # ---- steps 6-7: best embedder, cross-validation and test data ----
    sort_scores(test1)
    test_rows = score_rows(test1)
    grab(test1, "test_and_score_1_test", (1100, 560))
    cm_test = matrices(out[("Test and Score (1)", "Evaluation Results")],
                       "best_test", W("Confusion Matrix (1)"))
    set_resampling(test1, CROSS_VALIDATION)
    oh.settle(scheme, quiet=3.0, timeout=900)
    sort_scores(test1)
    best_cv_rows = score_rows(test1)
    grab(test1, "test_and_score_1_cv", (1100, 560))
    cm_best_cv = matrices(
        outputs(scheme)[("Test and Score (1)", "Evaluation Results")],
        "best_cv", W("Confusion Matrix (1)"))
    set_resampling(test1, TEST_ON_TEST)
    oh.settle(scheme, quiet=3.0, timeout=900)
    sort_scores(test1)

    experiments = None
    path = os.path.join(oh.RESULTS, "experiments.json")
    if os.path.exists(path):
        with open(path) as f:
            experiments = json.load(f)

    by_model = {r["Model"]: r for r in cv_rows}
    checks = {
        "widget_problems": problems,
        "folders_have_12_each": folders == {
            "train/cats": 12, "train/dogs": 12,
            "test/cats": 12, "test/dogs": 12},
        "paths_relative_to_workflow": as_saved["import_paths"] == [
            "classificationDataset/train", "classificationDataset/test"],
        "train_24_test_24": len(imported) == 24 and len(imported_test) == 24,
        "no_picture_in_both": not (train_names & test_names),
        "squeezenet_for_steps_2_to_5": as_saved["embedder"] == "squeezenet"
        and embed.cb_embedder_current_id == SQUEEZENET,
        "same_best_embedder_train_and_test":
            as_saved["embedder_best"] == as_saved["embedder_test"]
            and embed_best.cb_embedder_current_id == BEST_EMBEDDER,
        "no_picture_skipped": len(squeeze) == 24 and len(best_train) == 24
        and len(best_test) == 24,
        "ten_fold_cv_saved": as_saved["mode"] == CROSS_VALIDATION
        and as_saved["folds"] == [10, 10],
        "saved_as_test_on_test_data": as_saved["mode_1"] == TEST_ON_TEST,
        "five_models_each": sorted(by_model) == sorted(MODELS)
        and len(test_rows) == 5,
        "cv_sorted_by_accuracy": [r["CA"] for r in cv_rows]
        == sorted((r["CA"] for r in cv_rows), reverse=True),
        "my_cv_matches_widget": all(
            abs(by_model[m]["CA"] - float(ca)) < 1e-9
            for m, ca in zip(MODELS, by_hand)),
        "matrix_totals": all(
            int(np.sum(v["counts"])) == 24
            for cm in (cm_cv, cm_test, cm_best_cv) for v in cm.values()),
    }
    results = {
        "orange_version": Orange.__version__, "as_saved": as_saved,
        "folders": folders,
        "import_columns": [v.name for v in imported.domain.variables]
        + [m.name for m in imported.domain.metas],
        "squeezenet": describe(squeeze), "best_train": describe(best_train),
        "best_test": describe(best_test),
        "cv_scores": cv_rows, "confusion_cv": cm_cv,
        "best_cv_scores": best_cv_rows, "confusion_best_cv": cm_best_cv,
        "test_scores": test_rows, "confusion_test": cm_test,
        "tuned_settings": TUNED, "experiments": experiments,
        "checks": checks,
    }
    with open(os.path.join(oh.RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---------------- pictures ----------------
    keep = render_canvas(scheme, os.path.join(oh.RESULTS, "canvas.png"))
    grab(W("Import Images"), "import_images", (460, 220))
    grab(W("Import Images (test)"), "import_images_test", (460, 220))
    grab(embed, "image_embedding", (460, 300))
    grab(embed_best, "image_embedding_best", (460, 300))
    grab(W("Training Data"), "training_data", (1300, 480))
    for model in MODELS:
        grab(W(model), "model_" + model.lower().replace(" ", "_"),
             (460, 460))
    contact_sheet(os.path.join(oh.RESULTS, "examples.jpg"))
    del keep

    slim = {k: v for k, v in results.items() if k != "experiments"}
    print(json.dumps(slim, indent=1, default=str))
    failed = [k for k, v in checks.items() if k != "widget_problems" and not v]
    if problems or failed:
        print("CHECKS FAILED", problems, failed)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
