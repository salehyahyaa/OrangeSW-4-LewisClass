"""Run Lab6/submission/lab6_yahya.ows, check it, and collect what the report
needs.

Writes to Lab6/results/:
    results.json   numbers used in the report
    canvas.png     picture of the workflow canvas
    *.png          screenshots of the widgets

The Word Cloud and Document Map widgets draw in a web view, which only
works with real windows, so this script opens the widget windows on the
screen for a moment while it takes the pictures.

    /Applications/Orange.app/Contents/MacOS/python Lab6/run_workflow.py
"""
import json
import logging
import os
import sys
import time
import warnings
from collections import Counter, defaultdict

os.environ.setdefault("QT_QPA_PLATFORM", "cocoa")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

import numpy as np  # noqa: E402
from AnyQt.QtCore import QRectF, Qt  # noqa: E402
from AnyQt.QtGui import QImage, QPainter  # noqa: E402
from orangewidget.utils.overlay import MessageOverlayWidget  # noqa: E402
from Orange.widgets.unsupervised.utils import (  # noqa: E402
    Cosine, Euclidean, Jaccard, MetricDefs,
)
from orangecontrib.text.country_codes import CC_USA, INV_CC_USA  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import build_workflow as bw  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

LINKAGES = ["Single", "Average", "Weighted", "Complete", "Ward"]


def pump(seconds):
    end = time.time() + seconds
    while time.time() < end:
        oh.app().processEvents()
        time.sleep(0.01)


def grab(w, name, size, wait=0.4, part=None):
    """Screenshot of a widget window (or of one part of it)."""
    w.show()
    w.resize(*size)
    w.raise_()
    pump(wait)
    for tip in w.findChildren(MessageOverlayWidget):   # "Ok, got it" tips
        tip.hide()
    pump(0.2)
    path = os.path.join(oh.RESULTS, name + ".png")
    (part or w).grab().save(path)
    w.hide()
    pump(0.1)
    return path


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


def top_words(cloud, n=15):
    """The words a Word Cloud widget shows, heaviest first."""
    return [[w, round(float(c), 3)]
            for w, c in cloud.corpus_counter.most_common(n)]


def describe_cluster(corpus, n=12, examples=4):
    """Share of the tweets in a cluster that contain each token."""
    docs = [set(tokens) for tokens in corpus.ngrams]
    counts = Counter(t for d in docs for t in d)
    size = len(docs)
    texts = [str(t) for t in corpus.get_column(corpus.domain[bw.TEXT])]
    return {
        "size": size,
        "empty": sum(1 for d in docs if not d),
        "mean_tokens": round(float(np.mean([len(d) for d in docs])), 2)
        if size else 0,
        "share": [[t, round(c / size, 3)] for t, c in counts.most_common(n)],
        "examples": texts[:examples],
    }


def strip(paths, labels, out, height=520):
    """Put several pictures next to each other with a label over each."""
    pictures = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        pictures.append(im.resize((round(im.width * height / im.height),
                                   height)))
    gap, head = 12, 44
    width = sum(p.width for p in pictures) + gap * (len(pictures) - 1)
    sheet = Image.new("RGB", (width, height + head), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for p, label in zip(pictures, labels):
        draw.text((x + 8, 8), label, fill="black", font=font)
        sheet.paste(p, (x, head))
        x += p.width + gap
    sheet.save(out)
    for p in paths:
        os.remove(p)


def main():
    os.makedirs(oh.RESULTS, exist_ok=True)
    try:    # screenshots with a light background, whatever the Mac is set to
        oh.app().styleHints().setColorScheme(Qt.ColorScheme.Light)
    except AttributeError:
        pass
    scheme = oh.load_scheme(oh.WORKFLOW)
    oh.settle(scheme, quiet=3.0, timeout=900)
    nodes = {n.title: n for n in scheme.nodes}
    W = lambda title: scheme.widget_for_node(nodes[title])  # noqa: E731
    out = lambda title, ch: bw.output(scheme, nodes[title], ch)  # noqa: E731

    checks = []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    problems = oh.widget_problems(scheme, groups=("Error",))
    check("no widget errors", not problems, problems)
    warnings_ = oh.widget_problems(scheme, groups=("Warning",))

    # ---- Step 2 ----
    raw = out(bw.FILE, "Data")
    columns = [v.name for v in raw.domain.variables + raw.domain.metas]
    check("file has only the five text columns",
          sorted(columns) == sorted(bw.KEEP), columns)
    check("14640 tweets", len(raw) == 14640, len(raw))
    filled = {c: int(sum(1 for v in raw.get_column(raw.domain[c])
                         if str(v).strip() not in ("", "?", "nan")))
              for c in bw.KEEP}
    sample_rows = [[str(raw[i][c]) for c in bw.KEEP] for i in range(6)]

    # ---- Step 3 ----
    sample = out(bw.SAMPLER, "Data Sample")
    check("10% sample", len(sample) == round(len(raw) * 0.10), len(sample))
    corpus_widget = W(bw.CORPUS)
    used = [v.name for v in corpus_widget.used_attrs]
    check("corpus uses only text", used == [bw.TEXT], used)

    # ---- Step 4 ----
    pre = W(bw.PRE)
    stored = dict((k, v) for k, v in pre.storedsettings["preprocessors"])
    check("transform: lowercase, accents, urls",
          sorted(stored["preprocess.transform"]["methods"]) == [0, 1, 3])
    check("filter: stopwords, regexp, most frequent 100",
          sorted(stored["preprocess.filter"]["methods"]) == [0, 3, 6]
          and stored["preprocess.filter"]["n_tokens"] == 100)
    check("n-grams 1 to 2", stored["preprocess.ngrams"] ==
          {"start": 1, "end": 2}, stored["preprocess.ngrams"])
    processed = out(bw.PRE, "Corpus")
    unigrams = Counter(t for doc in processed.tokens for t in doc)
    check("100 token types", len(unigrams) == 100, len(unigrams))
    ngram_docs = [list(d) for d in processed.ngrams]
    n_bigrams = len({t for d in ngram_docs for t in d if " " in t})

    # ---- Step 5 ----
    docmap = W(bw.DOCMAP)
    check("map: tweet_location on USA", docmap.selected_attr == bw.LOCATION
          and docmap.map_combo.itemData(docmap.selected_map) == "us_aea_en",
          (docmap.selected_attr, docmap.selected_map))
    states = defaultdict(int)
    located = 0
    for locations in docmap._iter_locations():
        keys = {INV_CC_USA.get(loc, loc) for loc in locations}
        keys = {k for k in keys if k in CC_USA}
        located += bool(keys)
        for key in keys:
            states[key] += 1
    state_rows = sorted(([CC_USA[k], k, n] for k, n in states.items()),
                        key=lambda r: -r[2])
    locations = [str(v).strip() for v in
                 processed.get_column(processed.domain[bw.LOCATION])]
    with_location = sum(1 for v in locations if v not in ("", "?", "nan"))

    # ---- Step 6 ----
    bow_widget = W(bw.BOW)
    check("bag of words: Count and Smooth IDF",
          (bow_widget.wlocal, bow_widget.wglobal) == ("Count", "Smooth IDF"),
          (bow_widget.wlocal, bow_widget.wglobal))
    bow = out(bw.BOW, "Corpus")

    # ---- Steps 7-8 ----
    dist, hc, km = W(bw.DIST), W(bw.HC), W(bw.KM)
    check("Jaccard distance", dist.metric_id == Jaccard, dist.metric_id)
    check("complete linkage, top 3",
          LINKAGES[hc.linkage] == "Complete" and hc.top_n == 3
          and hc.selection_method == 2,
          (hc.linkage, hc.top_n, hc.selection_method))
    check("k-Means k=3", km.k == 3 and not km.optimize_k, km.k)

    def clusters(selects, clouds):
        rows = []
        for s, c in zip(selects, clouds):
            part = out(s, "Matching Data")
            info = describe_cluster(part)
            info["cloud"] = top_words(W(c))
            info["ids"] = [int(i) for i in part.ids]
            rows.append(info)
        return rows

    hc_clusters = clusters(bw.HC_SELECT, bw.HC_CLOUD)
    km_clusters = clusters(bw.KM_SELECT, bw.KM_CLOUD)
    for name, group in (("hierarchical", hc_clusters), ("k-Means", km_clusters)):
        sizes = [g["size"] for g in group]
        check("%s: three clusters cover the sample" % name,
              sum(sizes) == len(sample) and min(sizes) > 0, sizes)
    cross = [[len(set(a["ids"]) & set(b["ids"])) for b in km_clusters]
             for a in hc_clusters]

    # ---- pictures ----
    keep = render_canvas(scheme, os.path.join(oh.RESULTS, "canvas.png"))
    grab(W(bw.FILE), "file", (620, 700))
    grab(W(bw.TABLE), "data_table", (1500, 520))
    grab(W(bw.SAMPLER), "data_sampler", (300, 520))
    grab(corpus_widget, "corpus", (560, 420))
    grab(pre, "preprocess_text", (900, 900))
    grab(W(bw.CLOUD), "word_cloud_preprocessed", (1050, 640), wait=3.0)
    grab(docmap, "document_map", (1050, 640), wait=3.0)
    grab(bow_widget, "bag_of_words", (300, 420))
    grab(dist, "distances", (300, 480))
    grab(hc, "hierarchical_clustering", (1100, 760), wait=1.5)
    grab(km, "kmeans", (300, 420))
    for i, c in enumerate(bw.CLUSTERS):
        grab(W(bw.HC_SELECT[i]), "select_hc_%s" % c.lower(), (700, 330))
        grab(W(bw.HC_CLOUD[i]), "cloud_hc_%s" % c.lower(), (950, 560),
             wait=3.0)
        grab(W(bw.KM_CLOUD[i]), "cloud_km_%s" % c.lower(), (950, 560),
             wait=3.0)
    preprocessed_cloud = top_words(W(bw.CLOUD), 25)

    # Step 3 picture: the Word Cloud straight from the Corpus, as the
    # workflow was before Preprocess Text was put in between (step 4).
    link = next(l for l in scheme.links if l.source_node is nodes[bw.PRE]
                and l.sink_node is nodes[bw.CLOUD])
    scheme.remove_link(link)
    direct = scheme.new_link(
        nodes[bw.CORPUS], nodes[bw.CORPUS].output_channel("Corpus"),
        nodes[bw.CLOUD], nodes[bw.CLOUD].input_channel("Corpus"))
    oh.settle(scheme, quiet=3.0, timeout=600)
    raw_cloud = top_words(W(bw.CLOUD), 25)
    grab(W(bw.CLOUD), "word_cloud_raw", (1050, 640), wait=3.0)
    scheme.remove_link(direct)
    scheme.new_link(nodes[bw.PRE], nodes[bw.PRE].output_channel("Corpus"),
                    nodes[bw.CLOUD], nodes[bw.CLOUD].input_channel("Corpus"))
    oh.settle(scheme, quiet=3.0, timeout=600)
    check("word cloud is back on the preprocessed corpus",
          top_words(W(bw.CLOUD), 25) == preprocessed_cloud)

    # ---- Step 9: other distances and linkages ----
    def set_hierarchical(metric, linkage):
        if dist.metric_id != metric:
            dist._metric_changed(metric)
        oh.settle(scheme, quiet=2.0, timeout=900)
        if hc.linkage != LINKAGES.index(linkage):
            hc.linkage = LINKAGES.index(linkage)
            hc._invalidate_clustering()
        hc.top_n = 3
        hc._top_n_changed()
        oh.settle(scheme, quiet=3.0, timeout=900)
        for i, title in enumerate(bw.HC_SELECT, start=1):
            bw.select_cluster(W(title), i)
        oh.settle(scheme, quiet=3.0, timeout=900)

    experiments = []
    for metric, linkage, picture in [
            (Jaccard, "Single", None), (Jaccard, "Average", None),
            (Jaccard, "Ward", "exp_jaccard_ward"),
            (Cosine, "Complete", "exp_cosine_complete"),
            (Cosine, "Ward", None), (Euclidean, "Complete", None),
            (Euclidean, "Ward", None)]:
        set_hierarchical(metric, linkage)
        group = clusters(bw.HC_SELECT, bw.HC_CLOUD)
        experiments.append({
            "metric": MetricDefs[metric].name, "linkage": linkage,
            "picture": picture, "sizes": [g["size"] for g in group],
            "empty": [g["empty"] for g in group],
            "share": [g["share"][:6] for g in group]})
        print(MetricDefs[metric].name, linkage, [g["size"] for g in group])
        if picture:
            parts = [grab(W(c), "%s_%d" % (picture, i), (950, 560),
                          wait=3.0, part=W(c).webview)
                     for i, c in enumerate(bw.HC_CLOUD)]
            strip(parts, ["%s (%d tweets)" % (c, g["size"])
                          for c, g in zip(bw.CLUSTERS, group)],
                  os.path.join(oh.RESULTS, picture + ".png"))
    set_hierarchical(Jaccard, "Complete")
    again = clusters(bw.HC_SELECT, bw.HC_CLOUD)
    check("back to Jaccard + complete after the experiments",
          [g["size"] for g in again] == [g["size"] for g in hc_clusters])

    results = {
        "warnings": warnings_,
        "step2": {"rows": len(raw), "columns": columns, "filled": filled,
                  "sample_rows": sample_rows},
        "step3": {"sample": len(sample), "used": used,
                  "language": corpus_widget.language, "cloud": raw_cloud},
        "step4": {"tokens": int(sum(unigrams.values())),
                  "types": len(unigrams), "bigram_types": n_bigrams,
                  "empty_docs": sum(1 for d in ngram_docs if not d),
                  "top_tokens": unigrams.most_common(100),
                  "cloud": preprocessed_cloud},
        "step5": {"with_location": with_location, "located": located,
                  "states": state_rows},
        "step6": {"rows": int(bow.X.shape[0]), "features": int(bow.X.shape[1])},
        "step7": [{k: v for k, v in g.items() if k != "ids"}
                  for g in hc_clusters],
        "step8": [{k: v for k, v in g.items() if k != "ids"}
                  for g in km_clusters],
        "cross": cross,
        "step9": experiments,
    }
    with open(os.path.join(oh.RESULTS, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    failed = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print("%-55s %s %s" % (name, "ok" if ok else "FAILED",
                               "" if ok else detail))
    print("warnings:", warnings_ or "none")
    print("CHECKS FAILED" if failed else "ALL CHECKS PASSED")
    del keep
    sys.stdout.flush()
    os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
