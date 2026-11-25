"""Build Lab6/submission/lab6_yahya.ows - the Orange workflow for Lab 6.

The workflow is put together from the real Orange widgets (with the Text
add-on), run on Tweets.csv so each widget can be set up on the actual
data, and then saved in Orange's own .ows format.

    /Applications/Orange.app/Contents/MacOS/python Lab6/build_workflow.py
"""
import importlib
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from AnyQt.QtCore import Qt  # noqa: E402
from orangecanvas.scheme import (  # noqa: E402
    SchemeArrowAnnotation, SchemeNode, SchemeTextAnnotation,
)
from orangewidget.utils.filedialogs import RecentPath  # noqa: E402
from Orange.widgets.unsupervised.utils import Jaccard  # noqa: E402

logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)

W = "Orange.widgets."
T = "orangecontrib.text.widgets."
FILE_W = W + "data.owfile.OWFile"
TABLE_W = W + "data.owtable.OWTable"
SAMPLER_W = W + "data.owdatasampler.OWDataSampler"
SELECT_W = W + "data.owselectrows.OWSelectRows"
DISTANCES_W = W + "unsupervised.owdistances.OWDistances"
HIERARCHICAL_W = W + "unsupervised.owhierarchicalclustering." \
    "OWHierarchicalClustering"
KMEANS_W = W + "unsupervised.owkmeans.OWKMeans"
CORPUS_W = T + "owcorpus.OWCorpus"
PREPROCESS_W = T + "owpreprocess.OWPreprocess"
CLOUD_W = T + "owwordcloud.OWWordCloud"
MAP_W = T + "owdocmap.OWDocMap"
BOW_W = T + "owbagofwords.OWTBagOfWords"

DATA_NAME = "Tweets.csv"
KEEP = ("name", "text", "tweet_coord", "tweet_created", "tweet_location")
TEXT = "text"
LOCATION = "tweet_location"
CLUSTER = "Cluster"
N_CLUSTERS = 3

FIXED_PROPORTION = 0            # Data Sampler: "Fixed proportion of data"
SAMPLE_PERCENT = 10
ROWS = 0                        # Distances: between rows
JACCARD = Jaccard               # Distances: metric id of "Jaccard"
COMPLETE = 3                    # Hierarchical Clustering: complete linkage
USA_MAP = 2                     # Document Map: World, Europe, USA
IS = 0                          # Select Rows: "is"

# Preprocess Text, step 4. Numbers are the positions of the check boxes.
LOWERCASE, ACCENTS, URLS = 0, 1, 3              # Transformation
STOPWORDS, REGEXP, MOST_FREQUENT = 0, 3, 6      # Filtering
PREPROCESS = {"preprocessors": [
    ("preprocess.transform", {"methods": [LOWERCASE, ACCENTS, URLS]}),
    ("preprocess.tokenize", {}),                # default: Regexp \w+
    ("preprocess.filter", {"methods": [STOPWORDS, REGEXP, MOST_FREQUENT],
                           "n_tokens": 100}),
    ("preprocess.ngrams", {"start": 1, "end": 2}),
]}

FILE = "File (Tweets.csv)"
TABLE = "Data Table"
SAMPLER = "Data Sampler (10%)"
CORPUS = "Corpus (text only)"
PRE = "Preprocess Text"
CLOUD = "Word Cloud"
DOCMAP = "Document Map (USA)"
BOW = "Bag of Words"
DIST = "Distances (Jaccard)"
HC = "Hierarchical Clustering"
KM = "k-Means (k=3)"
CLUSTERS = ["C1", "C2", "C3"]
HC_SELECT = ["HC " + c for c in CLUSTERS]
HC_CLOUD = ["Word Cloud HC " + c for c in CLUSTERS]
KM_SELECT = ["KM " + c for c in CLUSTERS]
KM_CLOUD = ["Word Cloud KM " + c for c in CLUSTERS]

HC_Y = [80, 230, 380]
KM_Y = [620, 770, 920]

NODES = {
    FILE: (FILE_W, (60, 300), {}),
    TABLE: (TABLE_W, (250, 150), {}),
    SAMPLER: (SAMPLER_W, (250, 300), {
        "sampling_type": FIXED_PROPORTION,
        "sampleSizePercentage": SAMPLE_PERCENT,
        "use_seed": True, "replacement": False, "stratify": False}),
    CORPUS: (CORPUS_W, (440, 300), {}),
    PRE: (PREPROCESS_W, (630, 300), {"storedsettings": PREPROCESS}),
    CLOUD: (CLOUD_W, (820, 150), {}),
    BOW: (BOW_W, (820, 300), {"wlocal": "Count", "wglobal": "Smooth IDF",
                              "normalization": "(None)"}),
    DOCMAP: (MAP_W, (820, 450), {}),
    DIST: (DISTANCES_W, (1010, 230), {"axis": ROWS, "metric_id": JACCARD}),
    HC: (HIERARCHICAL_W, (1200, 230), {"linkage": COMPLETE,
                                       "top_n": N_CLUSTERS}),
    KM: (KMEANS_W, (1200, 770), {"k": N_CLUSTERS, "optimize_k": False}),
}
for _i in range(N_CLUSTERS):
    NODES[HC_SELECT[_i]] = (SELECT_W, (1420, HC_Y[_i]), {})
    NODES[HC_CLOUD[_i]] = (CLOUD_W, (1620, HC_Y[_i]), {})
    NODES[KM_SELECT[_i]] = (SELECT_W, (1420, KM_Y[_i]), {})
    NODES[KM_CLOUD[_i]] = (CLOUD_W, (1620, KM_Y[_i]), {})

LINKS_LOAD = [
    (FILE, "Data", TABLE, "Data"),
    (FILE, "Data", SAMPLER, "Data"),
    (SAMPLER, "Data Sample", CORPUS, "Data"),
]
LINKS_TEXT = [
    (CORPUS, "Corpus", PRE, "Corpus"),
    (PRE, "Corpus", CLOUD, "Corpus"),
    (PRE, "Corpus", DOCMAP, "Data"),
    (PRE, "Corpus", BOW, "Corpus"),
    (BOW, "Corpus", DIST, "Data"),
    (DIST, "Distances", HC, "Distances"),
    (BOW, "Corpus", KM, "Data"),
]
LINKS_CLUSTERS = (
    [(HC, "Selected Data", s, "Data") for s in HC_SELECT]
    + [(KM, "Data", s, "Data") for s in KM_SELECT]
    + [(s, "Matching Data", c, "Corpus")
       for s, c in zip(HC_SELECT + KM_SELECT, HC_CLOUD + KM_CLOUD)]
)

RED, BLUE, GREEN = "#C1272D", "#1F77B4", "#39B54A"
TEXTS = [
    (-60, 20, 300, 60,
     "**Lab 6**  \nText Mining and Clustering"),
    (-60, 400, 230, 130,
     "**Step 2: Load the tweets**  \nOnly the text columns are kept "
     "(name, text, tweet_coord, tweet_created, tweet_location)."),
    (160, 20, 220, 70,
     "**Step 2: Data Table**  \nA look at the loaded tweets."),
    (210, 420, 240, 110,
     "**Step 3: Sample and corpus**  \n10% of the tweets, then a corpus "
     "that uses only the text column."),
    (520, 400, 240, 150,
     "**Step 4: Preprocess Text**  \nLowercase, no accents, no URLs, "
     "stop words and RegExp filters, 100 most frequent tokens, "
     "n-grams 1 to 2."),
    (720, 20, 220, 70,
     "**Steps 3-4: Word Cloud**  \nThe most common words."),
    (720, 550, 230, 90,
     "**Step 5: Document Map**  \nTweets per US state, from "
     "tweet_location."),
    (1010, 330, 250, 90,
     "**Step 6: Bag of Words**  \nCount for term frequency, Smooth IDF "
     "for document frequency."),
    (1010, 20, 300, 110,
     "**Step 7: Hierarchical clustering**  \nJaccard distances, complete "
     "linkage, Top N = 3. One Select Rows and one Word Cloud for each "
     "cluster."),
    (940, 860, 320, 110,
     "**Step 8: k-Means**  \nk = 3 on the same bag of words. One Select "
     "Rows and one Word Cloud for each cluster."),
    (1740, 190, 220, 90,
     "**Step 7: Word clouds**  \nHierarchical clusters C1, C2 and C3."),
    (1740, 730, 220, 90,
     "**Step 8: Word clouds**  \nk-Means clusters C1, C2 and C3."),
]
ARROWS = [
    ((60, 400), (60, 365), RED),            # File
    ((250, 90), (250, 115), RED),           # Data Table
    ((250, 420), (250, 385), RED),          # Sampler
    ((630, 400), (630, 365), BLUE),         # Preprocess Text
    ((1005, 352), (880, 322), BLUE),        # Bag of Words
    ((820, 90), (820, 115), BLUE),          # Word Cloud
    ((820, 550), (820, 520), BLUE),         # Document Map
    ((1105, 130), (1105, 195), GREEN),      # Distances + Hierarchical
    ((1200, 860), (1200, 835), GREEN),      # k-Means
    ((1740, 230), (1690, 230), GREEN),      # HC word clouds
    ((1740, 770), (1690, 770), GREEN),      # k-Means word clouds
]


def output(scheme, node, channel):
    """Current value on a node's output channel."""
    sm = scheme.signal_manager
    for link in scheme.links:
        if link.source_node is node and link.source_channel.name == channel:
            contents = sm.link_contents(link)
            if contents:
                value = next(iter(contents.values()))
                return value.get_value() if hasattr(value, "get_value") \
                    else value
    raise LookupError("%s / %s" % (node.title, channel))


def keep_columns(file_widget, keep):
    """File widget: set the role of every other column to "skip"."""
    model = file_widget.domain_editor.model()
    for row in range(len(model.variables)):
        name = model.data(model.index(row, 0), Qt.DisplayRole)
        if name not in keep:
            model.setData(model.index(row, 2), "skip", Qt.EditRole)
    file_widget.apply_domain_edit()


def use_text_only(corpus_widget, name):
    """Corpus widget: one used text feature, the rest ignored."""
    strings = [v for v in corpus_widget.corpus.domain.metas if v.is_string]
    corpus_widget.used_attrs_model[:] = [v for v in strings
                                         if v.name == name]
    corpus_widget.unused_attrs_model[:] = [v for v in strings
                                           if v.name != name]
    corpus_widget.update_feature_selection()


def set_map(docmap, attribute, map_index):
    docmap.selected_attr = attribute
    docmap.selected_map = map_index
    docmap.map_combo.setCurrentIndex(map_index)
    docmap.on_map_change()


def select_cluster(select, index):
    """Select Rows: Cluster is C<index> (index counts from 1)."""
    select.remove_all_rows()
    select.add_row(select.data.domain[CLUSTER], IS, (index,))
    select.conditions_changed()
    select.commit.now()


def build():
    os.makedirs(oh.SUBMISSION, exist_ok=True)
    csv_path = os.path.join(oh.SUBMISSION, DATA_NAME)
    reg = oh.registry()
    scheme = oh.new_scheme(
        title="Lab 6 - Text Mining and Clustering (airline tweets)",
        description="Turns a 10% sample of the airline tweets into a "
                    "corpus, preprocesses it, shows word clouds and a "
                    "document map, and clusters the tweets with "
                    "hierarchical clustering and k-Means.")
    props = {t: dict(p) for t, (_, _, p) in NODES.items()}
    props[FILE]["active_path"] = RecentPath(csv_path, "basedir", DATA_NAME)
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
    oh.settle(scheme, quiet=2.0, timeout=240)

    # ---- Step 2: only the text columns ----
    keep_columns(widgets[FILE], KEEP)
    connect(LINKS_LOAD)
    oh.settle(scheme, quiet=3.0, timeout=240)

    # ---- Step 3: the corpus uses only the "text" column ----
    use_text_only(widgets[CORPUS], TEXT)
    oh.settle(scheme, quiet=2.0, timeout=240)

    # ---- Steps 4-8 ----
    connect(LINKS_TEXT)
    oh.settle(scheme, quiet=3.0, timeout=900)
    set_map(widgets[DOCMAP], LOCATION, USA_MAP)
    hierarchical = widgets[HC]
    hierarchical.top_n = N_CLUSTERS
    hierarchical._top_n_changed()                 # selection: "Top N"
    connect(LINKS_CLUSTERS)
    oh.settle(scheme, quiet=3.0, timeout=900)
    for titles in (HC_SELECT, KM_SELECT):
        for i, title in enumerate(titles, start=1):
            select_cluster(widgets[title], i)
    oh.settle(scheme, quiet=3.0, timeout=900)

    for x, y, w, h, text in TEXTS:
        scheme.add_annotation(SchemeTextAnnotation(
            (x, y, w, h), text, "text/markdown",
            font={"family": "Helvetica", "size": 14}))
    for start, end, color in ARROWS:
        scheme.add_annotation(SchemeArrowAnnotation(start, end, color))

    # quick look at what came out
    print("file:", len(output(scheme, nodes[FILE], "Data")), "rows,",
          [v.name for v in output(scheme, nodes[FILE], "Data").domain.metas])
    corpus = output(scheme, nodes[PRE], "Corpus")
    print("corpus:", len(corpus), "used:",
          [v.name for v in corpus.text_features])
    bow = output(scheme, nodes[BOW], "Corpus")
    print("bag of words:", bow.X.shape)
    for group in (HC_SELECT, KM_SELECT):
        print([len(output(scheme, nodes[s], "Matching Data"))
               for s in group])

    problems = oh.widget_problems(scheme)
    scheme.sync_node_properties()
    with open(oh.WORKFLOW, "wb") as f:
        scheme.save_to(f, pretty=True, pickle_fallback=True)
    print("saved", oh.WORKFLOW)
    print("widget problems:", problems or "none")
    os._exit(0)


if __name__ == "__main__":
    build()
