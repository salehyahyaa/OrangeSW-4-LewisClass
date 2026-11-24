"""Helpers for building and running Orange Canvas workflows without the GUI.

Everything here drives the real Orange widgets (the same classes the Orange
app uses), just with Qt's "offscreen" platform so no window is shown.

Run with Orange's bundled Python:
    /Applications/Orange.app/Contents/MacOS/python ML1/<script>.py
"""
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from AnyQt.QtCore import QThreadPool  # noqa: E402
from AnyQt.QtWidgets import QApplication  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMISSION = os.path.join(REPO, "ML1", "submission")
DATA_NAME = "ChicagoCrimes2019.csv"
WORKFLOW = os.path.join(SUBMISSION, "mp1_yahya.ows")
RESULTS = os.path.join(REPO, "ML1", "results")

_app = None
_registry = None
_config = None


def app():
    global _app
    if _app is None:
        _app = QApplication.instance() or QApplication(["orange-headless"])
    return _app


def registry():
    """Discover all installed Orange widgets (same as the canvas does)."""
    global _registry, _config
    if _registry is None:
        app()
        from Orange.canvas import config as orange_config
        from orangecanvas import config as canvas_config
        from orangecanvas.registry import WidgetRegistry, set_global_registry

        _config = orange_config.Config()
        canvas_config.set_default(_config)
        _registry = WidgetRegistry()
        discovery = _config.widget_discovery(_registry)
        discovery.run(_config.widgets_entry_points())
        set_global_registry(_registry)
    return _registry


def new_scheme(basedir=SUBMISSION, title="", description=""):
    registry()
    scheme = _config.workflow_constructor(title=title, description=description)
    scheme.set_runtime_env("basedir", basedir)
    return scheme


def load_scheme(path):
    scheme = new_scheme(basedir=os.path.dirname(os.path.abspath(path)))
    with open(path, "rb") as f:
        scheme.load_from(f, registry=registry())
    for node in scheme.nodes:  # force every widget to be created
        scheme.widget_for_node(node)
    return scheme


def settle(scheme, quiet=2.0, timeout=120):
    """Process Qt events until every widget has finished computing."""
    a = app()
    sm = scheme.signal_manager
    pool = QThreadPool.globalInstance()
    start = last_busy = time.time()
    while True:
        a.processEvents()
        time.sleep(0.01)
        waiting = [n.title for n in scheme.nodes if sm.is_blocking(n)
                   or sm.is_invalidated(n) or sm.is_active(n)]
        busy = sm.has_pending() or pool.activeThreadCount() > 0 or waiting
        now = time.time()
        if busy:
            last_busy = now
        elif now - last_busy > quiet:
            return
        if now - start > timeout:
            raise TimeoutError(
                "workflow did not finish in %s s; pending=%s threads=%s "
                "waiting on %s" % (timeout, sm.has_pending(),
                                   pool.activeThreadCount(), waiting))


def widget(scheme, title):
    for node in scheme.nodes:
        if node.title == title:
            return scheme.widget_for_node(node)
    raise KeyError(title)


def widget_problems(scheme, groups=("Error", "Warning")):
    """Return {title: [messages]} for widgets showing an error or warning."""
    out = {}
    for node in scheme.nodes:
        w = scheme.widget_for_node(node)
        msgs = []
        for group in groups:
            g = getattr(w, group, None)
            if g is not None:
                msgs += ["%s: %s" % (group, m) for m in g.active]
        if msgs:
            out[node.title] = msgs
    return out
