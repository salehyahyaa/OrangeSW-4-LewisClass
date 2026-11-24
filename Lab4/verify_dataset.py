"""Check that the Datasets widget is loading the dataset Lab 4 asks for:
"HDI ('Human Development Index')".

Three independent checks:
  1. the catalogue entry on Orange's dataset server
  2. what the real Datasets widget outputs when "HDI" is selected
  3. the values themselves, against the published UNDP 2015 HDI figures

    /Applications/Orange.app/Contents/MacOS/python Lab4/verify_dataset.py
"""
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import orange_headless as oh  # noqa: E402

from orangecanvas.scheme import SchemeNode  # noqa: E402

logging.getLogger("orangecanvas").setLevel(logging.CRITICAL)
DATASETS = "Orange.widgets.data.owdatasets.OWDataSets"

# HDI values for 2015 as published by UNDP (Human Development Report 2016,
# Table 1), typed in from the report - not taken from Orange.
UNDP_2015 = {"Norway": 0.949, "Australia": 0.939, "Switzerland": 0.939,
             "Germany": 0.926, "United States": 0.920, "Japan": 0.903,
             "China": 0.738, "India": 0.624, "Niger": 0.353,
             "Central African Republic": 0.352}


def load_with_widget(dataset_id):
    """Run the real Datasets widget with this entry selected."""
    scheme = oh.new_scheme()
    cls_props = {"selected_id": dataset_id, "__version__": 2}
    node = SchemeNode(oh.registry().widget(DATASETS), title="Datasets",
                      properties=cls_props)
    scheme.add_node(node)
    w = scheme.widget_for_node(node)
    oh.settle(scheme, quiet=3.0, timeout=180)
    info = w.selected_dataset()
    data = w.Outputs.data.last_sent if hasattr(w.Outputs.data, "last_sent") \
        else None
    if data is None:
        from Orange.data import Table
        data = Table(w.current_output) if w.current_output else None
    return info, data, w.current_output


def main():
    from Orange.widgets.data.owdatasets import OWDataSets, list_remote
    report = {}

    # ---- check 1: the catalogue ----
    catalogue = list_remote(OWDataSets.INDEX_URL)
    mentions = {k[-1]: d for k, d in catalogue.items()
                if "human development" in json.dumps(d).lower()}
    entry = mentions[oh.DATASET_ID]
    report["check1_catalogue"] = {
        "datasets_on_server": len(catalogue),
        "entries_about_human_development": {
            k: {"title": d["title"], "language": d["language"],
                "instances": d["instances"], "variables": d["variables"]}
            for k, d in mentions.items()},
        "chosen": oh.DATASET_ID, "title": entry["title"],
        "description": entry["description"], "year": entry["year"],
        "collection": entry["collection"], "target": entry["target"],
        "instances": entry["instances"], "variables": entry["variables"],
        "ok": entry["title"] == "HDI"
        and "Human Development Index (HDI)" in entry["description"]
        and entry["language"] == "English",
    }

    # ---- check 2: what the widget really outputs ----
    info, data, path = load_with_widget(oh.DATASET_ID)
    domain = data.domain
    names = [v.name for v in domain.variables + domain.metas]
    hdi_name = next(n for n in names if n.strip().upper() == "HDI")
    report["check2_widget_output"] = {
        "selected_title_in_widget": info.title,
        "file": os.path.basename(path),
        "rows": len(data),
        "features": len(domain.attributes),
        "target": domain.class_var.name if domain.class_var else None,
        "metas": [m.name for m in domain.metas],
        "has_HDI_column": hdi_name,
        "ok": info.title == "HDI" and len(data) == entry["instances"]
        and os.path.basename(path) == oh.DATASET_ID,
    }

    # ---- check 3: the numbers against UNDP ----
    country_var = next(m for m in domain.metas if m.is_string)
    countries = [str(c) for c in data.get_column(country_var)]
    hdi = data.get_column(hdi_name)
    lookup = dict(zip(countries, hdi))
    compared = {c: {"undp": v, "orange": round(float(lookup[c]), 3)
                    if c in lookup else None}
                for c, v in UNDP_2015.items()}
    order = np.argsort(-hdi)
    report["check3_values"] = {
        "country_column": country_var.name,
        "hdi_min": float(np.nanmin(hdi)), "hdi_max": float(np.nanmax(hdi)),
        "top5": [(countries[i], round(float(hdi[i]), 3)) for i in order[:5]],
        "bottom5": [(countries[i], round(float(hdi[i]), 3))
                    for i in order[-5:]],
        "compared_with_undp": compared,
        "ok": all(d["orange"] is not None and abs(d["orange"] - d["undp"])
                  < 0.0015 for d in compared.values()),
    }
    report["all_feature_names"] = [a.name for a in domain.attributes]
    report["VERIFIED"] = all(report[k]["ok"] for k in (
        "check1_catalogue", "check2_widget_output", "check3_values"))

    os.makedirs(oh.RESULTS, exist_ok=True)
    with open(os.path.join(oh.RESULTS, "dataset_check.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=1))
    sys.exit(0 if report["VERIFIED"] else 1)


if __name__ == "__main__":
    main()
