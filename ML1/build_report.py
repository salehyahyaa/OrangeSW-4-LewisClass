"""Build submission/mp1_yahya.pdf from the numbers the workflow produced.

Reads ML1/results/ (written by run_workflow.py), draws two charts, writes
ML1/results/report.html and prints it to PDF with headless Chrome.

    /Applications/Orange.app/Contents/MacOS/python ML1/build_report.py
"""
import csv
import html
import json
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
PDF = os.path.join(HERE, "submission", "mp1_yahya.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

BLUE, ORANGE = "#2a78d6", "#eb6834"      # positive / negative
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

# Official City of Chicago names for the community-area numbers that appear
# in the findings (the dataset itself only has the numbers).
AREA_NAMES = {"9": "Edison Park", "12": "Forest Glen",
              "23": "Humboldt Park", "25": "Austin",
              "26": "West Garfield Park", "29": "North Lawndale",
              "36": "Oakland"}

plt.rcParams.update({
    "font.family": "Helvetica", "font.size": 9, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": INK2, "svg.fonttype": "none",
})


def read_orange_csv(name):
    """Read a CSV saved by Orange (3 header rows: names, types, roles)."""
    with open(os.path.join(RESULTS, name), newline="") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    return [dict(zip(header, r)) for r in rows[3:]]


def pretty(feature):
    """'Primary Type=NARCOTICS' -> 'Crime type: Narcotics'."""
    if "=" not in feature:
        return feature
    group, level = feature.split("=", 1)
    if group == "Primary Type":
        return "Crime type: " + level.title()
    if group == "Community Area":
        name = AREA_NAMES.get(level)
        return "Community area %s%s" % (level, " (%s)" % name if name else "")
    if group == "Domestic":
        return "Domestic incident"
    return feature


def bar_chart(path, labels, values, fmt, xlabel, signed=False, xlim=None):
    fig, ax = plt.subplots(figsize=(7.0, 0.3 * len(labels) + 0.75))
    y = range(len(labels))
    colors = [(ORANGE if v < 0 else BLUE) for v in values] if signed else BLUE
    ax.barh(y, values, height=0.62, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    span = max(abs(v) for v in values)
    for i, v in enumerate(values):
        ax.text(v + (span * 0.015 if v >= 0 else -span * 0.015), i, fmt(v),
                va="center", ha="left" if v >= 0 else "right",
                color=INK2, fontsize=8)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    if signed:
        ax.axvline(0, color=MUTED, linewidth=0.8)
    if xlim:
        ax.set_xlim(*xlim)
    else:
        ax.margins(x=0.14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pct(rate):
    """0.2117 -> '21.2%'; never rounds a not-quite-100% rate up to 100%."""
    if 0.9995 <= rate < 1:
        return "%.2f%%" % (100 * rate)
    return "%.1f%%" % (100 * rate)


def table(headers, rows, align=None):
    align = align or ["l"] * len(headers)
    out = ["<table><thead><tr>"]
    out += ['<th class="%s">%s</th>' % (a, h) for h, a in zip(headers, align)]
    out.append("</tr></thead><tbody>")
    for row in rows:
        out.append("<tr>" + "".join(
            '<td class="%s">%s</td>' % (a, c) for c, a in zip(row, align))
            + "</tr>")
    out.append("</tbody></table>")
    return "".join(out)


def main():
    with open(os.path.join(RESULTS, "results.json")) as f:
        R = json.load(f)
    ex, pipe, sc = R["explore"], R["pipeline"], R["scores"]
    cls, base = R["classification_view"], R["baseline_scores"]
    target = pipe["target"]
    e = html.escape

    def other(row):
        return row["Feature 2"] if row["Feature 1"] == target \
            else row["Feature 1"]

    all_corr = [(other(r), float(r["Correlation"]))
                for r in read_orange_csv("all_correlations.csv")]
    corr = {name: r for name, r in all_corr}
    coefs = {r["name"]: float(r["coef"])
             for r in read_orange_csv("model_coefficients.csv")}
    top_corr = [(other(r), r["Correlation"]) for r in R["top_correlations"]]
    top_coef = [(r["name"], r["coef"]) for r in R["top_coefficients"]]
    positives = [n for n, r in sorted(all_corr, key=lambda t: -t[1])][:10]
    extra_pos = [n for n in positives if n not in dict(top_corr)]
    counts = {t["type"]: t for t in ex["by_type"]}

    # ---------------- charts ----------------
    types = sorted((t for t in ex["by_type"] if t["count"] >= 1000),
                   key=lambda t: -t["arrest_rate"])
    bar_chart(os.path.join(RESULTS, "chart_arrest_rate.svg"),
              ["%s  (%s)" % (t["type"].title(), format(t["count"], ","))
               for t in types],
              [100 * t["arrest_rate"] for t in types],
              lambda v: "%.0f%%" % v,
              "Share of reported crimes that led to an arrest (%)",
              xlim=(0, 112))
    bar_chart(os.path.join(RESULTS, "chart_correlations.svg"),
              [pretty(n) for n, _ in top_corr], [r for _, r in top_corr],
              lambda v: "%+.3f" % v,
              "Pearson correlation with Arrest (training sample)",
              signed=True, xlim=(-0.26, 0.56))

    # ---------------- tables ----------------
    missing = sorted(((k, v) for k, v in ex["missing"].items() if v),
                     key=lambda kv: -kv[1])
    t_missing = table(
        ["Column", "Missing rows", "Share of all rows"],
        [(e(k), format(v, ","), "%.2f%%" % (100 * v / ex["rows"]))
         for k, v in missing], ["l", "r", "r"])
    t_corr = table(
        ["Rank", "Feature", "Correlation with Arrest", "Direction"],
        [(i + 1, e(pretty(n)), "%+.3f" % r,
          "more arrests" if r > 0 else "fewer arrests")
         for i, (n, r) in enumerate(top_corr)], ["r", "l", "r", "l"])
    t_scores = table(
        ["Score", "Linear regression", "Always predict the average", "Meaning"],
        [("RMSE", "%.3f" % sc["RMSE"], "%.3f" % base["RMSE"],
          "typical size of the error, on the 0&ndash;1 arrest scale"),
         ("MAE", "%.3f" % sc["MAE"], "%.3f" % base["MAE"],
          "average absolute error"),
         ("MSE", "%.3f" % sc["MSE"], "%.3f" % base["RMSE"] ** 2,
          "average squared error"),
         ("R&sup2;", "%.3f" % sc["R2"], "0.000",
          "share of the variation in Arrest that the model explains")],
        ["l", "r", "r", "l"])

    def coef_note(name):
        group, level = name.split("=", 1)
        t = counts.get(level) if group == "Primary Type" \
            else ex["by_area"].get(level)
        note = "%s records, %s arrested" % (
            format(t["count"], ","), pct(t["arrest_rate"]))
        if name == "Community Area=0":
            note += "; 0 is not a valid area number"
        return note

    t_coef_pos = table(
        ["Feature", "Coefficient", "In the raw data"],
        [(e(pretty(n)), "%+.3f" % c, coef_note(n)) for n, c in top_coef[:5]],
        ["l", "r", "l"])
    t_coef_neg = table(
        ["Feature", "Coefficient", "In the raw data"],
        [(e(pretty(n)), "%+.3f" % c, coef_note(n)) for n, c in top_coef[5:]],
        ["l", "r", "l"])

    steps = [
        ("1", "Load ChicagoCrimes2019.csv",
         "CSV File Import", "Reads the file; a type is set for each column."),
        ("2", "Raw Crime Data, Feature Statistics (Missing Values), "
              "Arrests by Crime Type",
         "Data Table, Feature Statistics, Distributions",
         "Explore the data and look for problems."),
        ("2b", "Keep Model Columns, Drop Rows with Missing Values",
         "Select Columns, Impute",
         "Keep the 3 features and the target; drop incomplete rows."),
        ("3", "Create Dummy Variables, Dummy-Coded Data",
         "Continuize, Data Table",
         "Turn the nominal columns into 0/1 dummy columns."),
        ("4", "Split 80% Train / 20% Test, Train Sample (80%), "
              "Test Sample (20%)",
         "Data Sampler, Data Table", "Random 80/20 split, fixed seed."),
        ("5", "Pearson Correlations vs Arrest, All Correlations, "
              "Pick Top 10 Correlations, Top Correlations",
         "Correlations, Data Table, Python Script, Data Table",
         "Correlation of every feature with Arrest; top 10 to its own table."),
        ("6", "Drop Baseline Dummy Columns, Linear Regression (Arrest), "
              "Test and Score (Test Sample)",
         "Select Columns, Linear Regression, Test and Score",
         "Fit on the training sample, score on the test sample."),
        ("7", "Model Coefficients, Pick Top 5 + / Top 5 - Coefficients, "
              "Top Coefficients",
         "Data Table, Python Script, Data Table",
         "All coefficients; top 5 positive and negative to their own table."),
    ]
    t_steps = table(["Step", "Widgets (as named on the canvas)",
                     "Orange widget type", "Purpose"],
                    [tuple(map(e, s)) for s in steps], ["l", "l", "l", "l"])

    narc = counts["NARCOTICS"]
    theft = counts["THEFT"]
    n_missing_cols = len(missing)
    base_type, base_area = [b.split("=", 1)[1] for b in pipe["baseline_levels"]]
    extra_pos_txt = ", ".join(
        "%s (%+.3f)" % (pretty(n), corr[n]) for n in extra_pos)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>MP1 - Chicago Crimes 2019</title>
<style>
@page {{ size: Letter; margin: 0.8in 0.85in; }}
body {{ font: 10.5pt/1.5 Helvetica, Arial, sans-serif; color: {INK}; }}
h1 {{ font-size: 21pt; line-height: 1.2; margin: 0 0 4pt; }}
h2 {{ font-size: 13.5pt; margin: 20pt 0 6pt; padding-top: 8pt;
      border-top: 1px solid {GRID}; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 12pt 0 3pt; break-after: avoid; }}
p {{ margin: 0 0 7pt; }}
.meta {{ color: {INK2}; margin-bottom: 14pt; }}
.lead {{ background: #f4f3ef; border-radius: 6px; padding: 10pt 13pt;
         margin: 10pt 0 4pt; }}
.lead p:last-child, .lead ul {{ margin-bottom: 0; }}
ul {{ margin: 0 0 7pt; padding-left: 16pt; }}
li {{ margin-bottom: 3pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 10pt;
         font-size: 9.5pt; break-inside: avoid; }}
th {{ text-align: left; color: {INK2}; font-weight: 600;
      border-bottom: 1.2px solid {MUTED}; padding: 4pt 6pt; }}
td {{ border-bottom: 1px solid {GRID}; padding: 4pt 6pt; vertical-align: top; }}
.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
figure {{ margin: 8pt 0 10pt; break-inside: avoid; }}
figure img {{ width: 100%; }}
figcaption {{ color: {INK2}; font-size: 9pt; margin-top: 3pt; }}
.shot {{ border: 1px solid {GRID}; border-radius: 4px; }}
.note {{ color: {INK2}; font-size: 9.5pt; }}
.pb {{ break-before: page; }}
</style></head><body>

<h1>What predicts an arrest? Chicago crimes, 2019</h1>
<div class="meta">MP1 &middot; Orange Canvas workflow report &middot; Saleh Yahya
&middot; workflow file: mp1_yahya.ows &middot; Orange {e(R["orange_version"])}</div>

<div class="lead">
<p><b>Main findings.</b> The type of crime is by far the strongest signal of
whether an arrest is made; the community area adds a little, and whether the
incident was domestic adds very little.</p>
<ul>
<li>A narcotics report is the feature most related to an arrest
(correlation {corr["Primary Type=NARCOTICS"]:+.3f}):
{round(narc["arrest_rate"] * narc["count"]):,} of {narc["count"]:,} narcotics
records ended in an arrest ({pct(narc["arrest_rate"])}), against
{100 * ex["arrest_rate"]:.1f}% of all records.</li>
<li>Theft, criminal damage and deceptive practice are the features most related
to <i>no</i> arrest; only {100 * theft["arrest_rate"]:.1f}% of {theft["count"]:,}
thefts led to one.</li>
<li>The linear regression explains {100 * sc["R2"]:.0f}% of the variation in Arrest
on the test sample (R&sup2; = {sc["R2"]:.3f}, RMSE = {sc["RMSE"]:.3f}). It is
useful but far from complete: three features cannot tell apart most cases
within the same crime type.</li>
</ul>
</div>

<h2>1. The workflow</h2>
<p>The workflow runs left to right in seven labelled steps. Every widget is
renamed for what it does, and text boxes with arrows mark each part.</p>
<figure><img class="shot" src="canvas.png" alt="Orange canvas workflow">
<figcaption>The Orange canvas of mp1_yahya.ows, drawn by Orange from the
saved workflow file.</figcaption></figure>
{t_steps}
<p class="note">Two <i>Python Script</i> widgets do the "pick the top rows"
steps (a few lines each, visible inside the widget), so the top-10 and top-5
tables are recomputed from the data every time the workflow runs rather than
depending on a hand-made selection.</p>

<h2>2. Exploring the data and its problems</h2>
<p>The file has {ex["rows"]:,} records and {ex["columns"]} attributes, all from
{int(ex["year_values"][0])}. {ex["arrest_true"]:,} records
({100 * ex["arrest_rate"]:.1f}%) have Arrest = TRUE and {ex["domestic_true"]:,}
({100 * ex["domestic_true"] / ex["rows"]:.1f}%) are domestic. The problems found:</p>
<ul>
<li><b>Missing values in {n_missing_cols} columns.</b> The location columns are
the worst affected; see the table below. Community Area, which the model
needs, is missing in {ex["missing"]["Community Area"]} rows. Those rows are
dropped in step 2b, leaving {pipe["clean_rows"]:,}.</li>
<li><b>Codes stored as numbers.</b> Community Area, Ward, District, Beat and
IUCR are identifiers, not quantities. Treating Community Area 50 as "twice"
area 25 would be meaningless, so it is loaded as a category and dummy-coded.</li>
<li><b>Column types are not detected reliably.</b> Left on automatic, Orange's
importer read Community Area and the four coordinate columns as text because
of their blank cells, so the type of every column is set explicitly in the
load widget.</li>
<li><b>An invalid area code.</b> Chicago's community areas are numbered 1 to 77,
but {ex["community_area_zero"]} records have Community Area 0. They are kept
(two rows cannot move the results) but the code is flagged in section 6.</li>
<li><b>Very rare categories.</b> {len(ex["rare_types"])} of the
{ex["primary_types"]} crime types have fewer than 50 records
({", ".join("%s: %d" % (t["type"].title(), t["count"]) for t in ex["rare_types"])}).
Anything estimated for them rests on a handful of cases.</li>
<li><b>Unbalanced target.</b> Only about one record in five is an arrest, so a
model that always says "no arrest" is already right
{100 * cls["majority_baseline"]:.1f}% of the time. Accuracy alone would flatter
any model.</li>
<li><b>Duplicate case numbers.</b> {ex["duplicate_case_numbers"]} case numbers
appear more than once, although every record ID is unique. A case can involve
several offences, so these were left in.</li>
<li><b>Columns with no information for this task.</b> Year is the same in every
row; ID, Case Number, Block and Location are near-unique identifiers; and
several columns repeat each other (the coordinates, Latitude/Longitude and
Location; IUCR, FBI Code and Description all refine Primary Type).</li>
</ul>
{t_missing}
<figure><img src="chart_arrest_rate.svg" alt="Arrest rate by crime type">
<figcaption>Arrest rate by crime type, for types with at least 1,000 records
(record count in brackets). The spread from 5% to 100% is why crime type
dominates everything that follows.</figcaption></figure>

<h2>3. Dummy variables and the train / test split</h2>
<p>The <i>Continuize</i> widget turns the four nominal attributes into numeric
dummy variables: one 0/1 column for each of the {ex["primary_types"]} crime
types and each of the {pipe["community_area_levels"]} community-area codes, and
a single 0/1 column each for Domestic and Arrest (1 = TRUE). That gives
{pipe["dummy_columns"]} feature columns plus the target.</p>
<p>The <i>Data Sampler</i> then splits the {pipe["clean_rows"]:,} rows at random
into {pipe["train_rows"]:,} training rows (80%) and {pipe["test_rows"]:,} test
rows (20%), with a fixed seed so the split is repeatable. The arrest rate is
{100 * pipe["train_arrest_rate"]:.1f}% in the training sample and
{100 * pipe["test_arrest_rate"]:.1f}% in the test sample, so the two are
comparable.</p>

<h2>4. Correlations with Arrest</h2>
<p>Pearson correlations between Arrest and each of the {R["correlations_total"]}
features were computed on the training sample. "Highest" is taken as strongest
in either direction, so strong negative correlations are included.</p>
{t_corr}
<figure><img src="chart_correlations.svg" alt="Top 10 correlations">
<figcaption>The ten strongest correlations with Arrest. Blue bars point to
more arrests, orange bars to fewer.</figcaption></figure>
<ul>
<li>Nine of the ten are crime types. Narcotics stands far above the rest;
weapons violations, interference with a public officer and criminal trespass
follow. These are offences usually recorded <i>because</i> an officer caught
someone, so an arrest is almost built into the report.</li>
<li>Theft, criminal damage and deceptive practice are negative: they are
typically reported after the fact, with no suspect present.</li>
<li>Only one community area makes the list: area 26 (West Garfield Park), at
{corr["Community Area=26"]:+.3f}.</li>
<li>Domestic is weak: {corr["Domestic=TRUE"]:+.3f}.
{100 * ex["arrest_rate_domestic"]:.1f}% of domestic records ended in an arrest
against {100 * ex["arrest_rate_not_domestic"]:.1f}% of the others.</li>
</ul>
<p class="note">Even the largest value, {top_corr[0][1]:+.3f}, is a moderate
correlation, and most are small. With over 200,000 rows they are all
statistically significant, but significance is not strength. If only positive
correlations were counted, the three negative entries would be replaced by
{extra_pos_txt}.</p>

<h2>5. Linear regression: test results</h2>
<p>A linear regression (ordinary least squares, no regularization) was trained
on the training sample with Arrest as the dependent variable, and scored by
<i>Test and Score</i> on the {pipe["test_rows"]:,} test rows it had not seen.</p>
{t_scores}
<p>The model beats the no-information baseline clearly (RMSE {sc["RMSE"]:.3f}
against {base["RMSE"]:.3f}) and explains about {100 * sc["R2"]:.0f}% of the
variation in Arrest. The other {100 - 100 * sc["R2"]:.0f}% is not explained by
crime type, community area and the domestic flag.</p>
<p><b>Read as a yes/no prediction.</b> Calling an arrest whenever the model's
output is 0.5 or more gives {100 * cls["accuracy"]:.1f}% accuracy on the test
sample, against {100 * cls["majority_baseline"]:.1f}% for always answering "no
arrest". When it predicts an arrest it is right {100 * cls["precision"]:.0f}%
of the time, but it finds only {100 * cls["recall"]:.0f}% of the actual arrests
({cls["predicted_arrests"]:,} predicted, {cls["actual_arrests"]:,} actual). In
practice the model recognises the crime types that almost always end in an
arrest and misses arrests for everyday types such as battery and assault.
<span class="note">(This paragraph was calculated from the same model and test
sample outside Orange; the table above is Orange's Test and Score output.)</span></p>
<p class="note">Two limits of the method: Arrest is a yes/no outcome, and a
linear regression is not bounded, so its predictions ran from
{cls["prediction_min"]:.2f} to {cls["prediction_max"]:.2f} rather than staying
between 0 and 1. A logistic regression would be the natural next model. Orange
also reports MAPE for regression; it is left out because it divides by the true
value, which is 0 for every non-arrest.</p>

<h2>6. Model coefficients</h2>
<p>With a dummy column for every crime type and every community area, each
group of columns adds up to 1 and duplicates the intercept, which leaves the
coefficients undefined (in a first run they came out in the hundreds of
millions). So one dummy per group, the most common level, is removed before
the model is fitted: <b>{e(base_type.title())}</b> and <b>community area
{e(base_area)}</b> ({AREA_NAMES.get(base_area, "")}). The model therefore has
{pipe["model_columns"]} features, and each coefficient is the change in the
predicted chance of arrest <i>compared with a non-domestic theft in area
{e(base_area)}</i>, whose predicted chance is the intercept,
{R["intercept"]:.3f}. Predictions and test scores are the same either way.</p>
<h3>Top 5 most positive</h3>
{t_coef_pos}
<h3>Top 5 most negative</h3>
{t_coef_neg}
<ul>
<li>The positive side is all crime types with arrest rates at or near 100%.
A coefficient of about +0.90 on top of the {R["intercept"]:.2f} intercept gives
a prediction close to 1.</li>
<li>The negative coefficients are much smaller, because the baseline (theft)
already has a low arrest rate, so there is little room below it.</li>
<li><b>Treat the smallest groups with caution.</b> Public Indecency
({counts["PUBLIC INDECENCY"]["count"]} records), Human Trafficking
({counts["HUMAN TRAFFICKING"]["count"]} records) and community area 0
({ex["community_area_zero"]} records, an invalid code) reach the top of the
lists on almost no data. The other entries rest on hundreds or thousands of
records each, as the right-hand column shows.</li>
<li>Domestic has a coefficient of {coefs["Domestic=TRUE"]:+.3f}: once the crime
type is known, a domestic incident is about
{abs(100 * coefs["Domestic=TRUE"]):.0f} percentage points less likely to end in
an arrest.</li>
</ul>
<p class="note">Correlations and coefficients answer different questions.
Narcotics has the highest correlation because it is both common and almost
always an arrest; Public Indecency has the highest coefficient but is too rare
to correlate with anything.</p>

<h2>7. Conclusions</h2>
<ul>
<li><b>What is most related to an arrest:</b> the type of crime. Offences that
are discovered by police action (narcotics, weapons, prostitution, gambling,
liquor, trespass, interference with an officer) nearly always end in an
arrest; offences reported afterwards by a victim (theft, criminal damage,
burglary, deceptive practice) rarely do.</li>
<li><b>Community area matters a little,</b> with West Garfield Park the one
area in the top ten, and <b>the domestic flag matters least.</b></li>
<li><b>Can arrests be predicted from these three features?</b> Partly. The
model explains {100 * sc["R2"]:.0f}% of the variation and is
{100 * cls["accuracy"]:.1f}% accurate as a yes/no rule, a real gain over the
{100 * cls["majority_baseline"]:.1f}% baseline, but it misses about half of the
actual arrests.</li>
<li><b>These are associations, not causes.</b> A high arrest rate for narcotics
reflects how those crimes come to be recorded, not that the category itself
makes an arrest more likely.</li>
</ul>
</body></html>"""

    html_path = os.path.join(RESULTS, "report.html")
    with open(html_path, "w") as f:
        f.write(doc)
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--print-to-pdf=" + PDF, "file://" + html_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=180)
    print("wrote", PDF)


if __name__ == "__main__":
    sys.exit(main())
