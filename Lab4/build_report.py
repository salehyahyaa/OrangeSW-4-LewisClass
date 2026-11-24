"""Build Lab4/submission/lab4_yahya.pdf from what the workflow produced.

Reads Lab4/results/ (written by run_workflow.py), writes report.html there
and prints it to PDF with headless Chrome.

    /Applications/Orange.app/Contents/MacOS/python Lab4/build_report.py
"""
import html
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PDF = os.path.join(HERE, "submission", "lab4_yahya.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

LOSSES = ["Linear", "Square", "Exponential"]
ACTIVATIONS = ["Identity", "Logistic", "tanh", "ReLu"]
SOLVERS = ["L-BFGS-B", "SGD", "Adam"]
METRICS = ["Euclidean", "Manhattan", "Chebyshev", "Mahalanobis"]
WEIGHTS = ["Uniform", "By Distances"]


class Numbering:
    """Hands out 'Figure N' / 'Table N' numbers in order."""

    def __init__(self):
        self.fig = self.tab = 0

    def figure(self, image, caption, width=100):
        self.fig += 1
        return ('<figure><img class="shot" style="width:%d%%" src="%s.png" '
                'alt=""><figcaption><b>Figure %d.</b> %s</figcaption></figure>'
                % (width, image, self.fig, caption))

    def table(self, caption, headers, rows, align=None):
        self.tab += 1
        align = align or ["l"] * len(headers)
        out = ['<table><caption><b>Table %d.</b> %s</caption><thead><tr>'
               % (self.tab, caption)]
        out += ['<th class="%s">%s</th>' % (a, h)
                for h, a in zip(headers, align)]
        out.append("</tr></thead><tbody>")
        for row in rows:
            out.append("<tr>" + "".join(
                '<td class="%s">%s</td>' % (a, c)
                for c, a in zip(row, align)) + "</tr>")
        out.append("</tbody></table>")
        return "".join(out)


def score_table(N, caption, rows):
    return N.table(
        caption, ["Model", "MSE", "RMSE", "MAE", "R2"],
        [(r["Model"], "%.4f" % r["MSE"], "%.4f" % r["RMSE"],
          "%.4f" % r["MAE"], "%.3f" % r["R2"]) for r in rows],
        ["l", "r", "r", "r", "r"])


def describe_settings(model, s):
    """A model's settings the way they read in its Orange widget."""
    if model == "AdaBoost":
        return "%d estimators, learning rate %s, %s loss" % (
            s["n_estimators"], s["learning_rate"],
            LOSSES[s["loss_index"]].lower())
    if model == "Random Forest":
        parts = ["%d trees" % s["n_estimators"]]
        parts.append("%d attributes per split" % s["max_features"]
                     if s.get("use_max_features")
                     else "default attributes per split")
        parts.append("no subsets smaller than %d split"
                     % s["min_samples_split"]
                     if s.get("use_min_samples_split")
                     else "no limit on splitting small subsets")
        return ", ".join(parts)
    if model == "Neural Network":
        from Orange.widgets.model.owneuralnetwork import OWNNLearner
        return "hidden layers %s, %s activation, %s solver, " \
               "regularization %s, %d iterations" % (
                   s["hidden_layers_input"].rstrip(","),
                   ACTIVATIONS[s["activation_index"]],
                   SOLVERS[s["solver_index"]],
                   OWNNLearner.alphas[s["alpha_index"]], s["max_iterations"])
    return "%d neighbors, %s metric, %s weight" % (
        s["n_neighbors"], METRICS[s["metric_index"]],
        WEIGHTS[s["weight_index"]].lower())


def short(name, limit=62):
    return html.escape(name if len(name) <= limit else name[:limit - 1] + "…")


def main():
    with open(os.path.join(RESULTS, "results.json")) as f:
        R = json.load(f)
    with open(os.path.join(RESULTS, "dataset_check.json")) as f:
        D = json.load(f)
    N = Numbering()
    ex, s2, s5 = R["explore"], R["step2"], R["step5"]
    me = s5["missing_effect"]
    cv = {r["Model"]: r for r in R["cv_scores"]}
    ts = {r["Model"]: r for r in R["test_scores"]}
    co = R["coefficients"]
    ols, lasso, en = (co[k] for k in (
        "No Regularization", "Lasso Regression (L1)",
        "Elastic Net Regression"))
    en1 = co["Elastic Net Regression (1)"]
    other = R["other_cv_scores"]
    other_test = {r["Model"]: r for r in R["other_test_scores"]}
    tuning, tuned = R["tuning"], R["tuned_settings"]
    best_linear = R["cv_scores"][0]
    best_other, worst_other = other[0], other[-1]
    three = [r for r in R["cv_scores"]
             if r["Model"] != "Elastic Net Regression (1)"]
    one = s5["one_feature_cv"]
    e = html.escape

    # ---------------- figures and tables, in order ----------------
    f_datasets = N.figure("datasets", "Datasets widget with the HDI dataset "
                          "selected.")
    t_check = N.table(
        "The three checks I did to be sure this is the right dataset.",
        ["Check", "What I looked at", "Result"],
        [("1. The catalogue",
          "The entry in the Datasets widget list",
          "Title is HDI, and its description starts with \"The Human "
          "Development Index (HDI)\". Source: UNDP Human Development "
          "Reports, year %s." % D["check1_catalogue"]["year"]),
         ("2. What the widget loads",
          "The data coming out of the Datasets widget",
          "File HDI.tab with %d rows, matching the %d instances listed, and "
          "it has a column named HDI."
          % (D["check2_widget_output"]["rows"],
             D["check1_catalogue"]["instances"])),
         ("3. The values",
          "HDI values in the data against the published UNDP 2015 figures",
          "All %d countries I checked match, for example Norway %.3f and "
          "Niger %.3f." % (len(D["check3_values"]["compared_with_undp"]),
                           D["check3_values"]["compared_with_undp"]
                           ["Norway"]["orange"],
                           D["check3_values"]["compared_with_undp"]
                           ["Niger"]["orange"]))])
    f_table = N.figure("data_table", "Data Table with the HDI data as "
                       "loaded.")
    f_dist = N.figure("explore_dist", "Distributions of HDI.", 85)
    f_box = N.figure("explore_box", "Box Plot of HDI.", 85)
    f_select = N.figure("select_columns", "Select Columns with HDI moved to "
                        "Target.", 72)
    f_pre = N.figure("preprocess", "Preprocess widget set to Normalize "
                     "Features, normalize to interval [0, 1].", 72)
    f_sampler = N.figure("data_sampler", "Data Sampler set to a fixed "
                         "proportion of 50%.", 34)
    f_train = N.figure("training_set", "Training set (94 countries).")
    f_test = N.figure("testing_set", "Testing set (the other 94 countries).")
    f_m1 = N.figure("model_no_regularization", "No Regularization.", 46)
    f_m2 = N.figure("model_lasso_regression_l1", "Lasso Regression (L1) "
                    "with the regularization strength at the lowest value "
                    "(0.0001).", 46)
    f_m3 = N.figure("model_elastic_net_regression", "Elastic Net Regression "
                    "with regularization strength 0.01 and mixing "
                    "0.24 : 0.76.", 46)
    f_cv = N.figure("test_and_score_cv", "Test and Score with 10-fold "
                    "cross-validation, ordered from smallest to largest "
                    "RMSE. The Elastic Net Regression (1) row is the model "
                    "added in step 5.")
    t_cv = score_table(N, "Cross-validation scores of the three linear "
                       "regression models (10 folds, training set).", three)
    f_c1 = N.figure("coef_no_regularization", "Coefficients of No "
                    "Regularization, sorted from largest to smallest.", 80)
    f_c2 = N.figure("coef_lasso_regression_l1", "Coefficients of Lasso "
                    "Regression (L1), sorted from largest to smallest.", 80)
    f_c3 = N.figure("coef_elastic_net_regression", "Coefficients of Elastic "
                    "Net Regression, sorted from largest to smallest.", 80)
    t_shape = N.table(
        "How the three sets of coefficients compare.",
        ["Model", "Features kept (not zero)", "Features set to zero",
         "Largest coefficient (absolute)", "Intercept"],
        [(name, c["n_nonzero"], c["n_zero"], "%.3f" % c["largest_abs"],
          "%.3f" % c["intercept"])
         for name, c in (("No Regularization", ols),
                         ("Lasso Regression (L1)", lasso),
                         ("Elastic Net Regression", en))],
        ["l", "r", "r", "r", "r"])

    def top_rows(c, key):
        return [(short(d["name"]), "%+.3f" % d["coef"]) for d in c[key][:3]]

    t_top = N.table(
        "The three highest positive and three most negative coefficients in "
        "each model.",
        ["Model", "Highest positive", "Coef.", "Most negative", "Coef."],
        [(name if i == 0 else "", p[0], p[1], n[0], n[1])
         for name, c in (("No Regularization", ols),
                         ("Lasso Regression (L1)", lasso),
                         ("Elastic Net Regression", en))
         for i, (p, n) in enumerate(zip(top_rows(c, "top_positive"),
                                        top_rows(c, "top_negative")))],
        ["l", "l", "r", "l", "r"])
    f_rank = N.figure("rank", "Rank widget scoring by RReliefF, with the "
                      "best ranked 1 feature selected.", 80)
    f_reduced = N.figure("reduced_data", "The reduced data coming out of "
                         "Rank: one feature plus the HDI target.", 80)
    t_cv4 = score_table(N, "Cross-validation scores with the duplicated "
                        "Elastic Net model added.", R["cv_scores"])
    f_c4 = N.figure("coef_elastic_net_regression_1", "Coefficients of "
                    "Elastic Net Regression (1), the model trained on the "
                    "one top feature.", 72)
    t_missing = N.table(
        "The top RReliefF features and how many of their values are missing "
        "in the %d training rows." % me["training_rows"],
        ["Rank", "Feature", "RReliefF", "Missing values"],
        [(i + 1, short(d["name"], 75), "%.3f" % d["score"],
          d["missing_in_training"]) for i, d in enumerate(me["top8"][:6])],
        ["r", "l", "r", "r"])
    f_test_scores = N.figure("test_and_score_test", "Test and Score set to "
                             "Test on test data.")
    t_compare = N.table(
        "Cross-validation against test on test data.",
        ["Model", "RMSE (cross-validation)", "RMSE (test data)",
         "R2 (cross-validation)", "R2 (test data)"],
        [(r["Model"], "%.4f" % r["RMSE"], "%.4f" % ts[r["Model"]]["RMSE"],
          "%.3f" % r["R2"], "%.3f" % ts[r["Model"]]["R2"])
         for r in R["cv_scores"]], ["l", "r", "r", "r", "r"])
    model_figs = "".join(
        N.figure("model_" + m.lower().replace(" ", "_"),
                 "%s widget with the settings I ended up with." % m, 44)
        for m in ("AdaBoost", "Random Forest", "Neural Network", "kNN"))
    t_tune = N.table(
        "What I tried for each model and the settings with the lowest "
        "cross-validated RMSE.",
        ["Model", "Settings tried", "RMSE with default settings",
         "Best RMSE", "Best settings"],
        [(m, tuning[m]["tried"], "%.4f" % tuning[m]["default_rmse"],
          "%.4f" % tuning[m]["best"]["rmse"],
          e(describe_settings(m, tuned[m])))
         for m in ("AdaBoost", "Random Forest", "Neural Network", "kNN")],
        ["l", "r", "r", "r", "l"])
    f_other = N.figure("test_and_score_1_cv", "Test and Score (1): 10-fold "
                       "cross-validation of the four models, ordered by "
                       "RMSE.")
    everything = sorted(
        [dict(r, kind="Linear regression") for r in three]
        + [dict(r, kind="Step 7") for r in other], key=lambda r: r["RMSE"])
    t_all = N.table(
        "All seven models ranked by cross-validated RMSE.",
        ["Rank", "Model", "Type", "RMSE", "R2"],
        [(i + 1, r["Model"], r["kind"], "%.4f" % r["RMSE"], "%.3f" % r["R2"])
         for i, r in enumerate(everything)], ["r", "l", "l", "r", "r"])
    f_canvas = N.figure("canvas", "The finished Orange Canvas workflow.")

    better = [r["Model"] for r in other if r["RMSE"] < best_linear["RMSE"]]
    worse = [r["Model"] for r in other if r["RMSE"] >= best_linear["RMSE"]]
    filled_top = me["top5_after_filling_gaps"][0]["name"]
    gap = [abs(ts[m]["RMSE"] - cv[m]["RMSE"]) for m in cv]
    lower_on_test = all(ts[m]["RMSE"] < cv[m]["RMSE"] for m in cv)
    most = ex["most_missing"]

    def join(names):
        names = list(names)
        return names[0] if len(names) == 1 else \
            ", ".join(names[:-1]) + " and " + names[-1]

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Lab 4 - Regression Analysis</title>
<style>
@page {{ size: Letter; margin: 0.8in 0.85in; }}
body {{ font: 10.5pt/1.5 Helvetica, Arial, sans-serif; color: {INK}; }}
h1 {{ font-size: 20pt; line-height: 1.2; margin: 0 0 4pt; }}
h2 {{ font-size: 13.5pt; margin: 20pt 0 6pt; padding-top: 8pt;
      border-top: 1px solid {GRID}; break-after: avoid; }}
p {{ margin: 0 0 7pt; orphans: 3; widows: 3; }}
.meta {{ color: {INK2}; margin-bottom: 12pt; }}
ul {{ margin: 0 0 7pt; padding-left: 16pt; }}
li {{ margin-bottom: 3pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 6pt 0 12pt;
         font-size: 9.5pt; break-inside: avoid; }}
caption {{ caption-side: top; text-align: left; color: {INK2};
           font-size: 9pt; padding-bottom: 3pt; }}
th {{ text-align: left; color: {INK2}; font-weight: 600;
      border-bottom: 1.2px solid {MUTED}; padding: 4pt 6pt; }}
td {{ border-bottom: 1px solid {GRID}; padding: 4pt 6pt; vertical-align: top; }}
.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
figure {{ margin: 8pt 0 12pt; break-inside: avoid; text-align: center; }}
figcaption {{ color: {INK2}; font-size: 9pt; margin-top: 3pt;
              text-align: left; }}
.shot {{ border: 1px solid {GRID}; border-radius: 4px; }}
.q {{ background: #f4f3ef; border-radius: 6px; padding: 8pt 12pt;
      margin: 8pt 0; break-inside: avoid; }}
.q p:last-child, .q ul:last-child {{ margin-bottom: 0; }}
.row {{ display: flex; gap: 10pt; align-items: flex-start; }}
.row figure {{ flex: 1; }}
.row .shot {{ width: 100% !important; }}
</style></head><body>

<h1>Lab Exercise #4: Regression Analysis</h1>
<div class="meta">DATA-20000 Introduction to Data Science &middot; Saleh Yahya
&middot; workflow file: lab4_yahya.ows</div>

<h2>Step 1: Loading and describing the HDI data</h2>
<p>I used the Datasets widget and picked the dataset called HDI (Human
Development Index).</p>
{f_datasets}
<p>The Datasets list has more than one entry about human development (there
is also an HDI-edu version and a Slovenian one), so I checked that I had the
right one in three ways.</p>
{t_check}
{f_table}
<p><b>Describing the data.</b> Each row is a country, and there are
{ex["rows"]} of them. The HDI column is the Human Development Index for 2015,
a number between 0 and 1 that the UN uses to summarize how developed a
country is, based on health, education and income. Besides the country name
and HDI there are {ex["features"]} features, all numeric. They cover things
like life expectancy, years of schooling, income per person, mortality
rates, population and age structure, employment and unemployment, health
spending, internet use and inequality.</p>
<p>HDI goes from {ex["hdi"]["min"]:.3f}
({e(ex["bottom5"][0][0])}) to {ex["hdi"]["max"]:.3f}
({e(ex["top5"][0][0])}), with a mean of {ex["hdi"]["mean"]:.3f} and a median
of {ex["hdi"]["median"]:.3f}. The top five are
{join(e(c) for c, _ in ex["top5"])}, and the bottom five are
{join(e(c) for c, _ in ex["bottom5"])}.</p>
{f_dist}
{f_box}
<p>The distribution is not a bell shape. It leans to the left, with most
countries between about 0.65 and 0.9 and a tail of low values going down to
0.35.</p>
<p>The main problem with the data is missing values. About
{ex["missing_percent"]:.0f}% of all the cells are empty, and
{ex["columns_with_missing"]} of the {ex["features"]} features have at least
one missing value. The worst ones are "{e(most[0][0])}" (missing for
{most[0][1]} countries), "{e(most[1][0])}" ({most[1][1]}) and
"{e(most[2][0])}" ({most[2][1]}). This ends up mattering in step 5.</p>
<p>The target was not defined. HDI came in as a meta column, next to the
country name, so I used Select Columns to move HDI to Target.</p>
{f_select}

<h2>Step 2: Normalizing and splitting the data</h2>
<p>I connected Select Columns to a Preprocess widget and added Normalize
Features, set to normalize to the interval [0, 1]. After this every feature
runs from exactly 0 to 1. HDI itself is not changed, since it is the target
and not a feature.</p>
{f_pre}
<p>Then I connected Preprocess to a Data Sampler set to a fixed proportion
of 50%, and sent the Data Sample to a table I renamed Training set and the
Remaining Data to a table I renamed Testing set. Each one has
{s2["train_rows"]} countries and no country is in both. In each table I
selected all the rows, so that its Selected Data output passes the whole set
on to the models, the way the sample workflow is wired.</p>
{f_sampler}
{f_train}
{f_test}

<h2>Step 3: Three linear regression models</h2>
<p>I added three Linear Regression widgets, connected the training set to
each one, and renamed them after their settings.</p>
<div class="row">{f_m1}{f_m2}{f_m3}</div>
<p>I connected the three learners and the training set to Test and Score,
selected cross-validation with 10 folds, and clicked the RMSE header to
order the models from smallest to largest RMSE.</p>
{f_cv}
{t_cv}
<p><b>Describing the results.</b> All three models predict HDI well. RMSE is
in the same units as HDI, so an RMSE of
{cv["Lasso Regression (L1)"]["RMSE"]:.3f} means the Lasso model is typically
off by about {cv["Lasso Regression (L1)"]["RMSE"]:.2f} on a scale that runs
from 0 to 1. Its R2 of {cv["Lasso Regression (L1)"]["R2"]:.2f} means it
explains about {cv["Lasso Regression (L1)"]["R2"] * 100:.0f}% of the
differences in HDI between countries.</p>
<p>The order is Lasso first (RMSE
{cv["Lasso Regression (L1)"]["RMSE"]:.4f}), then Elastic Net
({cv["Elastic Net Regression"]["RMSE"]:.4f}), and No Regularization last
({cv["No Regularization"]["RMSE"]:.4f}). The plain model is clearly the
worst, with an error about
{(cv["No Regularization"]["RMSE"] / cv["Lasso Regression (L1)"]["RMSE"] - 1) * 100:.0f}%
higher than Lasso. I think that is because it has {ex["features"]} features
to fit and only {s2["train_rows"]} countries to learn from, and many of the
features measure nearly the same thing. With nothing holding it back it fits
the training countries too closely and does worse on the fold it has not
seen. The two regularized models are forced to keep their coefficients
small, which makes them more stable. Lasso did better than Elastic Net here
even though its regularization is much weaker (0.0001 against 0.01), so on
this data a light touch was enough.</p>

<h2>Step 4: Coefficients</h2>
<p>I connected the Coefficients output of each model to its own Data Table
and clicked the coef column header to sort from the largest coefficient to the smallest. Each
picture shows the top of the table, and the most negative ones are listed in
the table after the pictures. The features were
normalized to [0, 1] in step 2, so the coefficients can be compared with
each other: a coefficient is how much the predicted HDI changes when that
feature goes from its lowest value to its highest.</p>
{f_c1}
{f_c2}
{f_c3}
{t_shape}
{t_top}
<p><b>Comparing the three.</b></p>
<ul>
<li><b>No Regularization</b> uses all {ols["n_features"]} features and has
the largest coefficients, up to {ols["largest_abs"]:.2f}. Some of them make
no sense on their own. Under-five mortality has a large positive
coefficient and infant mortality has a large negative one, when both should
push HDI down. Those two features are almost copies of each other, so the
model can give them big opposite values that mostly cancel out. It is a sign
of overfitting.</li>
<li><b>Lasso</b> set {lasso["n_zero"]} of the coefficients to exactly zero
and kept {lasso["n_nonzero"]}. The ones it kept are smaller and easier to
read.</li>
<li><b>Elastic Net</b>, with the stronger regularization, went further. It
kept only {en["n_nonzero"]} features and its largest coefficient is
{en["largest_abs"]:.2f}. Its intercept is higher ({en["intercept"]:.2f})
because less of the prediction comes from the features.</li>
</ul>
<div class="q">
<p><b>Highest positive and negative coefficients.</b> The highest positive
coefficient in all three models is {e(ols["top_positive"][0]["name"])}
(+{ols["top_positive"][0]["coef"]:.2f}, +{lasso["top_positive"][0]["coef"]:.2f}
and +{en["top_positive"][0]["coef"]:.2f}). Life expectancy and income per
person (GNI per capita, or the estimated income columns) are also near the
top each time. The most negative coefficients change from model to model:
{e(ols["top_negative"][0]["name"])} in the plain model
({ols["top_negative"][0]["coef"]:.2f}),
{e(lasso["top_negative"][0]["name"])} in Lasso
({lasso["top_negative"][0]["coef"]:.2f}) and
{e(en["top_negative"][0]["name"])} in Elastic Net
({en["top_negative"][0]["coef"]:.2f}).</p>
<p><b>Which variables are most important in the model?</b> Mean years of
schooling, life expectancy and income per person. They have the biggest
positive coefficients, and they are the ones that stay large in every model
even when regularization removes most of the others.</p>
<p><b>Does this make sense?</b> Yes, completely. HDI is built from exactly
three things: a long and healthy life (life expectancy), knowledge (years of
schooling) and a decent standard of living (income per person). So the model
has found the ingredients that HDI is calculated from. The negative ones in
the regularized models also make sense: more inequality in life expectancy
and a higher share of young dependents both go with less developed
countries. The negative coefficients in the plain model make less sense, for
the reason above, so I would trust the regularized models more when reading
coefficients.</p>
</div>

<h2>Step 5: Rank and a one-feature Elastic Net</h2>
<p>I duplicated the Elastic Net Regression widget (same settings: 0.01 and
0.24 : 0.76), connected the training set to a Rank widget, ticked RReliefF
as the scoring method and set it to select the best ranked 1 feature. The
feature it picked is <b>{e(s5["top_feature"])}</b>.</p>
{f_rank}
{f_reduced}
<p>I connected the Rank output to the new Elastic Net widget, and sent its
learner to Test and Score and its coefficients to a new Data Table.</p>
{t_cv4}
{f_c4}
<div class="q">
<p><b>Comparing the scores.</b> In Test and Score, Elastic Net Regression
(1) has exactly the same scores as Elastic Net Regression (RMSE
{cv["Elastic Net Regression (1)"]["RMSE"]:.4f}, R2
{cv["Elastic Net Regression (1)"]["R2"]:.3f}). At first I expected it to be
worse because it only gets one feature. The reason it is not is how Orange
works: what goes from the model widget into Test and Score is the learner,
meaning the method and its settings, not a model that has already been
trained. Test and Score trains it again itself, on its own data, which is
the full training set with all {ex["features"]} features. The two widgets
have the same settings, so they get the same score.</p>
<p>The one-feature model does show up in the coefficients table, since that
comes from the model the widget trained on the data from Rank. It has one
feature, and its coefficient is 0. So the model
ignores the feature and predicts the intercept, {en1["intercept"]:.3f}, for
every country, which is just the average HDI of the training set. To see how
a model like that would really score, I ran the same 10-fold
cross-validation on the reduced data as an extra check (this is not part of
the saved workflow). The RMSE is {one["RMSE"]:.3f} and R2 is
{one["R2"]:.2f}, about
{one["RMSE"] / cv["Elastic Net Regression"]["RMSE"]:.0f} times worse than
with all the features.</p>
<p><b>Why this feature?</b> I did not expect HIV prevalence to be the single
best feature for predicting HDI, so I looked at the Rank table more closely.
The features at the top are the ones with the most missing values.</p>
{t_missing}
<p>{e(s5["top_feature"])} is missing for
{s5["top_feature_missing_in_training"]} of the {me["training_rows"]}
training countries. Across all the features, the RReliefF score and the
number of missing values have a correlation of
{me["corr_score_vs_missing"]:.2f}. So it looks like RReliefF is being thrown
off by the missing values instead of measuring how useful a feature is. As a
check, when the gaps are filled in with the column average first, RReliefF
puts "{e(filled_top)}" at the top instead. My takeaway is that picking one
feature this way did not work on this data, and that a feature selection
score needs to be checked against missing values before trusting it.</p>
</div>

<h2>Step 6: Test on test data</h2>
<p>I connected the Testing set to the Test Data input of Test and Score and
changed the setting to Test on test data. Now each model is trained on the
whole training set ({s2["train_rows"]} countries) and scored on the
{s2["test_rows"]} countries it has never seen.</p>
{f_test_scores}
{t_compare}
<div class="q">
<p><b>Comparing to cross-validation.</b> The results agree. The order of the
models is the same: Lasso is best, then Elastic Net, then No
Regularization. The numbers are close too, with RMSE changing by
{min(gap):.3f} to {max(gap):.3f}.
{"Every model actually did a little better on the test data than in cross-validation." if lower_on_test else "Some models did slightly better on the test data and some slightly worse."}
One reason is that in cross-validation each model learns from 9 of the 10
folds, about {round(s2["train_rows"] * 0.9)} countries, and here it learns
from all {s2["train_rows"]}. The rest is just that these are two different
sets of countries.</p>
<p>What this tells me is that the cross-validation scores were an honest
estimate. The models were not just memorizing the training countries,
because they hold up on completely new ones. It also confirms that
regularization helps on this data: the plain model is still the weakest on
the test set.</p>
</div>

<h2>Step 7: AdaBoost, Random Forest, Neural Network and kNN</h2>
<p>I added the four models and connected them to a second Test and Score
with 10-fold cross-validation on the training set (through Training set (1),
as in the sample workflow). For each model I tried different values for its
settings and kept the combination with the lowest RMSE.</p>
{t_tune}
<div class="row">{model_figs}</div>
{f_other}
{t_all}
<div class="q">
<p><b>Describing the results.</b></p>
<ul>
<li><b>{best_other["Model"]}</b> came out best of the four, with an RMSE of
{best_other["RMSE"]:.4f} and R2 of {best_other["R2"]:.3f}.
{"It beats" if best_other["RMSE"] < best_linear["RMSE"] else "It does not beat"}
the best linear model from step 3 (Lasso, {best_linear["RMSE"]:.4f}).</li>
<li>Compared with Lasso,
{(join(better) + (" are" if len(better) > 1 else " is") + " better") if better else "none of the four is better"}{" and " + join(worse) + (" are" if len(worse) > 1 else " is") + " worse" if worse and better else ""}{"" if better else ", so the simple linear model is hard to beat here"}.
The differences at the top are small though, a few thousandths of an HDI
point.</li>
<li><b>Tuning mattered most for the Neural Network.</b> With its default
settings it had an RMSE of {tuning["Neural Network"]["default_rmse"]:.3f},
far worse than everything else, and with the settings in the table it comes
down to {tuning["Neural Network"]["best"]["rmse"]:.4f}. For the other three
the settings made a smaller difference: AdaBoost went from
{tuning["AdaBoost"]["default_rmse"]:.4f} to
{tuning["AdaBoost"]["best"]["rmse"]:.4f}, Random Forest from
{tuning["Random Forest"]["default_rmse"]:.4f} to
{tuning["Random Forest"]["best"]["rmse"]:.4f}, and kNN from
{tuning["kNN"]["default_rmse"]:.4f} to
{tuning["kNN"]["best"]["rmse"]:.4f}.</li>
<li><b>{worst_other["Model"]}</b> is the weakest of the four even after
tuning ({worst_other["RMSE"]:.4f}).</li>
</ul>
<p>Overall, the more complicated models did not buy much. That fits with
step 4: HDI is basically a formula made from a few of these features, so a
linear model with a little regularization already captures most of it, and
with only {s2["train_rows"]} training countries there is not much data for
a flexible model to learn anything extra. One caution is that I picked these
settings using the same cross-validation scores I am reporting, so they are
probably a little optimistic. As a check, on the test set the four models
score {join("%s %.4f" % (r["Model"], other_test[r["Model"]]["RMSE"]) for r in other)}.</p>
</div>

<h2>Step 8: The workflow</h2>
<p>Every part of the workflow is labelled with a text box and an arrow, and
the widgets are renamed to say what they are.</p>
{f_canvas}
</body></html>"""

    html_path = os.path.join(RESULTS, "report.html")
    with open(html_path, "w") as f:
        f.write(doc)
    subprocess.run(
        [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
         "--print-to-pdf=" + PDF, "file://" + html_path],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        timeout=180)
    print("wrote", PDF, "figures:", N.fig, "tables:", N.tab)


if __name__ == "__main__":
    sys.exit(main())
