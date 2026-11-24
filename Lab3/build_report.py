"""Build Lab3/submission/lab3_yahya.pdf from what the workflow produced.

Reads Lab3/results/ (written by run_workflow.py), writes report.html there
and prints it to PDF with headless Chrome.

    /Applications/Orange.app/Contents/MacOS/python Lab3/build_report.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PDF = os.path.join(HERE, "submission", "lab3_yahya.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"

MEANING = {
    "CRIM": "crime rate", "ZN": "large-lot zoning", "INDUS": "industry",
    "CHAS": "on the Charles River", "NOX": "air pollution",
    "RM": "rooms per home", "AGE": "old homes", "DIS": "distance to jobs",
    "RAD": "highway access", "TAX": "property tax rate",
    "PTRATIO": "pupil-teacher ratio", "B": "B index",
    "LSTAT": "% lower status", "MEDV": "median home value",
}


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


def p_text(p):
    return "&lt; 0.001" if p < 0.001 else "%.3f" % p


def main():
    with open(os.path.join(RESULTS, "results.json")) as f:
        R = json.load(f)
    N = Numbering()
    ex = R["explore"]
    r1, r2 = R["pearson_with_medv"], R["pearson_with_medv_after_formula"]
    sp = R["spearman_with_medv"]
    s7, s8 = R["step7"], R["step8"]
    low, high = s7["labels"]
    g_low, g_high = s7["groups"][low], s7["groups"][high]
    rm_low, rm_high = s8["labels"]
    tests = s8["tests"]
    sig = [d for d in tests if d["p_widget"] < 0.05]
    not_sig = [d for d in tests if d["p_widget"] >= 0.05]
    lost = [n for n in s8["significant"]
            if n not in s8["significant_bonferroni"]]
    by_name = {d["attribute"]: d for d in tests}

    # ---------------- figures and tables, in order ----------------
    f_table = N.figure("data_table", "Data Table with the Housing dataset "
                       "(506 rows, MEDV is the target).")
    f_ebox = N.figure("explore_box", "Box Plot of MEDV.", 85)
    f_edist = N.figure("explore_dist", "Distributions of MEDV.", 85)
    f_select = N.figure("select_columns", "Select Columns with MEDV moved "
                        "from Target to Features, so all 14 columns are "
                        "features.", 72)
    f_corr = N.figure("correlations", "Correlations widget with MEDV typed "
                      "in the filter box. The highest positive pair, MEDV "
                      "and RM, is selected.", 48)
    ordered = sorted(r1.items(), key=lambda kv: -kv[1]["r"])
    t_corr = N.table(
        "Pearson correlations between MEDV and every other attribute, "
        "highest to lowest.",
        ["Attribute", "What it is", "Correlation with MEDV"],
        [(n, MEANING[n], "%+.3f" % v["r"]) for n, v in ordered],
        ["l", "l", "r"])
    f_srm = N.figure("scatter_rm_medv", "Scatter Plot of the selected pair, "
                     "MEDV (x) and RM (y), with the regression line.", 88)
    f_slstat = N.figure("scatter_lstat_medv", "Scatter Plot of LSTAT (x) "
                        "and MEDV (y), the strongest pair overall.", 88)
    f_formula = N.figure("formula", "Formula widget with the new log2 "
                         "columns.", 66)
    f_corr2 = N.figure("correlations_log2", "Correlations after the Formula "
                       "widget, filtered on MEDV. The most correlated pair, "
                       "MEDV and log2_LSTAT, is selected.", 48)
    f_slog = N.figure("scatter_log2", "Scatter Plot of the selected pair, "
                      "MEDV (x) and log2_LSTAT (y).", 88)
    t_log = N.table(
        "Correlation with MEDV before and after taking log2.",
        ["Feature", "Original", "log2 of it", "Change"],
        [(n, "%+.3f" % r1[n]["r"], "%+.3f" % r2["log2_" + n]["r"],
          "stronger" if abs(r2["log2_" + n]["r"]) > abs(r1[n]["r"])
          else "weaker") for n in ("LSTAT", "RM")],
        ["l", "r", "r", "l"])
    f_dcrim = N.figure("discretize_crim", "Discretize widget. Only CRIM is "
                       "set to equal frequency with 2 intervals, everything "
                       "else is kept numeric.", 72)
    f_dist = N.figure("dist_medv_by_crim", "Distributions of MEDV split by "
                      "the two CRIM groups.", 88)
    t_crim = N.table(
        "MEDV in the two CRIM groups.",
        ["CRIM group", "Towns", "Mean MEDV", "Median MEDV", "Std. dev."],
        [("%s (low crime)" % low, g_low["n"], "%.2f" % g_low["mean"],
          "%.1f" % g_low["median"], "%.2f" % g_low["sd"]),
         ("%s (high crime)" % high, g_high["n"], "%.2f" % g_high["mean"],
          "%.1f" % g_high["median"], "%.2f" % g_high["sd"])],
        ["l", "r", "r", "r", "r"])
    f_box = N.figure("box_medv_by_crim", "Box Plot of MEDV grouped by CRIM, "
                     "comparing means. The t statistic is printed under the "
                     "plot.", 88)
    f_drm = N.figure("discretize_rm", "Second Discretize widget. Only RM is "
                     "set to equal frequency with 2 intervals.", 72)
    top = tests[0]["attribute"]
    f_rm_top = N.figure("box_rm_" + top, "Box Plot (by RM) for %s, the "
                        "attribute with the largest t." % top, 88)
    f_rm_medv = N.figure("box_rm_MEDV", "Box Plot (by RM) for MEDV.", 88)
    weakest = not_sig[-1]["attribute"]
    f_rm_weak = N.figure("box_rm_" + weakest, "Box Plot (by RM) for %s, "
                         "one of the attributes that is not significant."
                         % weakest, 88)
    t_rm = N.table(
        "Student's t and p-value from the Box Plot for every attribute, "
        "grouped by RM (%s against %s)." % (rm_low, rm_high),
        ["Attribute", "Mean, RM %s" % rm_low, "Mean, RM %s" % rm_high,
         "Student's t", "p-value", "p &lt; 0.05?"],
        [(d["attribute"], "%.2f" % d["means"][0], "%.2f" % d["means"][1],
          "%.3f" % d["t_widget"], p_text(d["p_widget"]),
          "Yes" if d["p_widget"] < 0.05 else "No") for d in tests],
        ["l", "r", "r", "r", "r", "l"])
    f_canvas = N.figure("canvas", "The finished Orange Canvas workflow.")

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Lab 3 - Correlations and Hypothesis Testing</title>
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
code {{ font-family: Menlo, monospace; font-size: 9.5pt; }}
</style></head><body>

<h1>Lab Exercise #3: Correlations and Hypothesis Testing</h1>
<div class="meta">DATA-20000 Introduction to Data Science &middot; Saleh Yahya
&middot; workflow file: lab3_yahya.ows</div>

<h2>Step 1: Loading and exploring the data</h2>
<p>I loaded the Housing dataset that comes with Orange (housing.tab) in the
File widget, the same way the sample workflow does. It has {ex["rows"]} rows,
one per Boston area town, and {ex["columns"]} numeric columns with no missing
values. MEDV, the median home value in $1000's, comes in as the target.</p>
<p>The handout says this step doesn't need to be documented, so I'll keep it
short. I looked at the data with a Data Table, a Box Plot and a Distributions
widget, and marked those three on the canvas as the initial data
exploration.</p>
{f_table}
{f_ebox}
{f_edist}
<p>MEDV goes from {ex["medv"]["min"]:.0f} to {ex["medv"]["max"]:.0f} with a
mean of {ex["medv"]["mean"]:.1f}. One thing that stood out is the bar at
exactly 50: {ex["medv"]["at_50"]} towns have a MEDV of 50.0, which looks like
the values were capped there. That comes up again in step 4.</p>

<h2>Step 2: Select Columns</h2>
<p>I connected File to Select Columns and moved MEDV from the Target box to
the Features box, so MEDV and all the other columns are features. Then I
connected Select Columns to Correlations, and Correlations to a Scatter
Plot.</p>
{f_select}

<h2>Step 3: Pearson correlations with MEDV</h2>
<p>In the Correlations widget I typed MEDV in the filter box so it only
lists the pairs that include MEDV.</p>
{f_corr}
{t_corr}
<div class="q">
<p><b>The pair with the highest positive correlation is MEDV and RM, with a
Pearson correlation of +{r1["RM"]["r"]:.3f}.</b></p>
<p>RM is the average number of rooms per home, so towns with bigger homes
tend to have higher home values. I also noticed that the strongest
correlation overall is negative: LSTAT and MEDV at {r1["LSTAT"]["r"]:.3f}.
It is at the top of the list in Orange because the widget sorts by strength
and ignores the sign.</p>
</div>

<h2>Step 4: Does Pearson capture the relationship?</h2>
<p>With the MEDV and RM pair selected in Correlations, the Scatter Plot shows
those two attributes.</p>
{f_srm}
<div class="q">
<p><b>Do you think the Pearson correlation effectively captures the
relationship between the two selected attributes? Why or why not?</b></p>
<p>Mostly yes, but not completely. The points do follow a roughly straight,
upward band, which is the kind of relationship Pearson is made to measure,
and +{r1["RM"]["r"]:.2f} is a fair summary of it: a clear positive
relationship that is far from perfect. A few things in the plot are not
captured by that one number though:</p>
<ul>
<li>There is a vertical line of points at MEDV = 50. Those are the
{ex["medv"]["at_50"]} capped towns, and their RM values run from about 5 to
almost 9. Their real values are probably above 50, so they don't follow the
trend and they pull the correlation down.</li>
<li>There are a few outliers, like one town with almost 9 rooms but a MEDV
of about 22, and two towns with under 4 rooms and average values. Pearson is
sensitive to points like these.</li>
<li>The spread is not even. The points are tight in the middle and much
looser at low MEDV, so the line fits some parts better than others.</li>
</ul>
<p>So I'd say Pearson gives the right direction and a reasonable strength
for this pair, but it hides the cap at 50 and the outliers, which you only
see by looking at the plot.</p>
</div>
<p>For comparison I also clicked on the strongest pair in the list, LSTAT
and MEDV. Here the answer is clearly no.</p>
{f_slstat}
<p>The points follow a curve, not a line. MEDV drops fast when LSTAT is low
and then flattens out. Pearson only measures straight line relationships, so
{r1["LSTAT"]["r"]:.2f} understates how closely these two are tied together.
The straight regression line misses at both ends. This is the pair I used
for step 5.</p>

<h2>Step 5: Formula (log base 2)</h2>
<p>The handout says to take the log of the feature most correlated to MEDV.
Going by strength, that is LSTAT ({r1["LSTAT"]["r"]:.3f}), so I connected
Select Columns to a Formula widget and made a new numeric column:</p>
<p><code>log2_LSTAT = log2(LSTAT)</code></p>
<p>Since step 3 asked for the highest positive pair, which was RM, I also
added <code>log2_RM = log2(RM)</code> in the same widget so I could compare
both.</p>
{f_formula}
<p>Then I connected Formula to a second Correlations widget and a second
Scatter Plot, filtered on MEDV again, and selected the most correlated pair
that includes MEDV.</p>
{f_corr2}
{f_slog}
{t_log}
<div class="q">
<p><b>What do you notice? Describe what you now see in both widgets compared
to step 4 and explain the implications of the newly constructed
feature.</b></p>
<p><b>In Correlations:</b> the new feature goes straight to the top of the
list. MEDV and log2_LSTAT have a correlation of
{r2["log2_LSTAT"]["r"]:.3f}, compared with {r1["LSTAT"]["r"]:.3f} for plain
LSTAT, which is now second. In step 4 nothing was above 0.74 in strength.
The other pairs did not change, because I only added columns and did not
touch the old ones. log2_RM went the other way: its correlation is
+{r2["log2_RM"]["r"]:.3f}, a little lower than RM's +{r1["RM"]["r"]:.3f}.</p>
<p><b>In the Scatter Plot:</b> the curve is gone. With plain LSTAT the points
bent away from the line at both ends. With log2_LSTAT they fall along a
fairly straight band from top left to bottom right, and the regression line
runs through the middle of them the whole way. The capped towns at MEDV = 50
are still there as a vertical line on the right.</p>
<p><b>Implications:</b></p>
<ul>
<li>The relationship between LSTAT and MEDV was always strong. It just was
not linear, so Pearson was underrating it. Taking the log straightened it
out, and now Pearson describes it well.</li>
<li>The log means that what matters is the percentage change in LSTAT, not
the raw change. Going from 2% to 4% lower status population goes with about
the same drop in home value as going from 10% to 20%.</li>
<li>log2_LSTAT would be a better input than LSTAT for any method that
assumes a straight line, like linear regression. It should predict MEDV
better with the same data.</li>
<li>A transformation only helps when the relationship is curved. RM was
already close to linear, so taking its log made the correlation slightly
worse. So it is worth checking the scatter plot first and not just applying
a log to everything.</li>
</ul>
</div>

<h2>Step 6: Discretize CRIM and look at MEDV</h2>
<p>I connected File to a Discretize widget and set only CRIM to equal
frequency with 2 intervals. All the other attributes are left as keep
numeric. Equal frequency with 2 intervals splits at the median crime rate
({s7["crim_median"]:.5f}), so each group has {g_low["n"]} towns.</p>
{f_dcrim}
<p>Then I sent that to a Distributions widget and looked at MEDV split by
CRIM.</p>
{f_dist}
{t_crim}
<div class="q">
<p><b>What do you notice? How can one interpret this result?</b></p>
<p>The two groups have clearly different distributions. The low crime towns
(blue) sit further to the right, with a mean MEDV of {g_low["mean"]:.1f}
against {g_high["mean"]:.1f} for the high crime towns (red). That is a gap
of about ${round((g_low["mean"] - g_high["mean"]) * 10) * 100:,.0f} in median
home value. The low crime
group is also narrower and looks more like a bell curve. The high crime
group is more spread out (standard deviation {g_high["sd"]:.1f} against
{g_low["sd"]:.1f}) and almost all of the cheapest towns, the ones under
about 13, are in it.</p>
<p>The interpretation is that towns with higher crime rates tend to have
lower home values. It is not a clean split though. The two distributions
overlap a lot, and some high crime towns are expensive, including most of
the towns capped at 50. So crime rate tells you something about home value
on average, but it does not decide it for any one town. It also does not
prove that crime causes lower prices, since other things like LSTAT go along
with both.</p>
</div>

<h2>Step 7: Box Plot and the t statistic</h2>
<p>I connected the same Discretize widget to a Box Plot, chose MEDV as the
variable, CRIM as the subgroup, and set it to compare means.</p>
{f_box}
<div class="q">
<p><b>What is the value of the t statistic? What does that tell us about the
data? Assume alpha = 0.05.</b></p>
<p><b>Student's t = {s7["t_widget"]:.3f}</b>, with p = 0.000 as shown in
Orange (so p &lt; 0.001) and N = 506.</p>
<p>The null hypothesis is that the mean MEDV is the same in low crime and
high crime towns. The p-value is far below 0.05, so I reject it. A t of
{s7["t_widget"]:.2f} means the difference between the two means
({g_low["mean"]:.2f} and {g_high["mean"]:.2f}) is about 6 standard errors
wide, which would almost never happen by chance if the two groups really had
the same mean. So the difference in home values between low and high crime
towns is statistically significant. It tells us the gap is real, but like
in step 6 it does not tell us that crime is the cause.</p>
</div>

<h2>Step 8: Discretize RM and test every attribute</h2>
<p>I added a new Discretize widget connected to File, set only RM to equal
frequency with 2 intervals and kept everything else numeric. The split is at
{rm_low.replace("< ", "")} rooms, with {s8["counts"][0]} towns in each
group. I fed that to a new Box Plot, grouped by RM, and clicked through
every attribute to read its t and p-value.</p>
{f_drm}
{f_rm_top}
{f_rm_medv}
{f_rm_weak}
{t_rm}
<div class="q">
<p><b>Attributes with a p-value lower than 0.05:</b>
{", ".join(d["attribute"] for d in sig)} ({len(sig)} of the
{s8["n_tests"]}).</p>
<p>The ones that are not below 0.05 are
{", ".join("%s (p = %.3f)" % (d["attribute"], d["p_widget"])
           for d in not_sig)}.</p>
<p><b>Can we say all these are statistically significant? Why or why
not?</b></p>
<p>Not all of them, at least not with the same confidence. Each one passes
on its own, but I ran {s8["n_tests"]} tests, not one. With alpha = 0.05 every
test has a 5% chance of a false positive, so across {s8["n_tests"]} tests
the chance of getting at least one by luck is about
{s8["chance_of_any_false_positive"] * 100:.0f}%, not 5%. This is the
multiple comparisons problem.</p>
<p>One way to fix it is the Bonferroni correction, which divides alpha by
the number of tests: 0.05 / {s8["n_tests"]} =
{s8["bonferroni_alpha"]:.4f}. With that stricter cutoff,
{" and ".join(lost)} (both p = {by_name[lost[0]]["p_widget"]:.3f}) no longer
pass. The other {len(s8["significant_bonferroni"])} still do, since their
p-values are tiny, so I'm confident in those.</p>
<p>Two more things to keep in mind. First, the groups were made from RM, and
RM is closely related to several of these attributes (MEDV and LSTAT
especially), so those results are expected and the tests are not independent
of each other. Second, statistically significant does not mean important.
With 506 towns even a small difference can pass the test, so I'd still look
at the size of the difference in the table before reading much into it.</p>
</div>

<h2>The workflow</h2>
<p>Every part of the workflow is labelled with a text box and an arrow, and
I renamed the widgets that appear more than once so it's clear which is
which.</p>
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
