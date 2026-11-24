"""Build Lab2/submission/lab2_yahya.pdf from what the workflow produced.

Reads Lab2/results/ (written by run_workflow.py), writes report.html there
and prints it to PDF with headless Chrome.

    /Applications/Orange.app/Contents/MacOS/python Lab2/build_report.py
"""
import html
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PDF = os.path.join(HERE, "submission", "lab2_yahya.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


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


def main():
    with open(os.path.join(RESULTS, "results.json")) as f:
        R = json.load(f)
    N = Numbering()
    e = html.escape
    ex, s5 = R["explore"], R["step5"]
    st = ex["stats"]
    trials = R["impute_trials"]
    avg = trials["Average/Most frequent"]
    model = trials["Model-based imputer (simple tree)"]
    rand = trials["Random values"]
    ref = R["impute_reference"]
    oc = R["outlier_car"]
    methods = R["outlier_methods"]
    by_make = R["by_make"]
    # results.json holds the sample sd (n-1); Orange's Distributions legend
    # shows the population sd (n), so convert to match the figures.
    for v in by_make.values():
        k = ((v["count"] - 1) / v["count"]) ** 0.5
        for name, x in v.items():
            if name != "count":
                x["sd"] *= k
    cyl = ex["categories"]["cylinders"]
    origin = ex["categories"]["origin"]

    # ---------- figures and tables, in the order they appear ----------
    f_file = N.figure("file", "The File widget with auto-mpg.csv loaded. "
                      "cylinders, model_1 and origin are set to categorical "
                      "and model_2 is a text (meta) column.", 62)
    f_table = N.figure("data_table", "Data Table showing the first rows of "
                       "the 398 cars.")
    f_stats = N.figure("feature_statistics", "Feature Statistics. The "
                       "Missing column shows 6 missing values for "
                       "horsepower and none anywhere else.")
    f_box = N.figure("box_mpg_origin", "Box Plot of mpg grouped by origin.", 88)
    f_dist = N.figure("dist_mpg", "Distributions widget showing mpg.", 88)
    f_scatter = N.figure("scatter_weight_mpg", "Scatter Plot of weight "
                         "(x) against mpg (y), coloured by cylinders.", 88)
    t_problems = N.table(
        "Data problems I found in auto-mpg.csv.",
        ["Problem", "What I saw", "Why it matters"],
        [("Missing values",
          "6 cars have no horsepower (shown as ? in the table)",
          "They need to be filled in or dropped before any analysis that "
          "uses horsepower"),
         ("Misspelled makes",
          "chevroelt, vokswagen, maxda, toyouta",
          "The same make gets counted as different ones"),
         ("Same make written different ways",
          "chevrolet / chevy, volkswagen / vw, mercedes / mercedes-benz",
          "Same issue, so I needed two rows for VW in step 2"),
         ("Names that are not really a make",
          "hi 1200d, capri",
          "Can't be matched to a manufacturer from the name alone"),
         ("Car names repeat",
          "%d rows share a name with another row (for example several "
          "ford pinto rows from different years)" % ex["duplicate_car_names"],
          "The handout says model_2 is unique, but it is not, so it can't "
          "be used as an ID"),
         ("Codes with no labels",
          "origin is 1, 2 or 3 and model_1 is a two digit year",
          "They look like numbers but they are categories, so I set them "
          "to categorical in the File widget"),
         ("Unclear column names",
          "model_1 is the year and model_2 is the car name",
          "Easy to mix up"),
         ("Possible outliers",
          "Three trucks at 9 to 10 mpg, and a few cars with very high mpg "
          "for their engine size",
          "Looked at more closely in step 4")])

    f_create = N.figure("create_class", "Create Class widget. The Count "
                        "column matches the numbers in the handout: FORD "
                        "51, CHEVY 47, VW 15 + 6 = 21, OTHER 279.", 66)
    t_make = N.table(
        "Number of cars in each car_make class.",
        ["Class", "Substring used", "Cars"],
        [("FORD", "ford", R["make_counts"]["FORD"]),
         ("CHEVY", "chev", R["make_counts"]["CHEVY"]),
         ("VW", "volkswagen, vw (two rows)", R["make_counts"]["VW"]),
         ("OTHER", "(remaining instances)", R["make_counts"]["OTHER"])],
        ["l", "l", "r"])

    f_t1 = N.figure("table1_selected", "Data Table (1) sorted by "
                    "horsepower, with the 6 cars that have unknown "
                    "horsepower selected.")
    f_t2a = N.figure("table2_average", "Data Table (2) with "
                     "Average/Most frequent.", 92)
    f_t2m = N.figure("table2_model", "Data Table (2) with Model-based "
                     "imputer (simple tree).", 92)
    f_t2r = N.figure("table2_random", "Data Table (2) with Random values.", 92)
    t_impute = N.table(
        "Horsepower filled in for the 6 cars by each Impute option.",
        ["Car", "Cylinders", "Displacement", "Weight", "Average",
         "Model-based", "Random"],
        [(e(a["model_2"]), a["cylinders"], "%.0f" % a["displacement"],
          "%.0f" % a["weight"], "%.1f" % a["horsepower"],
          "%.1f" % m["horsepower"], "%.0f" % r["horsepower"])
         for a, m, r in zip(avg, model, rand)],
        ["l", "r", "r", "r", "r", "r", "r"])

    f_s1 = N.figure("scatter1_outlier", "Scatter Plot (1), displacement "
                    "against mpg after imputing. The outlier is the "
                    "selected point (orange ring) near displacement 262 "
                    "and 38 mpg.", 88)
    f_t3 = N.figure("table3_outlier", "Data Table (3) showing the selected "
                    "outlier.")
    f_outw = N.figure("outliers_widget", "Outliers widget settings.", 40)
    f_s2 = N.figure("scatter2_outliers", "Scatter Plot (2), fed from the "
                    "Outliers output of the Outliers widget. It shows the "
                    "%d cars that were flagged, and my outlier is one of "
                    "them (displacement 262, 38 mpg)." % R["outliers_flagged"],
                    88)
    f_inl = N.figure("scatter3_inliers_disp_mpg", "Displacement against mpg "
                     "for the %d inliers only. The point at 262 / 38 is "
                     "gone." % R["inliers"], 88)
    t_methods = N.table(
        "What each method in the Outliers widget did with my outlier.",
        ["Method", "Cars flagged", "Flags the Oldsmobile diesel?"],
        [(k, v["flagged"], "Yes" if v["catches_the_diesel"] else "No")
         for k, v in methods.items()], ["l", "r", "l"])

    f_fc = N.figure("feature_constructor", "Feature Constructor with the "
                    "Lper100k formula.", 66)
    f_hpmpg = N.figure("scatter3_hp_mpg", "Scatter Plot (3): horsepower "
                       "(x) against mpg (y), with the regression line.", 88)
    f_hpl = N.figure("scatter3_hp_lper100k", "Scatter Plot (3): horsepower "
                     "(x) against Lper100k (y), with the regression line.", 88)
    f_save = N.figure("save_data", "Save Data widget, saving to "
                      "auto-mpg-lper100k.csv.", 38)

    f_file1 = N.figure("file1", "File (1) with the saved csv loaded and "
                       "car_make set as the target.", 62)
    step6 = {n: N.figure("step6_" + n, "Step 6 Distributions for %s, "
                         "fitted with Normal and split by car_make." % n, 88)
             for n in ("mpg", "Lper100k", "displacement", "horsepower",
                       "weight", "acceleration")}
    cols = ["mpg", "Lper100k", "displacement", "horsepower", "weight",
            "acceleration"]
    t_by_make = N.table(
        "Mean (standard deviation) of each variable by car_make, from the "
        "saved file.",
        ["car_make", "Cars"] + cols,
        [[k, v["count"]] + ["%.1f (%.1f)" % (v[c]["mean"], v[c]["sd"])
                            for c in cols]
         for k, v in by_make.items()],
        ["l", "r"] + ["r"] * len(cols))
    f_canvas = N.figure("canvas", "The finished Orange Canvas workflow.")

    vw, ford, chevy, other = (by_make[k] for k in
                              ("VW", "FORD", "CHEVY", "OTHER"))

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Lab 2 - Data Exploration and Processing</title>
<style>
@page {{ size: Letter; margin: 0.8in 0.85in; }}
body {{ font: 10.5pt/1.5 Helvetica, Arial, sans-serif; color: {INK}; }}
h1 {{ font-size: 20pt; line-height: 1.2; margin: 0 0 4pt; }}
h2 {{ font-size: 13.5pt; margin: 20pt 0 6pt; padding-top: 8pt;
      border-top: 1px solid {GRID}; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 12pt 0 3pt; break-after: avoid; }}
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
.q p:last-child {{ margin-bottom: 0; }}
code {{ font-family: Menlo, monospace; font-size: 9.5pt; }}
</style></head><body>

<h1>Lab Assignment #2: Data Exploration and Processing</h1>
<div class="meta">DATA-20000 Introduction to Data Science &middot; Saleh Yahya
&middot; workflow file: lab2_yahya.ows</div>

<h2>Step 1: Loading and exploring the data</h2>
<p>I loaded auto-mpg.csv with the File widget. It has {ex["rows"]} cars and
{ex["columns"]} columns. I set the column types to match the list in the
handout: cylinders, model_1 and origin are categorical, model_2 is text, and
the rest are numeric.</p>
{f_file}
<h3>Data Table</h3>
{f_table}
<p>Each row is one car. Looking down the table, the cars from the early
1970s are mostly big 8 cylinder American cars with low mpg. I also noticed
that horsepower has a few cells with a "?" where the value is missing, and
that the same car name shows up more than once.</p>
<h3>Feature Statistics</h3>
{f_stats}
<p>This gives a quick summary of every column. mpg goes from
{st["mpg"]["min"]:.0f} to {st["mpg"]["max"]:.1f} with a mean of
{st["mpg"]["mean"]:.1f}. Weight goes from {st["weight"]["min"]:.0f} to
{st["weight"]["max"]:.0f}. Displacement, horsepower and weight are all skewed
to the right, meaning most cars are on the smaller side with a tail of big
ones. Horsepower is the only column with missing data
({st["horsepower"]["missing"]} values, about 2%). For cylinders,
{cyl["4"]} cars have 4, {cyl["8"]} have 8, {cyl["6"]} have 6, and only
{cyl["3"]} have 3 and {cyl["5"]} have 5.</p>
<h3>Box Plot</h3>
{f_box}
<p>I grouped mpg by origin. Origin 1 has the lowest mpg (median
{ex["median_mpg_by_origin"]["1"]:.1f}), origin 2 is in the middle
({ex["median_mpg_by_origin"]["2"]:.1f}) and origin 3 is the highest
({ex["median_mpg_by_origin"]["3"]:.1f}). Going by the car names, origin 1 is
American cars, 2 is European and 3 is Japanese, so the American cars in this
data use the most fuel. Origin 1 is also the biggest group with
{origin["1"]} of the {ex["rows"]} cars.</p>
<h3>Distributions</h3>
{f_dist}
<p>mpg is not symmetric. Most cars sit between about 13 and 30 mpg, and there
is a longer tail on the right going up past 40. So there are a few very
efficient cars but not many.</p>
<h3>Scatter Plot</h3>
{f_scatter}
<p>Heavier cars get lower mpg, and the relationship is strong (the
correlation is {ex["corr_weight_mpg"]:.2f}). The colours show that the 8
cylinder cars are the heavy, low mpg group and the 4 cylinder cars are the
light, high mpg group. The trend is curved rather than a straight line: mpg
drops quickly at first and then levels off for the heaviest cars.</p>
<div style="break-inside: avoid">
<h3>Potential problems with the data</h3>
<p>These are the things I found that would need cleaning:</p>
{t_problems}
</div>

<h2>Step 2: Creating the car_make class</h2>
<p>I used Create Class on the model_2 column to make a new class called
car_make. For Chevy I used the substring "chev" because it matches
chevrolet, chevy and the misspelled chevroelt all at once. VW needed two rows
with the same class name, one for "volkswagen" and one for "vw". Everything
that didn't match went into OTHER.</p>
{f_create}
{t_make}
<p>The counts match the handout exactly. One thing I noticed is that the
misspelled "vokswagen" car ends up in OTHER, since it doesn't contain either
VW substring. The handout's count of 21 for VW only works if that car is left
out, so I kept it that way.</p>

<h2>Step 3: Imputing the missing horsepower</h2>
<p>I sent the Create Class output to Data Table (1) and to Impute. In Data
Table (1) I sorted by horsepower so the unknown values came to the top, and
selected those 6 rows.</p>
{f_t1}
<p>The selected rows go into the Data Subset input of Select by Data Index,
and the Impute output goes into its Data input. That way Data Table (2) shows
only those same 6 cars, but with whatever value Impute filled in. I tried the
three options one at a time.</p>
{f_t2a}
{f_t2m}
{f_t2r}
{t_impute}
<p>The model-based option worked for me without connecting a Tree widget,
because Impute already uses a simple tree by default, so I did not need to
add one.</p>
<div class="q">
<p><b>Which option seems to generate the most appropriate values to be
filled in for the missing ones?</b></p>
<p>The model-based imputer (simple tree). It gives each car its own value
based on the rest of that car's data, so the small 4 cylinder cars get lower
numbers (the Ford Pinto gets {model[0]["horsepower"]:.1f}) and the 6 cylinder
Ford Maverick gets a higher one ({model[1]["horsepower"]:.1f}). That makes
sense for these cars.</p>
<p>Average gives every car the same value, {avg[0]["horsepower"]:.1f}, which
is the mean of the whole dataset including the big V8s. Five of the six cars
are 4 cylinder, and the average horsepower for 4 cylinder cars in this data
is only {ref["mean_hp_4cyl"]:.1f}, so {avg[0]["horsepower"]:.1f} is too high
for them.</p>
<p>Random just picks an existing horsepower value by chance, so it has
nothing to do with the car. In my run the Renault LeCar, the smallest car of
the six, got {rand[2]["horsepower"]:.0f}. The values also change every time
it runs.</p>
</div>

<h2>Step 4: Outliers</h2>
<p>I connected the Impute output to Scatter Plot (1) and looked at
displacement against mpg. Most of the points follow a curve going down: the
bigger the engine, the lower the mpg. One point sits well above everything
around it, at a displacement of {oc["displacement"]:.0f} and
{oc["mpg"]:.0f} mpg. Other cars with engines that size get roughly 15 to 25
mpg.</p>
{f_s1}
<p>I selected that point and sent it to Data Table (3) to see which car it
is.</p>
{f_t3}
<p>It is the <b>{e(oc["model_2"])}</b>, a 19{oc["model_1"]} car with
{oc["cylinders"]} cylinders, {oc["horsepower"]:.0f} horsepower and a weight
of {oc["weight"]:.0f}.</p>
<div class="q">
<p><b>Do you think this outlier is due to data entry errors or just a vehicle
with unusual characteristics?</b></p>
<p>I think it is a vehicle with unusual characteristics, not a data entry
error. The name says it is a diesel, and diesel engines get much better
mileage than gas engines of the same size. Its other numbers back that up:
{oc["horsepower"]:.0f} horsepower is low for a 262 engine, which is normal
for a diesel. The other diesels in the data (the VW Rabbit and Dasher
diesels, for example) also have very high mpg. It is also a 1982 car, and the
newer cars in this data are more efficient in general. So the values look
real, the car is just different from the gas cars around it.</p>
</div>
<p>Next I fed the Impute output into the Outliers widget. With the default
method, Local Outlier Factor, my outlier was <i>not</i> flagged, so it stayed
in the inliers. I tried the other methods and switched to Covariance
Estimator, which does catch it.</p>
{t_methods}
{f_outw}
<p>With Covariance Estimator at 10% contamination the widget splits the data
into {R["outliers_flagged"]} outliers and {R["inliers"]} inliers. Following
the sample workflow, Scatter Plot (2) is connected to the Outliers output, so
it shows the cars that were taken out.</p>
{f_s2}
<p>To double check that it is really gone from the data that continues on,
I also looked at displacement against mpg for the inliers.</p>
{f_inl}
<p>So yes, the outlier I found is eliminated. One thing to keep in mind is
that the widget always removes about 10% of the data, so it also took out
cars I would not have called outliers by eye, like the pickup trucks, the
rotary Mazdas and the other diesels.</p>

<h2>Step 5: Lper100k</h2>
<p>I sent the inliers to Feature Constructor and made a new numeric column
called Lper100k using the formula from the handout:</p>
<p><code>Lper100k = 100 * 3.785411784 / (1.609344 * mpg)</code></p>
{f_fc}
<p>I put brackets around the bottom part so that mpg is divided, not
multiplied. A quick check: {e(s5["example"][0]["model_2"])} has
{s5["example"][0]["mpg"]:.0f} mpg, which comes out as
{s5["example"][0]["Lper100k"]:.2f} L per 100 km. The values go from
{s5["lper100k_min"]:.1f} to {s5["lper100k_max"]:.1f}, which is a realistic
range.</p>
{f_hpmpg}
{f_hpl}
<p>The two plots show the same cars but they tell the story differently.
With mpg, the points curve. At low horsepower mpg is spread out a lot and
falls fast, and at high horsepower it flattens out, because mpg can't go
below zero. With Lper100k the direction flips, since it measures fuel used
instead of distance travelled: more horsepower means more fuel burned. The
points also line up along the line much better. Each extra bit of horsepower
adds about the same amount of fuel use across the whole range.</p>
<div class="q">
<p><b>Which of the graphs appears more linear in nature? What are the
implications?</b></p>
<p>Horsepower against Lper100k is more linear. Orange's regression line
gives r = {s5["r_hp_lper100k"]:.2f} for Lper100k and
r = {s5["r_hp_mpg"]:.2f} for mpg, so the Lper100k relationship is stronger,
and the mpg points bend away from their line at both ends.</p>
<p>The implication is that fuel use per distance is the better thing to work
with if I want to use a straight line method, like linear regression or a
correlation. A straight line would describe Lper100k well, but for mpg it
would predict too high in the middle and too low at the ends. It also
means mpg can be misleading when comparing cars. Going from 10 to 15 mpg
saves a lot more fuel than going from 30 to 35 mpg, even though both are a 5
mpg improvement. Lper100k shows that difference directly.</p>
</div>
<p>Last, I saved this dataset with the Save Data widget as
auto-mpg-lper100k.csv.</p>
{f_save}

<h2>Step 6: Distributions by car make</h2>
<p>I opened the saved csv with a new File widget and set car_make as the
target.</p>
{f_file1}
<p>Then I connected it to a Distributions widget (named Step 6), set the
fitted distribution to Normal and split by car_make. This file has the
{R["inliers"]} inliers from step 4, so the class sizes are a little smaller
than in step 2.</p>
{t_by_make}
{step6["mpg"]}
{step6["Lper100k"]}
{step6["displacement"]}
{step6["horsepower"]}
{step6["weight"]}
{step6["acceleration"]}
<p><b>What I noticed about the differences between the classes:</b></p>
<ul>
<li><b>VW stands apart from the others.</b> Its curve is shifted toward
better fuel economy (mean {vw["mpg"]["mean"]:.1f} mpg against about 20 for
Ford and Chevy), and toward small engines, low horsepower and low weight.
Its curves are also much narrower. For displacement the VW standard
deviation is only {vw["displacement"]["sd"]:.0f}, compared with
{ford["displacement"]["sd"]:.0f} for Ford. VW basically made one type of car,
a small 4 cylinder.</li>
<li><b>Ford and Chevy are almost the same.</b> Their curves sit nearly on top
of each other for every variable. Both have low average mpg
({ford["mpg"]["mean"]:.1f} and {chevy["mpg"]["mean"]:.1f}) and are wide,
because both sold everything from small 4 cylinder cars to big V8s.</li>
<li><b>OTHER is in the middle and is the widest.</b> That makes sense
because it mixes a lot of different makes, from small Japanese and European
cars to big American ones.</li>
<li><b>The normal curve doesn't fit every class well.</b> For Ford and Chevy
the bars for displacement and weight have two bumps (small cars and big
cars), so one bell curve smooths over that. It fits VW better since VW is one
tight group.</li>
<li><b>Acceleration is the one variable where the classes look alike.</b>
All four are centred around 15 to 16 and overlap almost completely, so car
make doesn't tell you much about acceleration.</li>
<li><b>VW has few cars.</b> With only {vw["count"]} VWs, its curve is based
on a lot less data than the others.</li>
</ul>

<h2>Step 7: The workflow</h2>
<p>I labelled each part of the workflow with a text box and an arrow.</p>
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
