"""Build Lab6/submission/lab6_yahya.pdf from what the workflow produced.

Reads Lab6/results/ (written by run_workflow.py), writes report.html there
and prints it to PDF with headless Chrome.

    /Applications/Orange.app/Contents/MacOS/python Lab6/build_report.py
"""
import html
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
PDF = os.path.join(HERE, "submission", "lab6_yahya.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
e = html.escape


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


def pct(x):
    return "%d%%" % round(100 * x)


def words(pairs, n, fmt="%s (%d)"):
    return ", ".join(fmt % (e(w), c) for w, c in pairs[:n])


def shares(pairs, n):
    return ", ".join("%s (%s)" % (e(w), pct(s)) for w, s in pairs[:n])


def main():
    with open(os.path.join(RESULTS, "results.json")) as f:
        r = json.load(f)
    s2, s3, s4, s5, s6 = (r["step2"], r["step3"], r["step4"], r["step5"],
                          r["step6"])
    hc, km, cross, exps = r["step7"], r["step8"], r["cross"], r["step9"]
    N = Numbering()
    n = s3["sample"]
    raw = dict(s3["cloud"])
    top = dict(s4["cloud"])
    states = s5["states"]

    f_file = N.figure("file", "The File widget. The five text columns are "
                      "kept as meta and every other column is set to skip.",
                      52)
    f_table = N.figure("data_table", "The Data Table with the first tweets.")
    t_sample = N.table(
        "A small sample of the data (first rows of the file).",
        ["name", "text", "tweet_coord", "tweet_created", "tweet_location"],
        [[e(c if c not in ("?", "") else "(empty)") for c in row]
         for row in s2["sample_rows"][:5]])
    f_sampler = N.figure("data_sampler", "Data Sampler set to 10% of the "
                         "data.", 34)
    f_corpus = N.figure("corpus", "Corpus widget. Only text is under Used "
                        "text features.", 60)
    f_raw = N.figure("word_cloud_raw", "Word cloud straight from the Corpus "
                     "widget, before any preprocessing.", 88)
    f_pre = N.figure("preprocess_text", "My Preprocess Text settings.", 70)
    f_cloud = N.figure("word_cloud_preprocessed", "Word cloud after "
                       "Preprocess Text.", 88)
    f_map = N.figure("document_map", "Document Map with tweet_location as "
                     "the region attribute and USA as the map type.", 88)
    t_states = N.table(
        "States with the most tweets in my sample.", ["State", "Tweets"],
        [[e(names[1]), c] for names, _, c in states[:8]], ["l", "r"])
    f_bow = N.figure("bag_of_words", "Bag of Words settings.", 36)
    f_dist = N.figure("distances", "Distances set to Jaccard.", 38)
    f_hc = N.figure("hierarchical_clustering", "Hierarchical Clustering "
                    "with complete linkage and Top N = 3.", 88)
    f_select = N.figure("select_hc_c1", "One of the Select Rows widgets. "
                        "This one keeps the rows where Cluster is C1.", 62)
    f_hc_clouds = "".join(
        N.figure("cloud_hc_c%d" % (i + 1),
                 "Hierarchical clustering, cluster C%d (%d tweets)."
                 % (i + 1, g["size"]), 80) for i, g in enumerate(hc))
    t_hc = N.table(
        "Hierarchical clusters. The percentage is the share of the tweets "
        "in the cluster that have the word.",
        ["Cluster", "Tweets", "Most common words"],
        [["C%d" % (i + 1), g["size"], shares(g["share"], 7)]
         for i, g in enumerate(hc)], ["l", "r", "l"])
    f_km = N.figure("kmeans", "k-Means with 3 clusters.", 40)
    f_km_clouds = "".join(
        N.figure("cloud_km_c%d" % (i + 1),
                 "k-Means, cluster C%d (%d tweets)." % (i + 1, g["size"]),
                 80) for i, g in enumerate(km))
    t_km = N.table(
        "k-Means clusters.", ["Cluster", "Tweets", "Most common words"],
        [["C%d" % (i + 1), g["size"], shares(g["share"], 7)]
         for i, g in enumerate(km)], ["l", "r", "l"])
    t_cross = N.table(
        "The same tweets in both clusterings. Rows are the hierarchical "
        "clusters and columns are the k-Means clusters.",
        ["", "k-Means C1", "k-Means C2", "k-Means C3"],
        [["Hierarchical C%d" % (i + 1)] + row for i, row in enumerate(cross)],
        ["l", "r", "r", "r"])
    base = {"metric": "Jaccard", "linkage": "Complete",
            "sizes": [g["size"] for g in hc],
            "share": [g["share"] for g in hc]}
    t_exp = N.table(
        "What I tried in Hierarchical Clustering. The first row is the "
        "setting from step 7.",
        ["Distance", "Linkage", "Cluster sizes (C1, C2, C3)",
         "Top word in each cluster"],
        [[x["metric"], x["linkage"], ", ".join(str(s) for s in x["sizes"]),
          ", ".join(e(s[0][0]) for s in x["share"])]
         for x in [base] + exps])
    exp = {(x["metric"], x["linkage"]): x for x in exps}
    f_ward = N.figure("exp_jaccard_ward", "Jaccard distance with Ward "
                      "linkage.")
    f_cos = N.figure("exp_cosine_complete", "Cosine distance with complete "
                     "linkage.")
    f_canvas = N.figure("canvas", "The finished Orange Canvas workflow.")

    airlines = ["united", "usairways", "americanair", "jetblue",
                "southwestair", "virginamerica"]
    ward, single = exp[("Jaccard", "Ward")], exp[("Jaccard", "Single")]
    average = exp[("Jaccard", "Average")]
    cosine, eucl = exp[("Cosine", "Complete")], exp[("Euclidean", "Complete")]
    eucl_ward = exp[("Euclidean", "Ward")]
    in_hc3 = cross[2][1] + cross[2][2]

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Lab 6 - Saleh Yahya</title>
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
         font-size: 9pt; break-inside: avoid; }}
caption {{ caption-side: top; text-align: left; color: {INK2};
           font-size: 9pt; padding-bottom: 3pt; }}
th {{ text-align: left; color: {INK2}; font-weight: 600;
      border-bottom: 1.2px solid {MUTED}; padding: 4pt 6pt; }}
td {{ border-bottom: 1px solid {GRID}; padding: 4pt 6pt; vertical-align: top;
      overflow-wrap: anywhere; }}
td:first-child {{ white-space: nowrap; }}
.r {{ text-align: right; font-variant-numeric: tabular-nums; }}
figure {{ margin: 8pt 0 12pt; break-inside: avoid; text-align: center; }}
figcaption {{ color: {INK2}; font-size: 9pt; margin-top: 3pt;
              text-align: left; }}
.shot {{ border: 1px solid {GRID}; border-radius: 4px; }}
.q {{ background: #f4f3ef; border-radius: 6px; padding: 8pt 12pt;
      margin: 8pt 0; }}
.q p:last-child, .q ul:last-child {{ margin-bottom: 0; }}
.row {{ break-inside: avoid; display: flex; gap: 10pt; align-items: flex-start; }}
.row figure {{ flex: 1; }}
.row .shot {{ width: 100% !important; }}
</style></head><body>

<h1>Lab Exercise #6: Text Mining and Clustering</h1>
<div class="meta">DATA-20000 Introduction to Data Science &middot; Saleh Yahya
&middot; workflow file: lab6_yahya.ows</div>

<h2>Step 1: Installing the Text add-on</h2>
<p>I downloaded Tweets.csv and installed the Text add-on for Orange. After restarting Orange the Text Mining widgets (Corpus, Preprocess Text, Word Cloud, Bag of Words, Document Map) showed up in the widget list.</p>

<h2>Step 2: Loading the tweets</h2>
<p>I opened Tweets.csv with the File widget. The file has {s2["rows"]:,} tweets about airlines and 15 columns. I set every column to skip except the five text columns: name, text, tweet_coord, tweet_created and tweet_location. Then I connected the File widget to a Data Table.</p>
{f_file}
{f_table}
{t_sample}
<p>Looking at the table, every tweet has a name, a text and a time. The other two columns are mostly empty. Only {s2["filled"]["tweet_coord"]:,} tweets have coordinates and {s2["filled"]["tweet_location"]:,} have a location. The location is whatever the user typed in their profile, so it is not always a real place (for example "Lets Play").</p>

<h2>Step 3: Sample, corpus and the first word cloud</h2>
<p>I connected the File widget to a Data Sampler and set it to a fixed proportion of 10%. That gives {n:,} tweets. Then I connected it to the Corpus widget. When I opened the Corpus widget it had picked the name column as the text feature, so I moved name out and put text in. Now text is the only column under Used text features.</p>
{f_sampler}
{f_corpus}
<p>Then I connected the Corpus to a Word Cloud.</p>
{f_raw}
<div class="q">
<p><b>What I see.</b> The word cloud is mostly noise. The biggest things are not really words. The top one is the @ sign ({raw["@"]:.0f} times) and then the period ({raw["."]:.0f}). After that it is small common words like to ({raw["to"]:.0f}), i ({raw["i"]:.0f}), the ({raw["the"]:.0f}), you, a and for. There are also pieces of links like http, t and co, and other symbols like # and ?.</p>
<p>The only useful words I can find are the airline names (united, usairways, americanair, southwestair, jetblue) and flight. The airline names are there because every tweet starts with @ and the airline. So without cleaning the text, the word cloud does not tell me much about what people are saying.</p>
</div>

<h2>Step 4: Preprocess Text</h2>
<p>I put a Preprocess Text widget between the Corpus and the Word Cloud. In Transformation I checked Lowercase, Remove accents and Remove urls. I left Tokenization the way it was. In Filtering I kept Stopwords, added Regexp with its default pattern, and checked Most frequent tokens with 100. Then I added N-grams Range and set it to 1 to 2.</p>
{f_pre}
<p>After this the corpus has {s4["tokens"]:,} tokens and {s4["types"]} different words.</p>
{f_cloud}
<div class="q">
<p><b>What I see.</b> This word cloud is a lot easier to read. The symbols, the links and the small words like "to" and "the" are gone. The biggest words are still the airline names: {words([(a, top[a]) for a in airlines if a in top], 6)}. United is the airline people tweet at the most in my sample.</p>
<p>The other big word is flight ({top["flight"]:.0f}). After that the words are about problems and customer service: {words([(w, top[w]) for w in ("cancelled", "service", "help", "customer", "time", "hours", "hold") if w in top], 7)}. There are polite words too, like thanks ({top["thanks"]:.0f}) and thank. So most of these tweets look like people asking an airline for help or complaining about a flight.</p>
<p>Because the n-grams range is 1 to 2, there are also two word phrases, like "customer service" ({top.get("customer service", 0):.0f}), "cancelled flightled" and "united flight". A few things are still not clean. "amp" is left over from the &amp;amp; code for the &amp; sign, "flightled" is a strange word that is in the data itself ("Cancelled Flightled"), and "2" is used as a short way to write "to".</p>
</div>

<h2>Step 5: Document Map</h2>
<p>I connected Preprocess Text to the Document Map widget. I picked tweet_location as the Region attribute and USA as the Map type.</p>
{f_map}
{t_states}
<div class="q">
<p><b>Describing the map.</b> The map colors each state by how many tweets came from there. Darker red means more tweets. The darkest states are {e(states[0][0][1])} ({states[0][2]}), {e(states[1][0][1])} ({states[1][2]}) and {e(states[2][0][1])} ({states[2][2]}), then {e(states[3][0][1])}, {e(states[4][0][1])}, {e(states[5][0][1])}, {e(states[6][0][1])} and {e(states[7][0][1])}. These are states with a lot of people and big airports, so that makes sense. The states in the middle north, like Montana, Wyoming and the Dakotas, are white because no tweet in my sample matched them.</p>
<p>The map only shows part of the data. Out of the {n:,} tweets in my sample, {s5["with_location"]} have a location and only {s5["located"]} of them could be matched to one of {len(states)} states. The rest are empty, outside the US, or not a real place. Also the widget matches by name, so I would not trust every count. For example Washington is high, and some of those are probably Washington, D.C. and not the state.</p>
</div>

<h2>Step 6: Bag of Words</h2>
<p>I connected Preprocess Text to the Bag of Words widget. I picked Count for Term Frequency and Smooth IDF for Document Frequency. The output is a table with one row for each of the {s6["rows"]:,} tweets and {s6["features"]:,} columns, one for each word or two word phrase. Count means the number is how many times the word is in the tweet, and Smooth IDF makes words that are in a lot of tweets count less.</p>
{f_bow}

<h2>Step 7: Hierarchical clustering</h2>
<p>I connected Bag of Words to the Distances widget and picked Jaccard. Then I connected Distances to Hierarchical Clustering, set the linkage to Complete, and picked Top N with N = 3.</p>
{f_dist}
{f_hc}
{f_select}
<p>I connected the Selected Data output to three Select Rows widgets. Each one has the condition "Cluster is" C1, C2 or C3. Then each Select Rows goes into its own Word Cloud. The word clouds here show the bag of words weights and not plain counts, so the numbers on the left are small.</p>
{f_hc_clouds}
{t_hc}
<div class="q">
<p><b>Describing the word clouds.</b></p>
<ul>
<li><b>C1 ({hc[0]["size"]} tweets)</b> is the American Airlines cluster. Every tweet in it has americanair. Most of the other things in the cloud are phrases that start with it, like "americanair thanks", "americanair call" and "americanair flight".</li>
<li><b>C2 ({hc[1]["size"]} tweets)</b> is the Southwest cluster. Every tweet has southwestair, and the cloud has "southwestair flight", "southwestair best" and "love".</li>
<li><b>C3 ({hc[2]["size"]:,} tweets)</b> is everything else. It has united ({pct(dict(hc[2]["share"])["united"])} of the tweets), usairways ({pct(dict(hc[2]["share"])["usairways"])}) and jetblue ({pct(dict(hc[2]["share"])["jetblue"])}), plus the general words like flight, cancelled, service and thanks.</li>
</ul>
<p><b>Compare and contrast.</b> What is the same is that all three clouds have the same everyday words: flight, get, thanks, help, time. What is different is the airline. C1 and C2 each have one airline name that is much bigger than everything else, and C3 is a mix of the other airlines. So the clustering split the tweets by which airline they were sent to, not by what the tweet is about. I think this is because Jaccard distance only looks at which words two tweets share. The tweets are short, so the airline name is usually the main word two tweets have in common. The clusters are also not even. C3 has {pct(hc[2]["size"] / n)} of the tweets.</p>
</div>

<h2>Step 8: k-Means</h2>
<p>I connected Bag of Words to a k-Means widget and set the number of clusters to 3. The widget shows a warning that sparse data cannot be normalized, which is normal for bag of words data. Then I did the same thing as in step 7, with three Select Rows widgets and three Word Clouds.</p>
{f_km}
{f_km_clouds}
{t_km}
<div class="q">
<p><b>Describing the word clouds.</b></p>
<ul>
<li><b>C1 ({km[0]["size"]:,} tweets)</b> has almost all the tweets. Its cloud looks like the word cloud from step 4. All the airline names are there at about the same size, with flight, thanks and service.</li>
<li><b>C2 ({km[1]["size"]} tweets)</b> is about cancelled flights. {pct(dict(km[1]["share"])["cancelled"])} of the tweets have cancelled and {pct(dict(km[1]["share"])["cancelled flightled"])} have "cancelled flightled". The airline names are small in this cloud, because the tweets are to different airlines.</li>
<li><b>C3 ({km[2]["size"]} tweets)</b> is tiny. All {km[2]["size"]} tweets have the word still, and most have "flight still". For example one says "I missed my outbound flight - can I still use my return ticket?".</li>
</ul>
<p><b>Compare and contrast.</b> C1 is a general cloud with no topic. C2 and C3 are small but they each have a clear topic, and the topic is a word or phrase, not an airline. The sizes are very uneven. {pct(km[0]["size"] / n)} of the tweets are in C1, so k-Means mostly pulled out two small groups and left everything else together.</p>
<p><b>Hierarchical clustering compared to k-Means.</b></p>
{t_cross}
<ul>
<li>They found different things. Hierarchical clustering grouped the tweets by airline (American, Southwest, the rest). k-Means grouped by a repeated phrase (cancelled flights, "still") and ignored the airline.</li>
<li>The two do not agree. Almost all of the tweets in the two small k-Means clusters ({in_hc3} of {km[1]["size"] + km[2]["size"]}) are inside hierarchical C3, and the American and Southwest tweets are all in k-Means C1.</li>
<li>I think the reason is how each one measures distance. Jaccard only checks if a word is in the tweet or not. k-Means uses the real numbers in the table, so a tweet that repeats a rare phrase like "cancelled flightled" gets big values and ends up far from the others.</li>
<li>Both have one big cluster, but hierarchical clustering is more balanced ({", ".join(str(g["size"]) for g in hc)}) than k-Means ({", ".join(str(g["size"]) for g in km)}).</li>
<li>For this data I think the hierarchical clusters are easier to explain, but the k-Means C2 cluster is the most interesting one because it is about a real topic.</li>
</ul>
</div>

<h2>Step 9: Trying other settings</h2>
<p>I changed the distance in the Distances widget and the linkage in Hierarchical Clustering, and looked at the three word clouds each time. I kept Top N = 3 the whole time.</p>
{t_exp}
{f_ward}
{f_cos}
<div class="q">
<p><b>What I noticed.</b></p>
<ul>
<li><b>The linkage changes the result a lot.</b> With single linkage I get two clusters with just {single["sizes"][0]} tweet each and one cluster with the other {single["sizes"][2]:,}, so the word clouds are useless. Average linkage is better ({", ".join(str(s) for s in average["sizes"])}) and it finds a Virgin America cluster and a JetBlue cluster. Ward gives the most even clusters ({", ".join(str(s) for s in ward["sizes"])}), with a JetBlue cluster, a Southwest cluster and the rest.</li>
<li><b>With Jaccard the clusters are always about airlines.</b> Which airlines get their own cluster changes with the linkage (American and Southwest with complete, JetBlue and Southwest with Ward), but it is always the airline name that decides.</li>
<li><b>The distance matters too.</b> With Cosine and complete linkage the two small clusters only have {cosine["sizes"][0]} and {cosine["sizes"][1]} tweets. With Euclidean and complete linkage it is like single linkage, with two clusters of one tweet. Euclidean with Ward ({", ".join(str(s) for s in eucl_ward["sizes"])}) finds a "cancelled" cluster, which looks a lot like what k-Means found. That makes sense because k-Means also uses Euclidean distance.</li>
<li><b>One big cluster every time.</b> No matter what I picked, one cluster had most of the tweets. I think tweets are too short and have too few words in common for the clustering to make three clear groups. Keeping only the 100 most frequent words probably adds to this.</li>
</ul>
<p>So the clusters depend a lot on the settings. Different settings gave me different stories about the same tweets, so I would not trust one clustering without checking the word clouds.</p>
</div>

<h2>The workflow</h2>
<p>This is my whole workflow. I labelled each part with a text box and an arrow, and renamed the widgets so it is clear what each one does.</p>
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
