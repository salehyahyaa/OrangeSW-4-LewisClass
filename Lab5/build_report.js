// Build Lab5/submission/lab5.docx - the Word report for Lab 5.
//
// Every number comes from Lab5/results/results.json (written by
// run_workflow.py) and every picture from Lab5/results/.
//
//   NODE_PATH=<folder with the "docx" package>/node_modules \
//       node Lab5/build_report.js
const fs = require("fs");
const path = require("path");
const {
  AlignmentType, BorderStyle, Document, Footer, HeadingLevel, ImageRun,
  Packer, PageNumber, Paragraph, ShadingType, Table, TableCell, TableRow,
  TextRun, VerticalAlign, WidthType,
} = require("docx");

const HERE = __dirname;
const RESULTS = path.join(HERE, "results");
const OUT = path.join(HERE, "submission", "lab5.docx");
const R = JSON.parse(fs.readFileSync(path.join(RESULTS, "results.json")));
const F = JSON.parse(fs.readFileSync(path.join(RESULTS, "figures.json")));

const FONT = "Arial";
const PAGE_W = 9360;            // text width in DXA (6.5 in)
const PX_PER_IN = 96;

// ---------------------------------------------------------------- helpers
function runs(text, base = {}) {
  // **bold** and *italic* inside a string
  return text.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/).filter(Boolean).map((part) => {
    if (part.startsWith("**")) {
      return new TextRun({ ...base, text: part.slice(2, -2), bold: true });
    }
    if (part.startsWith("*")) {
      return new TextRun({ ...base, text: part.slice(1, -1), italics: true });
    }
    return new TextRun({ ...base, text: part });
  });
}
const P = (text, opts = {}) => new Paragraph({
  spacing: { after: 140, line: 288 }, ...opts, children: runs(text),
});
const H1 = (text, opts = {}) => new Paragraph({
  heading: HeadingLevel.HEADING_1, ...opts, children: [new TextRun(text)],
});
const H2 = (text) => new Paragraph({
  heading: HeadingLevel.HEADING_2, children: [new TextRun(text)],
});
const BULLET = (text) => new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  spacing: { after: 80, line: 288 }, children: runs(text),
});
// what the handout asks for in this step
const ASK = (text) => new Paragraph({
  spacing: { before: 60, after: 160, line: 276 },
  indent: { left: 200 },
  border: { left: { style: BorderStyle.SINGLE, size: 18, color: "2E75B6",
                    space: 8 } },
  children: runs(text, { italics: true, color: "404040" }),
});

let figureNo = 0;
let tableNo = 0;
const caption = (label, text) => new Paragraph({
  alignment: AlignmentType.CENTER, spacing: { before: 60, after: 240 },
  children: [
    new TextRun({ text: label + ". ", bold: true, size: 20 }),
    ...runs(text, { size: 20 }),
  ],
});

function imageRun(name, widthIn) {
  const [w, h] = F.sizes[name];
  const width = Math.round(widthIn * PX_PER_IN);
  return new ImageRun({
    type: name.endsWith(".png") ? "png" : "jpg",
    data: fs.readFileSync(path.join(RESULTS, name)),
    transformation: { width, height: Math.round(width * h / w) },
  });
}
function figure(name, widthIn, text) {
  figureNo += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true,
      spacing: { before: 120, after: 0 },
      children: [imageRun(name, widthIn)],
    }),
    caption("Figure " + figureNo, text),
  ];
}

const THIN = { style: BorderStyle.SINGLE, size: 4, color: "A6A6A6" };
const BORDERS = { top: THIN, bottom: THIN, left: THIN, right: THIN };
const NONE = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const NO_BORDERS = { top: NONE, bottom: NONE, left: NONE, right: NONE };

function cell(text, width, opts = {}) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA }, borders: BORDERS,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    shading: opts.fill
      ? { type: ShadingType.CLEAR, color: "auto", fill: opts.fill }
      : undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      children: runs(String(text), { size: 19, bold: !!opts.bold }),
    })],
  });
}
// rows: arrays of strings; the first row is the header.
// highlight(rowIndex, colIndex) may return a fill colour.
function table(widths, rows, text, { numeric = [], highlight } = {}) {
  tableNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const body = rows.map((row, r) => new TableRow({
    tableHeader: r === 0, cantSplit: true,
    children: row.map((value, c) => cell(value, widths[c], {
      bold: r === 0,
      fill: r === 0 ? "DDEBF7" : (highlight && highlight(r, c)) || undefined,
      align: numeric.includes(c) ? AlignmentType.CENTER : AlignmentType.LEFT,
    })),
  }));
  return [
    new Paragraph({
      keepNext: true, spacing: { before: 160, after: 80 },
      children: [
        new TextRun({ text: "Table " + tableNo + ". ", bold: true, size: 20 }),
        ...runs(text, { size: 20 }),
      ],
    }),
    new Table({
      width: { size: total, type: WidthType.DXA }, columnWidths: widths,
      alignment: AlignmentType.CENTER, rows: body,
    }),
    new Paragraph({ spacing: { after: 160 }, children: [] }),
  ];
}
// pictures side by side, in a table without borders
function pictureGrid(names, perRow, widthIn) {
  const colW = Math.floor(PAGE_W / perRow);
  const rows = [];
  for (let i = 0; i < names.length; i += perRow) {
    const cells = [];
    for (let j = 0; j < perRow; j += 1) {
      const name = names[i + j];
      cells.push(new TableCell({
        width: { size: colW, type: WidthType.DXA }, borders: NO_BORDERS,
        margins: { top: 60, bottom: 60, left: 40, right: 40 },
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: name ? [imageRun(name, widthIn)] : [],
        })],
      }));
    }
    rows.push(new TableRow({ cantSplit: true, children: cells }));
  }
  return new Table({
    width: { size: colW * perRow, type: WidthType.DXA },
    columnWidths: Array(perRow).fill(colW), rows,
  });
}

const f3 = (x) => Number(x).toFixed(3);
const pct = (x) => (100 * x).toFixed(1) + "%";
const tag = (m) => m.toLowerCase().replace(/ /g, "_");
const MODELS = ["Neural Network", "Random Forest", "Logistic Regression",
                "kNN", "SVM"];

// ------------------------------------------------------------- the numbers
const sq = R.squeezenet;
const bt = R.best_train;
const cv = R.cv_scores;                 // SqueezeNet, sorted by CA
const bestCv = R.best_cv_scores;        // Inception v3, cross-validation
const test = R.test_scores;             // Inception v3, test pictures
const cmCv = R.confusion_cv;
const cmTest = R.confusion_test;
const exp = R.experiments;
const EMBEDDERS = Object.keys(exp.features);
const byEmbedder = {};
for (const row of exp.rows) {
  (byEmbedder[row.embedder] = byEmbedder[row.embedder] || []).push(row);
}
const settings = byEmbedder[EMBEDDERS[0]].map((r) => [r.model, r.setting]);
const expCa = (embedder, model, setting) => byEmbedder[embedder]
  .find((r) => r.model === model && r.setting === setting).ca;
const summary = EMBEDDERS.map((e) => {
  const cas = byEmbedder[e].map((r) => r.ca);
  const best = Math.max(...cas);
  return {
    embedder: e, features: exp.features[e], best,
    worst: Math.min(...cas),
    mean: cas.reduce((a, b) => a + b, 0) / cas.length,
    nBest: cas.filter((c) => c === best).length,
    bestModels: [...new Set(byEmbedder[e].filter((r) => r.ca === best)
      .map((r) => r.model))],
  };
});
const sOf = (e) => summary.find((s) => s.embedder === e);
const wrongCount = (cm) => cm.counts[0][1] + cm.counts[1][0];
const top = cv[0];
const last = cv[cv.length - 1];
const nTrain = sq.rows;
const allTestSame = test.every((r) => r.CA === test[0].CA);
const missedTest = F.missed_test;

// ------------------------------------------------------------- the report
const body = [];
const add = (...items) => body.push(...items.flat());

add(
  new Paragraph({
    heading: HeadingLevel.TITLE, alignment: AlignmentType.LEFT,
    children: [new TextRun("Lab Exercise #5: Classification of Images")],
  }),
  new Paragraph({
    spacing: { after: 240 },
    children: [new TextRun({
      text: "DATA-20000 Introduction to Data Science  ·  Saleh Yahya",
      color: "595959",
    })],
  }),
  P("In this lab I taught Orange to tell pictures of **cats** from pictures "
    + "of **dogs**. I collected 24 pictures of each animal, turned every "
    + "picture into numbers with an image embedder, trained five "
    + "classification models on half of the pictures and tested them on the "
    + "other half. This report goes through the steps in the same order as "
    + "the lab handout."),
  P("Files I am submitting with this report: the workflow "
    + "**lab5_yahya.ows** and the folder **classificationDataset** with all "
    + "48 pictures. The workflow looks for that folder next to itself, so "
    + "the two should be kept together."),

  P("My results in short:"),
  BULLET("With the **SqueezeNet** embedder the best model (" + top.Model
    + ") got " + pct(top.CA) + " of the training pictures right in 10-fold "
    + "cross-validation."),
  BULLET("The embedder made the biggest difference. With **Inception v3** "
    + "all five models reached 100% in cross-validation."),
  BULLET("On the 24 test pictures that the models had never seen, all five "
    + "models got **" + pct(test[0].CA) + "** right (23 of 24). The one "
    + "miss was a hairless kitten."),

  H1("The whole workflow", { pageBreakBefore: true }),
  P("Figure 1 shows the finished workflow. I added text boxes and arrows "
    + "that say which step each part belongs to. The top half uses the "
    + "SqueezeNet embedder (steps 2 to 5). The bottom half uses the embedder "
    + "that did best in my experiments and scores the models on the test "
    + "pictures (steps 6 and 7)."),
  figure("canvas.png", 6.0,
    "My Orange workflow for Lab 5, with each step labelled."),

  H1("Step 1: Installing Image Analytics"),
  ASK("Install the Image Analytics add-on and restart Orange."),
  P("I added the Image Analytics add-on to Orange (version 0.13.0, on "
    + "Orange " + R.orange_version + "). Once it was installed, the image "
    + "widgets were available. I used two of them in this "
    + "lab: **Import Images** and **Image Embedding**."),

  H1("Step 2: The pictures"),
  ASK("Pick two types of objects, at least 20 pictures of each. Put about "
    + "50% in a train folder and the rest in a test folder, then import the "
    + "training images and embed them with SqueezeNet."),
  P("My two types of objects are **cats** and **dogs**. I found the "
    + "pictures online on Wikimedia Commons, which only hosts pictures that "
    + "are free to reuse. I kept 24 pictures of each animal, so 48 in "
    + "total. I picked pictures where the animal is easy to see, and I "
    + "tried to get a mix of breeds, colors, poses and backgrounds. The "
    + "file SOURCES.txt in the dataset folder lists where each picture came "
    + "from, who took it and its license."),
  figure("examples.jpg", 6.3,
    "Twelve of my training pictures: six cats (top) and six dogs (bottom), "
    + "cropped to squares here to fit on the page."),
  P("I split each animal at random, half into train and half into test. "
    + "Table 1 shows the folders. No picture is in both train and test."),
  table([4200, 1900, 1900], [
    ["Folder", "Type", "Pictures"],
    ["classificationDataset/train/cats", "cats", R.folders["train/cats"]],
    ["classificationDataset/train/dogs", "dogs", R.folders["train/dogs"]],
    ["classificationDataset/test/cats", "cats", R.folders["test/cats"]],
    ["classificationDataset/test/dogs", "dogs", R.folders["test/dogs"]],
  ], "My picture folders. Train and test each get 50% of the pictures.",
  { numeric: [2] }),
  P("The **Import Images** widget reads the train folder. It found "
    + nTrain + " pictures in 2 categories. It uses the name of the "
    + "subfolder (cats or dogs) as the category, so I did not have to label "
    + "anything by hand. I connected it to **Image Embedding** and chose "
    + "**SqueezeNet** as the embedder. SqueezeNet runs on my own computer, "
    + "so the pictures are not sent anywhere."),
  new Paragraph({
    alignment: AlignmentType.CENTER, keepNext: true,
    spacing: { before: 120, after: 0 },
    children: [imageRun("import_images.png", 3.1),
               new TextRun("   "),
               imageRun("image_embedding.png", 2.5)],
  }),
  (() => { figureNo += 1; return caption("Figure " + figureNo,
    "Import Images reading the train folder (left) and Image Embedding "
    + "set to SqueezeNet (right)."); })(),

  H1("Step 3: The features"),
  ASK("Connect Image Embedding to a Data Table. What feature values are "
    + "generated and how many features exist?"),
  P("I connected Image Embedding to a Data Table that I named **Training "
    + "Data** (Figure " + (figureNo + 1) + ")."),
  figure("training_data.png", 6.4,
    "The Training Data table after SqueezeNet. Each row is one picture."),
  P("What I saw in the table:"),
  BULLET("There are **" + sq.rows + " rows**, one for each training "
    + "picture, and no missing data."),
  BULLET("SqueezeNet made **" + sq.features + " features** for every "
    + "picture. They are named " + sq.first_features[0] + " to "
    + sq.last_feature + "."),
  BULLET("Every feature is a number. In my data the values go from "
    + sq.min.toFixed(1) + " to " + sq.max.toFixed(1) + ", with an average "
    + "of " + sq.mean.toFixed(1) + ". For example, the first picture ("
    + sq.first_row.picture + ") starts with "
    + sq.first_row.values.slice(0, 4).map((v) => v.toFixed(2)).join(", ")
    + "."),
  BULLET("There are also " + sq.metas.length + " extra columns that are "
    + "not used for learning: " + sq.metas.join(", ") + "."),
  BULLET("The target is **" + sq.target + "**, with the 2 values "
    + sq.classes.join(" and ") + " (" + sq.class_counts[0] + " and "
    + sq.class_counts[1] + " pictures)."),
  P("The numbers do not mean anything I can read directly. I cannot say "
    + "that n5 is \"has whiskers\". They are what a neural network that was "
    + "already trained on a huge number of photos produces when it looks at "
    + "the picture. Pictures that look alike get similar numbers, and that "
    + "is what lets a normal classification model work on pictures. One "
    + "thing I noticed is that there are far more features (" + sq.features
    + ") than pictures (" + sq.rows + ")."),

  H1("Step 4: Classification models with cross-validation"),
  ASK("Connect the embeddings to different classification models and to "
    + "Test and Score with 10-fold cross-validation. Rank the results by "
    + "accuracy and show them."),
  P("I connected the Training Data to five models: **Neural Network**, "
    + "**Random Forest**, **Logistic Regression**, **kNN** and **SVM**. For "
    + "this step I left every model on Orange's default settings. I only "
    + "turned on the fixed random seed for Random Forest and Neural "
    + "Network, so I get the same numbers every time I open the workflow. "
    + "All five go into **Test and Score**, set to cross-validation with "
    + "10 folds."),
  P("I clicked the CA column (classification accuracy) to sort the models "
    + "from best to worst. Figure " + (figureNo + 1) + " is the widget and "
    + "Table " + (tableNo + 1) + " has the same numbers."),
  figure("test_and_score_cv.png", 6.4,
    "Test and Score with 10-fold cross-validation on the SqueezeNet "
    + "features, sorted by accuracy (CA)."),
  table([700, 2500, 1100, 1100, 1100, 1100, 1100], [
    ["Rank", "Model", "CA", "AUC", "F1", "Precision", "Recall"],
    ...cv.map((r, i) => [i + 1, r.Model, f3(r.CA), f3(r.AUC), f3(r.F1),
                         f3(r.Prec), f3(r.Recall)]),
  ], "Models ranked by accuracy. SqueezeNet features, default settings, "
     + "10-fold cross-validation.", { numeric: [0, 2, 3, 4, 5, 6] }),
  P("**" + top.Model + "** came first with an accuracy of " + f3(top.CA)
    + ", which is " + Math.round(top.CA * nTrain) + " of the " + nTrain
    + " pictures right. " + cv[1].Model + " was second with " + f3(cv[1].CA)
    + ". " + cv[2].Model + " and " + cv[3].Model + " tied at "
    + f3(cv[2].CA) + ", and **" + last.Model + "** was last with "
    + f3(last.CA) + " (" + Math.round(last.CA * nTrain) + " of " + nTrain
    + ")."),
  P("All five models do much better than guessing, which would be about "
    + "0.5 with two equal groups. So the SqueezeNet numbers really do carry "
    + "information about cat versus dog. But I do not want to read too "
    + "much into the order. With only " + nTrain + " pictures, one picture "
    + "is worth about " + (100 / nTrain).toFixed(1) + " percentage points, "
    + "so the gap between first and last place is just "
    + Math.round((top.CA - last.CA) * nTrain) + " pictures."),

  H1("Step 5: Confusion matrices"),
  ASK("Connect Test and Score to a Confusion Matrix. Examine the matrices "
    + "for the different models and interpret them."),
  P("I connected Test and Score to a **Confusion Matrix** widget and "
    + "clicked through the five models in its list. The rows are what the "
    + "picture really is and the columns are what the model said. The blue "
    + "cells are correct and the red cells are mistakes."),
  pictureGrid(MODELS.map((m) => "confusion_squeezenet_cv_" + tag(m) + ".png"),
              2, 3.15),
  (() => { figureNo += 1; return caption("Figure " + figureNo,
    "Confusion matrices for the five models (SqueezeNet features, 10-fold "
    + "cross-validation). The selected model is highlighted in each "
    + "list."); })(),
  table([2500, 1500, 1700, 1700, 1500], [
    ["Model", "Cats right", "Cats called dogs", "Dogs called cats",
     "Dogs right"],
    ...cv.map((r) => {
      const c = cmCv[r.Model].counts;
      return [r.Model, c[0][0], c[0][1], c[1][0], c[1][1]];
    }),
  ], "The confusion matrices as numbers (12 cats and 12 dogs for each "
     + "model).", { numeric: [1, 2, 3, 4] }),
  P("How I read them:"),
  BULLET("**" + top.Model + "** made the fewest mistakes, "
    + wrongCount(cmCv[top.Model]) + " in total: "
    + cmCv[top.Model].counts[0][1] + " cats called dogs and "
    + cmCv[top.Model].counts[1][0] + " dog called a cat."),
  BULLET("**Random Forest** was the most lopsided. It got "
    + cmCv["Random Forest"].counts[0][0] + " of 12 cats right but called "
    + cmCv["Random Forest"].counts[1][0] + " of the 12 dogs cats. So when "
    + "it is unsure it leans toward cat."),
  BULLET("Four of the five models made more \"dog called a cat\" mistakes "
    + "than \"cat called a dog\" mistakes. Only SVM went the other way. My "
    + "guess is that the dogs in my set are more varied (very different "
    + "breeds, sizes and coats) than the cats, so they are harder to pin "
    + "down."),
  BULLET("The models mostly miss the **same pictures**. Two pictures were "
    + "missed by all five models and three more by at least three models "
    + "(Figure " + (figureNo + 1) + "). So the mistakes are about hard "
    + "pictures more than about which model I use."),
  figure("missed_cv.jpg", 6.4,
    "The training pictures that at least three of the five models got "
    + "wrong. The two cats were called dogs and the three dogs were called "
    + "cats."),
  P("Looking at these, I can see why they are hard. Both cats are lying "
    + "down asleep, and one of them is small in the "
    + "frame with a big red carpet around it. The beagle is asleep and "
    + "curled up, so its nose and ears do not stand out. The other two dogs "
    + "have pointed ears or a fluffy face, which is a bit cat-like."),

  H1("Step 6: Experiments to get the best accuracy"),
  ASK("Try different embedders, different classification models and "
    + "different parameters within each model. Provide a table summarizing "
    + "the experiments."),
  P("I changed three things: the **embedder**, the **model**, and the "
    + "**settings** inside each model. I tried " + EMBEDDERS.length
    + " embedders and " + settings.length + " model set-ups for each one, "
    + "which is " + exp.rows.length + " experiments. Every experiment is "
    + "scored the same way as in step 4, with 10-fold cross-validation on "
    + "the " + nTrain + " training pictures. I did not look at the test "
    + "pictures here, so they stay a fair test for step 7."),
  P("One thing to know: SqueezeNet is the only embedder that runs on my "
    + "computer. For the others, Orange sends the pictures to its own "
    + "server and gets the numbers back, so they need an internet "
    + "connection."),
  table([1700, 1300, 1300, 1300, 1300, 2460], [
    ["Embedder", "Features", "Best CA", "Average CA", "Worst CA",
     "Model(s) with the best CA"],
    ...summary.map((s) => [s.embedder, s.features, f3(s.best), f3(s.mean),
                           f3(s.worst), s.nBest > 12
                             ? "almost all (" + s.nBest + " of "
                               + settings.length + " set-ups)"
                             : s.bestModels.join(", ")]),
  ], "Summary of the experiments by embedder (" + settings.length
     + " model set-ups each, 10-fold cross-validation accuracy).",
  { numeric: [1, 2, 3, 4],
    highlight: (r) => (summary[r - 1].embedder === "Inception v3"
      ? "E2EFDA" : undefined) }),
  table([1500, 2760, 850, 850, 850, 850, 850, 850], [
    ["Model", "Settings", ...EMBEDDERS.map((e) => e.replace(" v3", ""))],
    ...settings.map(([m, s]) => [m, s,
      ...EMBEDDERS.map((e) => f3(expCa(e, m, s)))]),
  ], "All " + exp.rows.length + " experiments. Each number is the "
     + "cross-validated accuracy (CA). Green cells are 1.000.",
  { numeric: [2, 3, 4, 5, 6, 7],
    highlight: (r, c) => (c >= 2
      && expCa(EMBEDDERS[c - 2], settings[r - 1][0], settings[r - 1][1]) === 1
      ? "E2EFDA" : undefined) }),
  P("What I learned from the experiments:"),
  BULLET("**The embedder matters the most.** With **Inception v3** almost "
    + "every model got every training picture right (CA 1.000 in "
    + sOf("Inception v3").nBest + " of " + settings.length + " set-ups, "
    + "and never below " + f3(sOf("Inception v3").worst) + "). SqueezeNet "
    + "never got above " + f3(sOf("SqueezeNet").best) + "."),
  BULLET("VGG-16 and VGG-19 were in between. They reached "
    + f3(sOf("VGG-16").best) + " and " + f3(sOf("VGG-19").best)
    + " at best, but they were less steady from model to model. kNN in "
    + "particular did poorly with them."),
  BULLET("**Painters** and **DeepLoc** were bad choices, and that makes "
    + "sense. Painters was trained to tell painters apart from their "
    + "artwork and DeepLoc was trained on pictures of yeast cells. DeepLoc "
    + "averaged " + f3(sOf("DeepLoc").mean) + ", which is no better than "
    + "guessing. An embedder only helps if it was trained on pictures like "
    + "mine."),
  BULLET("**The model and its settings mattered much less.** With "
    + "SqueezeNet, changing settings moved the accuracy by one or two "
    + "pictures at most. Two changes that did help a little: more trees in "
    + "the Random Forest (with Inception v3, VGG-16 and VGG-19, 10 trees "
    + "was the weakest forest), and tanh instead of ReLu in the Neural "
    + "Network (0.792 to 0.833 on SqueezeNet)."),
  BULLET("Using just 1 neighbor in kNN was usually worse than 5, because "
    + "one odd picture can then decide the answer."),
  P("So for step 7 I use **Inception v3** as the embedder. For the models "
    + "I kept the default settings except for two that the experiments "
    + "pointed to: Random Forest with **50 trees** instead of 10, and SVM "
    + "with a **linear kernel** instead of RBF. With these settings all "
    + "five models score 1.000 in cross-validation (Figure "
    + (figureNo + 1) + ")."),
  figure("test_and_score_1_cv.png", 6.4,
    "Cross-validation with Inception v3 features and the chosen settings. "
    + "All five models get every training picture right."),

  H1("Step 7: Testing on the test pictures"),
  ASK("Import the test data with a new Import Images widget, embed it with "
    + "the embedder that achieved the highest accuracy, connect it to Test "
    + "and Score as Test Data and select \"Test on test data\". Record and "
    + "interpret the results."),
  P("I added a second Import Images widget, named **Import Images "
    + "(test)**, that reads the test folder (" + R.best_test.rows
    + " pictures). It goes into its own Image Embedding widget set to "
    + "**Inception v3**. The training pictures in this part of the "
    + "workflow also go through Inception v3, so train and test use the "
    + "same embedder and both have " + bt.features + " features. I "
    + "connected the test embeddings to the **Test Data** input of Test "
    + "and Score and chose **Test on test data**. Now each model learns "
    + "from the " + bt.rows + " training pictures and is scored on "
    + R.best_test.rows + " pictures it has never seen."),
  figure("test_and_score_1_test.png", 6.4,
    "Test and Score set to \"Test on test data\" (Inception v3 features)."),
  table([2600, 1500, 1500, 1500, 1500], [
    ["Model", "CA in cross-validation", "CA on test data", "Test pictures "
     + "right", "AUC on test data"],
    ...test.map((r) => [r.Model,
      f3(bestCv.find((b) => b.Model === r.Model).CA), f3(r.CA),
      Math.round(r.CA * R.best_test.rows) + " of " + R.best_test.rows,
      f3(r.AUC)]),
  ], "Final results with Inception v3: cross-validation on the training "
     + "pictures next to the score on the test pictures.",
  { numeric: [1, 2, 3, 4] }),
  pictureGrid(["confusion_best_test_logistic_regression.png",
               "missed_test.jpg"], 2, 3.0),
  (() => { figureNo += 1; return caption("Figure " + figureNo,
    "Left: the confusion matrix on the test pictures (shown for Logistic "
    + "Regression; the other four models have exactly the same matrix). "
    + "Right: the one test picture that was missed."); })(),
  P("How I read the results:"),
  BULLET(allTestSame
    ? "All five models got the same accuracy on the test pictures: **"
      + f3(test[0].CA) + "**, or " + Math.round(test[0].CA * 24)
      + " of 24 right (" + pct(test[0].CA) + "). Since they tie, I cannot "
      + "pick a single best model from this test. Any of them works."
    : "The best model on the test pictures was **" + test[0].Model
      + "** with " + f3(test[0].CA) + "."),
  BULLET("Every one of the 12 dogs was classified correctly. "
    + cmTest[test[0].Model].counts[0][0] + " of the 12 cats were correct "
    + "and " + cmTest[test[0].Model].counts[0][1] + " cat was called a "
    + "dog."),
  BULLET("All five models missed the **same picture** (" + missedTest[0]
    + "). It is a hairless Sphynx kitten. None of my 12 training cats is "
    + "hairless, and with its wrinkled skin and big ears it does not look "
    + "like any of them. So the models had no example to learn this kind "
    + "of cat from."),
  BULLET("The test accuracy (" + f3(test[0].CA) + ") is a little lower "
    + "than the cross-validation accuracy (1.000). That is normal, since "
    + "the test pictures are new to the models. The drop is only one "
    + "picture, so the models did not just memorize the training pictures. "
    + "They carry over to new ones."),
  BULLET("The AUC is 1.000 for four of the models. That means that even "
    + "for the missed kitten, these models still ranked it as more "
    + "cat-like than every dog. It just fell on the wrong side of the "
    + "50% line."),

  H1("What I take away from this lab"),
  BULLET("Orange can classify pictures with very little work, because the "
    + "embedder does the hard part of turning a picture into useful "
    + "numbers."),
  BULLET("Picking the right embedder was worth far more than tuning the "
    + "models. Going from SqueezeNet to Inception v3 took the best "
    + "cross-validated accuracy from " + f3(sOf("SqueezeNet").best)
    + " to 1.000."),
  BULLET("On pictures the models had never seen, I got " + pct(test[0].CA)
    + " accuracy with only 12 training pictures of each animal."),
  BULLET("My data set is small, so each picture counts for about 4 "
    + "percentage points. With more pictures, and more unusual ones like "
    + "hairless cats in the training set, the numbers would be more "
    + "trustworthy."),
);

const doc = new Document({
  creator: "Saleh Yahya",
  title: "Lab Exercise #5: Classification of Images",
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
      title: {
        run: { font: FONT, size: 40, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 0, after: 80 }, keepNext: true },
      },
      heading1: {
        run: { font: FONT, size: 28, bold: true, color: "1F3864" },
        paragraph: { spacing: { before: 320, after: 120 }, keepNext: true },
      },
      heading2: {
        run: { font: FONT, size: 24, bold: true, color: "2E75B6" },
        paragraph: { spacing: { before: 200, after: 100 }, keepNext: true },
      },
    },
  },
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: "bullet", text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 540, hanging: 270 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({
            children: ["Saleh Yahya  ·  Lab 5  ·  Page ",
                       PageNumber.CURRENT],
            size: 18, color: "7F7F7F",
          })],
        })],
      }),
    },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  console.log("wrote", OUT, Math.round(buffer.length / 1024) + " KB,",
              figureNo, "figures,", tableNo, "tables");
});
