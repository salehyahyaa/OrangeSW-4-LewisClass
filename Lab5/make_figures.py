"""Small extra pictures for the report, and the size of every picture.

Writes to Lab5/results/:
    missed_cv.jpg    training pictures the SqueezeNet models got wrong
    missed_test.jpg  the test picture every model got wrong
    figures.json     width and height of each picture (for the Word file)

    /Applications/Orange.app/Contents/MacOS/python Lab5/make_figures.py
"""
import json
import os

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DATASET = os.path.join(HERE, "submission", "classificationDataset")


def find(name):
    for root, _, files in os.walk(DATASET):
        if name + ".jpg" in files:
            return os.path.join(root, name + ".jpg")
    raise FileNotFoundError(name)


def strip(names, path, height=260):
    pictures = []
    for name in names:
        im = Image.open(find(name)).convert("RGB")
        im = im.resize((round(im.width * height / im.height), height))
        pictures.append(im)
    gap = 8
    width = sum(p.width for p in pictures) + gap * (len(pictures) - 1)
    sheet = Image.new("RGB", (width, height), "white")
    x = 0
    for p in pictures:
        sheet.paste(p, (x, 0))
        x += p.width + gap
    sheet.save(path, quality=88)


def main():
    with open(os.path.join(RESULTS, "results.json")) as f:
        results = json.load(f)
    # training pictures missed by at least three of the five models
    counts = {}
    for model in results["confusion_cv"].values():
        for wrong in model["wrong"]:
            counts[wrong["picture"]] = counts.get(wrong["picture"], 0) + 1
    often = sorted(p for p, n in counts.items() if n >= 3)
    strip(often, os.path.join(RESULTS, "missed_cv.jpg"))
    missed_test = sorted({w["picture"]
                          for m in results["confusion_test"].values()
                          for w in m["wrong"]})
    strip(missed_test, os.path.join(RESULTS, "missed_test.jpg"))

    sizes = {}
    for name in sorted(os.listdir(RESULTS)):
        if name.endswith((".png", ".jpg")):
            with Image.open(os.path.join(RESULTS, name)) as im:
                sizes[name] = list(im.size)
    with open(os.path.join(RESULTS, "figures.json"), "w") as f:
        json.dump({"sizes": sizes, "missed_cv": often,
                   "missed_cv_counts": {p: counts[p] for p in often},
                   "missed_test": missed_test}, f, indent=2)
    print("missed in cross-validation:", {p: counts[p] for p in often})
    print("missed in test:", missed_test)


if __name__ == "__main__":
    main()
