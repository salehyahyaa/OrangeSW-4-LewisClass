"""Build Lab5/submission/classificationDataset from the reviewed photos.

The photos were downloaded from Wikimedia Commons (640 px wide JPEGs) and
checked by eye; PICKED lists the ones kept. About half of each type goes
to train/ and the rest to test/, chosen at random with a fixed seed.

    python Lab5/make_dataset.py <folder with cats/ dogs/ meta.json>
"""
import html
import json
import os
import random
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "submission", "classificationDataset")
PICKED = {
    "cats": [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 14, 16, 17, 18, 19, 20, 22, 25,
             26, 28, 31, 33, 37, 45],
    "dogs": [1, 2, 3, 4, 11, 12, 13, 17, 18, 20, 23, 24, 25, 27, 30, 31, 32,
             33, 34, 38, 42, 44, 45, 48],
}
SEED = 5


def clean(text):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", text))).strip()


def main(src):
    with open(os.path.join(src, "meta.json")) as f:
        meta = json.load(f)
    if os.path.isdir(DATASET):
        shutil.rmtree(DATASET)
    rng = random.Random(SEED)
    lines = ["Lab 5 picture dataset: cats and dogs",
             "All pictures are from Wikimedia Commons (commons.wikimedia.org),",
             "resized to 640 pixels wide. Source, author and license of each:", ""]
    for kind, numbers in PICKED.items():
        order = list(numbers)
        rng.shuffle(order)
        half = len(order) // 2
        for split, chosen in (("train", sorted(order[:half])),
                              ("test", sorted(order[half:]))):
            folder = os.path.join(DATASET, split, kind)
            os.makedirs(folder)
            for i, n in enumerate(chosen, 1):
                old = "%s/%s_%02d.jpg" % (kind, kind, n)
                new = "%s_%s_%02d.jpg" % (kind[:-1], split, i)
                shutil.copyfile(os.path.join(src, old),
                                os.path.join(folder, new))
                m = meta[old]
                lines.append("%s/%s/%s | %s | %s | %s | %s" % (
                    split, kind, new, m["title"], clean(m["artist"]),
                    clean(m["license"]), m["page"]))
            print(split, kind, len(chosen))
    with open(os.path.join(DATASET, "SOURCES.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
