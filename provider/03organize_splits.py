"""
03organize_splits.py
==================

Move the label .txt files AND the image .jpg files into the standard
YOLO directory layout, split by train / val / test:

    data/MAR20/
      images/
        train/  val/  test/     <- the .jpg image files
      labels/
        train/  val/  test/     <- the .txt label files

YOLO expects each image to have a matching label file with the SAME name
in the matching split folder. For example:
    images/train/2.jpg  <->  labels/train/2.txt

HOW THIS SCRIPT WORKS
---------------------
1. Read the id list for each split from ImageSets/Main/<split>.txt.
   (Those files were created by 01split_ids.py with the 70/15/15 split.)
2. For every id, MOVE both its .jpg and its .txt file into the matching
   split folder.

NOTE: This must be run AFTER:
  - 01split_ids.py      (so the split .txt files are up to date)
  - 02convert_all.py    (so the label .txt files already exist)
"""

import os
import shutil                       # shutil.move() lets us MOVE a file

DATA = "data/MAR20"

# ----------------------------------------------------------------------
# STEP 1: read each split's id list from ImageSets/Main/<split>.txt
# ----------------------------------------------------------------------
# We build a dictionary:  {"train": set_of_ids, "val": set_of_ids, ...}
# A set is used because membership checks are fast and ids are unique.
splits = {}

for split in ["train", "val", "test"]:
    ids = set()                     # an empty set for this split
    split_file = os.path.join(DATA, "ImageSets", "Main", split + ".txt")

    with open(split_file) as f:
        for line in f:              # read the file one line at a time
            line = line.strip()     # remove the trailing newline / spaces
            if line:                # ignore empty lines
                ids.add(line)       # ids are strings, e.g. "2"

    splits[split] = ids             # store this split's set of ids

# ----------------------------------------------------------------------
# STEP 2: move each image + label into its split folder
# ----------------------------------------------------------------------
for split, ids in splits.items():
    # `split` is the folder name ("train", "val" or "test")
    # `ids`   is the set of image ids for this split

    # create the destination folders if they do not exist yet
    # exist_ok=True means: don't error if the folder is already there
    os.makedirs(os.path.join(DATA, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATA, "labels", split), exist_ok=True)

    for id_ in ids:
        # ---- move the LABEL (.txt) ----
        src_lab = os.path.join(DATA, "labels", id_ + ".txt")
        dst_lab = os.path.join(DATA, "labels", split, id_ + ".txt")

        if os.path.exists(src_lab):
            shutil.move(src_lab, dst_lab)   # MOVE the file
        else:
            print("WARNING: no label file for id", id_)

        # ---- move the IMAGE (.jpg) ----
        src_img = os.path.join(DATA, "JPEGImages", id_ + ".jpg")
        dst_img = os.path.join(DATA, "images", split, id_ + ".jpg")

        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)   # MOVE the file
        else:
            print("WARNING: no image file for id", id_)

    print(split, "moved", len(ids), "images + labels")
