"""
01split_ids.py
============

Reorganize the train / val / test image lists into a clean
70% / 15% / 15% split.

The MAR20 dataset ships with its OWN split in ImageSets/Main/*.txt
(the original authors' choice: 1132 train / 199 val / 2511 test).
Here we OVERWRITE those files so they become:
    70% train, 15% val, 15% test  (of all images).

HOW THIS SCRIPT WORKS
---------------------
1. Gather all image ids. The images are JPEGImages/1.jpg .. JPEGImages/3842.jpg,
   so the ids are the numbers 1..3842 (verified: no gaps).
2. Shuffle them with a FIXED random seed so the split is reproducible:
   running this script again gives the exact same train/val/test sets.
3. Take the first 70% for train, next 15% for val, last 15% for test.
4. OVERWRITE ImageSets/Main/{train,val,test}.txt with the new id lists.

WHY A FIXED SEED
----------------
random.shuffle(ids) without a seed gives a different order every run,
which would produce a different split each time. random.seed(0) makes
the "random" order the same every run, so the split is deterministic
and reproducible.

NOTE: This MUST be run BEFORE 03organize_splits.py, because that script
reads these .txt files to decide where to move the images/labels.
"""

import os
import random

DATA = "data/MAR20"
SPLIT_DIR = os.path.join(DATA, "ImageSets", "Main")

# ----------------------------------------------------------------------
# STEP 1: build the full list of image ids (1..3842)
# ----------------------------------------------------------------------
# List every .jpg in JPEGImages, strip the folder + extension to get the id.
jpg_files = os.listdir(os.path.join(DATA, "JPEGImages"))
all_ids = [int(f.replace(".jpg", "")) for f in jpg_files if f.endswith(".jpg")]
all_ids.sort()                      # now it is 1, 2, 3, ..., 3842

total = len(all_ids)
print("Total images:", total)

# ----------------------------------------------------------------------
# STEP 2: compute the split sizes (70/15/15)
# ----------------------------------------------------------------------
n_train = round(total * 0.70)       # 2689
n_val   = round(total * 0.15)       # 576
# test takes whatever is left over so all images are used exactly once
n_test  = total - n_train - n_val   # 577
print(f"Split sizes -> train: {n_train}, val: {n_val}, test: {n_test}")

# ----------------------------------------------------------------------
# STEP 3: shuffle the ids reproducibly and cut into the three parts
# ----------------------------------------------------------------------
random.seed(0)                      # SAME seed = SAME split every run
random.shuffle(all_ids)             # mix the id order randomly

train_ids = all_ids[:n_train]       # first  2689 ids  -> train
val_ids   = all_ids[n_train:n_train + n_val]   # next   576 -> val
test_ids  = all_ids[n_train + n_val:]         # last   577 -> test

# ----------------------------------------------------------------------
# STEP 4: write the new split files (OVERWRITE the originals)
# ----------------------------------------------------------------------
# We store each split as a dict so the code below just loops over it.
new_splits = {
    "train": train_ids,
    "val":   val_ids,
    "test":  test_ids,
}

for split, ids in new_splits.items():
    out_path = os.path.join(SPLIT_DIR, split + ".txt")
    # Sort each id list for readability (order in the file does not matter).
    ids.sort()

    with open(out_path, "w") as f:
        for id_ in ids:
            f.write(str(id_) + "\n")     # one id per line, e.g. "2\n"

    print("Wrote", len(ids), "ids to", out_path)

# ----------------------------------------------------------------------
# STEP 5: sanity check -- every id is used exactly once, no overlaps
# ----------------------------------------------------------------------
all_split_ids = set(train_ids) | set(val_ids) | set(test_ids)
assert len(all_split_ids) == total, "Some image is missing or duplicated!"
assert len(set(train_ids) & set(val_ids)) == 0, "train/val overlap!"
assert len(set(train_ids) & set(test_ids)) == 0, "train/test overlap!"
assert len(set(val_ids) & set(test_ids)) == 0, "val/test overlap!"
print("OK: all", total, "images split with no overlaps.")
