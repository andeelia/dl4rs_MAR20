"""
05make_yaml.py
============

Generate data.yaml - the configuration file that YOLO reads at training
time to learn where your dataset lives and what the class names are.

WHAT THE FILE LOOKS LIKE
------------------------
    train: images/train
    val: images/val
    test: images/test
    names:
      0: A1
      1: A10
      ...
      19: A9

WHAT EACH FIELD MEANS
---------------------
  (no path line) - We OMIT "path:" on purpose. Ultralytics then uses the
                   folder where data.yaml lives as the dataset root, so the
                   dataset is RELOCATABLE between machines.
  train  - folder containing the training images (images/train)
  val    - folder containing the validation images (images/val)
  test   - folder containing the test images (images/test)
  names  - the class name for each index. Index i = the class on line i
           of classes.txt. This is exactly how YOLO maps label number -> name.

NOTE: The folders it points to (images/train, images/val, ...) are created
by 03organize_splits.py. So run order is:
    1. 01split_ids.py       (build the 70/15/15 split)
    2. 02convert_all.py     (build the .txt labels)
    3. 03organize_splits.py (move images+labels into train/val/test)
    4. 04collect_classes.py (build classes.txt)
    5. 05make_yaml.py       (build data.yaml from classes.txt)  <- THIS script
"""

import os

DATA = "data/MAR20"

# ----------------------------------------------------------------------
# STEP 1: read the class names from classes.txt
# ----------------------------------------------------------------------
# classes.txt has one name per line. The list comprehension keeps every
# non-empty line and strips the trailing newline.
with open(os.path.join(DATA, "classes.txt")) as f:
    class_names = [line.strip() for line in f if line.strip()]

# ----------------------------------------------------------------------
# STEP 2: build the lines of the data.yaml file as a list of strings
# ----------------------------------------------------------------------
lines = []

# We deliberately DO NOT write a "path:" line here.
# Ultralytics computes the dataset root from the FOLDER WHERE THIS
# data.yaml LIVES (it reads the fallback `Path(yaml_file).parent`).
# Since data.yaml is at data/MAR20/data.yaml, "images/train" below resolve
# to data/MAR20/images/train. Omitted path -> the whole dataset is
# RELOCATABLE: copy the repo to the GPU machine and it just works.
# (Writing "path: ." would instead resolve relative to the CURRENT WORKING
# DIRECTORY, which breaks when you run the script from a different folder.)

lines.append("train: images/train")   # where the training images are
lines.append("val: images/val")       # where the validation images are
lines.append("test: images/test")     # where the test images are

lines.append("names:")                # the class-name mapping starts here
for i, name in enumerate(class_names):
    # enumerate gives (0, 'A1'), (1, 'A10'), ... -- exactly the indices.
    lines.append(f"  {i}: {name}")

# Convert the list of lines into one big string with newlines between them.
yaml_text = "\n".join(lines) + "\n"

# ----------------------------------------------------------------------
# STEP 3: write the data.yaml file
# ----------------------------------------------------------------------
yaml_path = os.path.join(DATA, "data.yaml")
with open(yaml_path, "w") as f:
    f.write(yaml_text)

print("Wrote", yaml_path)
print(yaml_text)
