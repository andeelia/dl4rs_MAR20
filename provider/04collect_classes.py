"""
04collect_classes.py
==================

Scan every Oriented Bounding Box XML file, collect every class name
(A1..A20), sort them alphabetically, and write the list to classes.txt.

WHY THIS IS NEEDED
------------------
YOLO does not use the class NAME (like "A2") in its label files - it uses
a NUMBER (the class index). Our 02convert_all.py already wrote label files
where each aircraft got a number based on an alphabetical class list.

This script produces that same alphabetical list as a file, so that:
  - classes.txt  (here)   defines the name -> index mapping
  - data.yaml     (in 05make_yaml.py) reads this file to build its "names:"
  - YOLO reads data.yaml  and knows index 0 = "A1", index 1 = "A10", ...

IMPORTANT
---------
The alphabetical ORDER here MUST match the order used inside
02convert_all.py, because that is what already decided the numbers
in the .txt label files. We rely on Python's sorted() giving the exact
same result in both places.
"""

import glob      # find files matching a pattern
import os        # build file paths
import re        # regular expressions to extract <name> tags

DATA = "data/MAR20"
xml_pattern = os.path.join(DATA, "Annotations", "Oriented Bounding Boxes", "*.xml")

# ----------------------------------------------------------------------
# STEP 1: gather every unique class name from all XML files
# ----------------------------------------------------------------------
class_names = set()                 # a set keeps only unique values

for xml in glob.glob(xml_pattern):  # loop over every OBB xml file
    with open(xml) as f:            # open the file as text
        content = f.read()          # read the whole file into one string

    # re.findall returns a list of every match of the pattern.
    # The pattern <name>(\w+)</name> grabs the text inside <name>...</name>.
    found = re.findall(r"<name>(\w+)</name>", content)
    class_names.update(found)       # add them to the set (duplicates removed)

# ----------------------------------------------------------------------
# STEP 2: sort alphabetically
# ----------------------------------------------------------------------
# Python's string sort is lexicographic: 'A10' < 'A2' because '1' < '2'.
# This exact order must match 02convert_all.py's hard-coded list.
class_names = sorted(class_names)

# ----------------------------------------------------------------------
# STEP 3: write one class name per line to classes.txt
# ----------------------------------------------------------------------
out_path = os.path.join(DATA, "classes.txt")
with open(out_path, "w") as f:
    f.write("\n".join(class_names) + "\n")   # join with newlines, add a final one

print("Wrote", len(class_names), "classes to", out_path)
print(class_names)

# sanity check: MAR20 has exactly 20 classes (A1..A20)
assert len(class_names) == 20, "There must be 20 classes!"
