# convert_one.py
# ==============
# Phase 2 teaching script: convert ONE Oriented Bounding Box XML file
# into a YOLO-OBB .txt label file.
#
# YOLO-OBB label format (one line per aircraft object):
#     <class_index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
# where each (x, y) corner is divided by the image width/height so that
# all coordinate values are between 0 and 1 (this is called "normalizing").
#
# This is the *building block* version: it handles only image "2" on
# purpose, so you can hand-check the result before scaling up to the
# whole dataset (which 02convert_all.py does).

import os                              # build file paths + create folders
import xml.etree.ElementTree as ET     # Python's built-in XML parser

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------
id_ = "2"                                       # the image number we convert
DATA = "data/MAR20"

# path to the Oriented Bounding Box XML annotation for this image
xml_path = os.path.join(DATA, "Annotations", "Oriented Bounding Boxes", id_ + ".xml")

# where we will write the resulting label file
out_folder = os.path.join(DATA, "labels")
os.makedirs(out_folder, exist_ok=True)          # create folder if missing
out_path = os.path.join(out_folder, id_ + ".txt")

print("Reading .xml:", xml_path)

# ----------------------------------------------------------------------
# PARSE THE XML
# ----------------------------------------------------------------------
# ET.parse() loads the whole file, .getroot() gives the top <annotation>.
tree = ET.parse(xml_path)
root = tree.getroot()

# ----------------------------------------------------------------------
# READ IMAGE SIZE (for normalization)
# ----------------------------------------------------------------------
# NOTE: This version reads the size from the XML. Some XML files wrongly
# say 0x0 (a data bug), which caused a ZeroDivisionError. 02convert_all.py
# instead reads the real size from the JPEG to avoid that problem.
size = root.find('size')
width = int(size.find('width').text)     # .text -> string, int() -> number
height = int(size.find('height').text)
print("Image size:", width, "x", height)

# ----------------------------------------------------------------------
# CLASS NAMES -> INDEX MAPPING
# ----------------------------------------------------------------------
# YOLO uses a NUMBER, not the name. List the 20 classes in alphabetical
# order (same as 02convert_all.py and 04collect_classes.py).
# Note the string sort: 'A10' < 'A2' because '1' < '2'.
# This list MUST contain all 20 names, including A20 and A3!
class_names = ["A1", "A10", "A11", "A12", "A13", "A14", "A15",
               "A16", "A17", "A18", "A19", "A2", "A20", "A3",
               "A4", "A5", "A6", "A7", "A8", "A9"]

# ----------------------------------------------------------------------
# LOOP OVER EACH OBJECT AND BUILD A YOLO LINE
# ----------------------------------------------------------------------
lines = []                              # collect all output lines here

for obj in root.iter('object'):         # 'object' = one aircraft annotation
    name = obj.find('name').text        # class name, e.g. "A2"
    class_idx = class_names.index(name) # turn the name into a number

    # read the 4 corner points from <robndbox>.
    # The tag order already matches YOLO's required order:
    #   left_top, right_top, right_bottom, left_bottom
    rbox = obj.find('robndbox')
    x1 = float(rbox.find('x_left_top').text)
    y1 = float(rbox.find('y_left_top').text)
    x2 = float(rbox.find('x_right_top').text)
    y2 = float(rbox.find('y_right_top').text)
    x3 = float(rbox.find('x_right_bottom').text)
    y3 = float(rbox.find('y_right_bottom').text)
    x4 = float(rbox.find('x_left_bottom').text)
    y4 = float(rbox.find('y_left_bottom').text)

    # normalize: divide every x by width, every y by height,
    # then round to 6 decimals to keep the file clean.
    x1, x2, x3, x4 = x1/width, x2/width, x3/width, x4/width
    y1, y2, y3, y4 = y1/height, y2/height, y3/height, y4/height
    x1, x2, x3, x4 = round(x1, 6), round(x2, 6), round(x3, 6), round(x4, 6)
    y1, y2, y3, y4 = round(y1, 6), round(y2, 6), round(y3, 6), round(y4, 6)

    # build one YOLO line and add it to the list
    line = f"{class_idx} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
    lines.append(line)

    print("Converted:", name, "-> class index", class_idx)

# ----------------------------------------------------------------------
# WRITE THE OUTPUT FILE
# ----------------------------------------------------------------------
# join the lines with newlines, then add a trailing newline at the end.
with open(out_path, "w") as f:
    f.write("\n".join(lines) + "\n")

print("Wrote", len(lines), "object(s) to:", out_path)

# ----------------------------------------------------------------------
# READ THE FILE BACK AND PRINT IT (so you can verify what we wrote)
# ----------------------------------------------------------------------
print("----- contents of", out_path, "-----")
print(open(out_path).read())
