# 02convert_all.py
# convert ALL Oriented Bounding Box XML annotations into
# YOLO-OBB .txt label files (one line per aircraft object).
#
# YOLO-OBB label format (one line per object):
#   <class_index> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
# where each (x, y) corner is divided by the image width/height
# so all values are between 0 and 1.

import glob
import os
import xml.etree.ElementTree as ET

from PIL import Image          # reads the ACTUAL image dimensions from the JPEG

DATA = "data/MAR20"

# Class names, sorted alphabetically (this defines their index).
# String sorting puts "A10" before "A2" because '1' < '2'.
class_names = ["A1", "A10", "A11", "A12", "A13", "A14", "A15",
               "A16", "A17", "A18", "A19", "A2", "A20", "A3",
               "A4", "A5", "A6", "A7", "A8", "A9"]


def convert_one_xml(id_, DATA, class_names):
    xml_path = os.path.join(DATA, "Annotations", "Oriented Bounding Boxes", id_ + ".xml")
    jpg_path = os.path.join(DATA, "JPEGImages", id_ + ".jpg")

    out_folder = os.path.join(DATA, "labels")
    os.makedirs(out_folder, exist_ok=True)
    out_path = os.path.join(out_folder, id_ + ".txt")

    # ---- READ THE IMAGE SIZE FROM THE JPEG ----
    # The real JPEG dimensions are always correct, so we use them.
    with Image.open(jpg_path) as im:
        width, height = im.size

    # ---- PARSE THE XML ----
    tree = ET.parse(xml_path)
    root = tree.getroot()

    lines = []
    for obj in root.iter('object'):
        name = obj.find('name').text
        class_idx = class_names.index(name)

        # grab the 4 corner points
        rbox = obj.find('robndbox')
        x1 = float(rbox.find('x_left_top').text)
        y1 = float(rbox.find('y_left_top').text)
        x2 = float(rbox.find('x_right_top').text)
        y2 = float(rbox.find('y_right_top').text)
        x3 = float(rbox.find('x_right_bottom').text)
        y3 = float(rbox.find('y_right_bottom').text)
        x4 = float(rbox.find('x_left_bottom').text)
        y4 = float(rbox.find('y_left_bottom').text)

        # normalize all coordinates
        x1, x2, x3, x4 = x1/width, x2/width, x3/width, x4/width
        y1, y2, y3, y4 = y1/height, y2/height, y3/height, y4/height
        x1, x2, x3, x4 = round(x1, 6), round(x2, 6), round(x3, 6), round(x4, 6)
        y1, y2, y3, y4 = round(y1, 6), round(y2, 6), round(y3, 6), round(y4, 6)

        # build one YOLO line
        line = f"{class_idx} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}"
        lines.append(line)

    # ---- WRITE THE OUTPUT FILE ----
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---- DRIVER: loop over all images ----
jpg_files = glob.glob(os.path.join(DATA, "JPEGImages", "*.jpg"))
print("Found", len(jpg_files), "images")

count = 0
for jpg_file in jpg_files:
    id_ = os.path.basename(jpg_file).replace(".jpg", "")
    convert_one_xml(id_, DATA, class_names)
    count += 1
    if count % 500 == 0:                  # progress print every 500
        print("Processed", count, "images")

print("Done. Created", count, ".txt label files.")

# ---- VERIFY ----
made = glob.glob(os.path.join(DATA, "labels", "*.txt"))
print("Label files:", len(made), "vs images:", len(jpg_files))
assert len(made) == len(jpg_files), "Mismatch! Something went wrong."
print("All labels created successfully.")
