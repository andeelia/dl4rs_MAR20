"""
training.py
===========

Train a YOLO11-OBB model on the MAR20 aircraft dataset.

This is the REAL training script, meant to run on the GPU machine
(NVIDIA RTX 3060). It uses a PRETRAINED model (yolo11m-obb), which was
originally trained on the DOTA aerial/satellite imagery dataset - a very
good starting point for our SAR aircraft task.

WHY PRETRAINED (vs from scratch)
--------------------------------
- The pretrained OBB weights already know how to detect objects.
- Ultralytics AUTO-REBUILDS the detection head to match our 20 classes
  (it reads the "names:" list from data.yaml), so we do not need to do
  anything special for the class count.
- Transfer learning needs far fewer epochs and reaches better accuracy
  than training from scratch.

BEFORE YOU RUN (on the GPU machine)
-----------------------------------
- Copy the whole repo there (it is relocatable: data.yaml has no hardcoded path).
- Install ultralytics:  pip install ultralytics
- The first run downloads yolo11m-obb.pt (needs internet once).

HOW TO RUN
----------
    python training.py
"""

from ultralytics import YOLO

# Load a PRETRAINED YOLO11 Medium OBB model.
# - "yolo11m"          = Medium size (good balance for an RTX 3060, 12GB)
# - "-obb"             = Oriented Bounding Box task
# - ".pt"              = pretrained weights (the model already knows detection)
model = YOLO("yolo11m-obb.pt")

# Train on our dataset.
model.train(
    data="data/MAR20/data.yaml",   # our config (train/val/test + 20 class names)
    epochs=50,                     # full training run (adjust as needed)
    imgsz=640,                     # image size YOLO resizes to
    batch=16,                      # images per step (fits the 3060's 12GB)
    device=0,                      # 0 = first (and only) CUDA GPU on the 3060
    project="runs",
    name="mar20_obb",              # results saved to runs/mar20_obb
    seed=0,                        # reproducible
)

# After training, evaluate on the validation set and show the metrics.
# model.val() returns precision, recall, mAP50, mAP50-95, etc. for free.
metrics = model.val()
print("Validation metrics:")
print(metrics.results_dict)
