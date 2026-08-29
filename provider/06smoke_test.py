"""
06smoke_test.py
===============

A QUICK smoke test of the whole YOLO-OBB pipeline, intended to run on a
CPU machine (like this one) BEFORE you do the real training on the GPU PC.

WHY WE DO THIS
--------------
The real training takes many hours on the GPU. This tiny 2-epoch run on
CPU verifies, in a few minutes, that everything is wired up correctly:
  1. YOLO can load the model (here: untrained yolo11n-obb.yaml, offline).
  2. YOLO reads our data.yaml and finds the train/val images.
  3. YOLO accepts our OBB label format (class + 4 normalized corners).
  4. The training loop starts and completes a couple of steps.

If this passes, we know the SAME code will run on the GPU machine with a
bigger model and more epochs (see training.py).

WHY UNTRAINED / OFFLINE
-----------------------
"yolo11n-obb.yaml" is the model ARCHITECTURE only (a small text file that
ships with ultralytics). No pretrained weights are downloaded, so this
test works even without internet. It does NOT produce a useful model -
it just proves the pipeline works.
"""

from ultralytics import YOLO

# Build an untrained (from-scratch) YOLO11 nano-OBB model.
# - "yolo11n"        = the nano (smallest) variant
# - "-obb"           = oriented bounding box task
# - ".yaml"          = architecture only, NO pretrained weights (offline)
model = YOLO("yolo11n-obb.yaml")

# Train for just 2 epochs on the CPU to validate the pipeline.
model.train(
    data="data/MAR20/data.yaml",   # our dataset config (train/val/test + names)
    epochs=2,                      # tiny - only to prove it works
    imgsz=640,                     # image size YOLO resizes to
    batch=4,                       # small batch (CPU has limited memory)
    device="cpu",                  # this machine has no GPU
    project="runs",
    name="smoke_test",             # results go to runs/smoke_test
    seed=0,                        # reproducible
)

print("SMOKE TEST FINISHED SUCCESSFULLY - pipeline is working.")
