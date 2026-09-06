"""
training.py
===========

Train a YOLO11-OBB model on the MAR20 aircraft dataset.

It uses a PRETRAINED model (yolo11m-obb), which was
originally trained on the DOTA aerial/satellite imagery dataset.

WHY PRETRAINED (vs from scratch)
--------------------------------
- The pretrained OBB weights already know how to detect objects.
- Ultralytics AUTO-REBUILDS the detection head to match our 20 classes
  (it reads the "names:" list from data.yaml), so we do not need to do
  anything special for the class count.
- Transfer learning needs far fewer epochs and reaches better accuracy
  than training from scratch.

HYPERPARAMETER TUNING
---------------------
This script has two phases:

1. TUNING: model.tune() runs many short training trials with different
   hyperparameters, evaluates each one, and saves the best configuration
   to runs/tune/best_hyperparameters.yaml.

2. FINAL TRAINING: model.train() runs the full training with those tuned
   hyperparameters spliced into the call (via **best_hyp). A FRESH,
   UNTUNED pretrained model is loaded so the tuned hyperparameters are
   validated on a clean model, not one already weighted by tuning trials.

FLOW:
    SMOKE_TEST=True   -> runs a 2-trial tune sanity check, then exits
                         (verifies the tune->yaml pipeline works).
    SMOKE_TEST=False  -> full tune (TUNE_ITERATIONS x TUNE_EPOCHS) then
                         final training (FINAL_EPOCHS).

HOW TO RUN
----------
    python training.py
"""

from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils import LOGGER, YAML

# --- Configuration -----------------------------------------------------------
FINAL_EPOCHS    = 50    # final training epochs
TUNING          = False  # master on/off for the whole tune workflow
SMOKE_TEST      = True # True = run ONLY a 2-trial tune sanity check then exit
TUNE_EPOCHS     = 20    # epochs per tuning trial
TUNE_ITERATIONS = 8     # number of tuning trials
OPTIMIZER       = "AdamW"  # fixed optimizer during tuning (not mutated)

# What the tuner is allowed to mutate. Narrowing the search space to the
# highest-impact hyperparameters makes a small tuning budget far more
# efficient than mutating all ~24 defaults at once.
SEARCH_SPACE = {
    "lr0": (1e-5, 1e-2),   # initial learning rate
    "lrf": (0.01, 1.0),    # final learning rate as fraction of lr0
    "momentum": (0.7, 0.98),
    "weight_decay": (0.0, 0.001),
    "box": (1.0, 20.0),    # box loss gain
    "cls": (0.1, 4.0),     # classification loss gain
}

DATA       = "data/MAR20/data.yaml"
SMOKE_DATA = "data/smoke_dataset/data.yaml"
IMGSZ  = 640
BATCH  = 16
DEVICE = 0
PROJECT = "runs"
TUNE_NAME  = "tune"      # tuning results go to runs/tune/
FINAL_NAME = "mar20_obb_raw_tuned" # final results go to runs/mar20_obb/
TUNE_YAML  = Path("runs/tune/best_hyperparameters_raw.yaml")


def main():
    if SMOKE_TEST:
        LOGGER.info(
            f"\nSMOKE TEST mode: running a 2-trial tune sanity check "
            f"({TUNE_EPOCHS} epochs/trial) to verify the pipeline, skipping final training."
        )
        model = YOLO("yolo11m-obb.pt")
        model.tune(
            data=SMOKE_DATA,
            epochs=TUNE_EPOCHS,
            iterations=2,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT,
            name=TUNE_NAME,
            optimizer=OPTIMIZER,
            space=SEARCH_SPACE,
            plots=False,
            save=False,
        )
        if TUNE_YAML.exists():
            LOGGER.info(
                "\nSmoke test PASSED: best_hyperparameters.yaml was written. "
                "Now set SMOKE_TEST=False and rerun for the full tune + final training."
            )
        else:
            LOGGER.error(
                "\nSmoke test FAILED: no best_hyperparameters.yaml produced. "
                "Check the tuning logs for failed trials before the real run."
            )
        return

    # --- Phase 1: hyperparameter tuning ------------------------------------
    if TUNING:
        #model = YOLO("yolo11m-obb.pt")
        model = YOLO("yolo11m-obb.yaml")
        model.tune(
            data=DATA,
            epochs=TUNE_EPOCHS,
            iterations=TUNE_ITERATIONS,
            imgsz=IMGSZ,
            batch=BATCH,
            device=DEVICE,
            project=PROJECT,
            name=TUNE_NAME,
            optimizer=OPTIMIZER,
            space=SEARCH_SPACE,
            plots=True,
            save=True,
        )

    # --- Phase 2: load tuned hyperparameters -------------------------------
    best_hyp = {}
    if TUNE_YAML.exists():
        best_hyp = YAML(TUNE_YAML)
        LOGGER.info(f"Loaded tuned hyperparameters from {TUNE_YAML}: {best_hyp}")
    else:
        LOGGER.warning(
            f"No {TUNE_YAML} found. Training with default hyperparameters "
            "instead. Run tuning first (TUNING=True)."
        )

    # --- Phase 3: final training with tuned hyperparameters ----------------
    # Load a FRESH, UNTUNED pretrained model so the tuned hyperparameters are
    # validated on a clean model (not one already weighted by tuning trials).
    #model = YOLO("yolo11m-obb.pt")
    model = YOLO("yolo11m-obb.yaml")
    model.train(
        data=DATA,
        epochs=FINAL_EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=FINAL_NAME,
        seed=0,                # reproducible
        **best_hyp,            # tuned hyperparameters override defaults last
    )

    # After training, evaluate on the validation set and show the metrics.
    # model.val() returns precision, recall, mAP50, mAP50-95, etc. for free.
    metrics = model.val()
    print("Validation metrics:")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
