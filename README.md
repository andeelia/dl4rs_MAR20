# AI Approaches in Earth Observation

Deep learning detection of military aircraft in RGB imagery using
**YOLO11-OBB** on the **MAR20** dataset.

The project converts MAR20's PASCAL-VOC oriented annotations into the YOLO-OBB
format, splits the data into train/val/test, and trains a YOLO11 Oriented
Bounding Box model to detect the (rotated) aircraft.

---

## 1. Dataset: MAR20 – Military Aircraft

- **Source:** https://gcheng-nwpu.github.io/ (MAR20.zip) — military aircraft in RGB images.
  See `Datenquelle` for all links.
- **Size:** 3842 JPG images, each paired with a PASCAL-VOC XML annotation.
- **Annotations:** each image has both an Oriented Bounding Box (OBB) and a
  Horizontal Bounding Box (HBB). This project uses the **oriented** boxes.
- **Classes (20):** `A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15 A16 A17 A18 A19 A20`
- **Known data quirk:** 4 XMLs (ids 334, 338, 409, 443) have a broken `0x0`
  `<size>` element, so image dimensions are read from the JPEG via PIL instead
  of trusting the XML.

---

## 2. Requirements / setup

- Python **3.13** (see `pyproject.toml`, managed with `uv`).
- Key dependencies:
  - `ultralytics>=8.4.132`
  - `torch`, `torchvision`
  - `numpy`, `matplotlib`, `scikit-image`, `albumentations`, `tqdm`

```bash
uv sync          # or: pip install -e .
```

---

## 3. Repository layout

```
dl4rs-MAR20/
├── provider/                 # data-preparation pipeline scripts
│   ├── 01split_ids.py        # train/val/test split (70/15/15, seed 0)
│   ├── 02convert_all.py      # OBB-VOC -> YOLO .txt for all images
│   ├── 03organize_splits.py  # move images+labels into images/ and labels/
│   ├── 04collect_classes.py  # build classes.txt
│   ├── 05make_yaml.py        # generate data.yaml
│   └── 06smoke_test.py       # legacy quick CPU pipeline validation
├── data/MAR20/               # the dataset (see section 5)
├── training.py               # real YOLO11-OBB training (GPU) + SMOKE_TEST mode
├── weights/                  # pretrained OBB weights (yolo11m-obb.pt)
├── checkpoints/              # epoch checkpoints from base-code runs
├── runs/                     # training/tuning outputs
├── Datenquelle               # record of all dataset source links
├── todo.txt                  # future-work notes
├── base code/                # original UNet pipeline
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 4. Pipeline: from raw data to YOLO format

All data-prep steps live in `provider/` and are meant to be run **once** in the
correct order. The pipeline is expressed as a series of short, well explained
scripts, one concept each.

| Step | Script                 | What it does |
|------|------------------------|--------------|
| 1    | `01split_ids.py`       | 70/15/15 train/val/test split by image id (seed 0); overwrites `ImageSets/Main/{train,val,test}.txt`. |
| 2    | `02convert_all.py`     | Converts every oriented XML annotation to YOLO-OBB `.txt` (stdlib `xml.etree.ElementTree`; reads JPEG size via PIL to handle the broken-`<size>` bug). |
| 3    | `03organize_splits.py` | Moves images and labels into the standard `images/{train,val,test}` + `labels/{...}` layout. |
| 4    | `04collect_classes.py` | Collects the class list into `data/MAR20/classes.txt`. |
| 5    | `05make_yaml.py`       | Generates the relocatable `data.yaml` (no `path:` line, so paths resolve relative to the YAML's own directory). |
| 6    | `06smoke_test.py`      | Legacy sanity test: trains an untrained `yolo11n-obb.yaml` on CPU for 2 epochs to validate the pipeline end-to-end. |

---

## 5. Final data layout

After the pipeline runs, `data/MAR20/` looks like this:

```
data/MAR20/
├── data.yaml                 # relocatable YOLO config (20 classes)
├── classes.txt               # A1 ... A20
├── ImageSets/Main/           # {train,val,test}.txt id lists
├── Annotations/
│   ├── Horizontal Bounding Boxes/
│   └── Oriented Bounding Boxes/
├── images/
│   ├── train/   (2689)
│   ├── val/     (576)
│   └── test/    (577)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

**YOLO-OBB label format** (one line per object, space-separated):

```
class x1 y1 x2 y2 x3 y3 x4 y4
```

- `class` = class index (0–19)
- `x1..x4, y1..y4` = normalized corner coordinates of the rotated box

**Split summary (70/15/15, seed 0):**

| Split | Images |
|-------|--------|
| train | 2689   |
| val   | 576    |
| test  | 577    |
| **Total** | **3842** |

---

## 6. Training

`training.py` is the real training script. It can use either a **pretrained**
model or an **un-pretrained** (architecture-only) one, and optionally runs
hyperparameter **tuning** before the final training run.

```bash
python training.py
```

Key settings inside `training.py`:

| Parameter      | Value                                   | Reason                                                                             |
|----------------|-----------------------------------------|------------------------------------------------------------------------------------|
| `model`        | `yolo11m-obb.pt  ` / `yolo11m-obb.yaml` | (Un-) Pretrained Medium OBB; ultralytics auto-rebuilds the head to our 20 classes. |
| `data`         | `data/MAR20/data.yaml`                  | Our config.                                                                        |
| `epochs`       | 50                                      | Full run.                                                                          |
| `imgsz`        | 640                                     | Image size YOLO resizes to.                                                        |
| `batch`        | 16                                      | Fits the 12 GB VRAM.                                                               |
| `device`       | 0                                       | First CUDA GPU.                                                                    |
| `project`      | `runs`                                  | Output dir.                                                                        |
| `seed`         | 0                                       | Reproducibility.                                                                   |

The script has two phases controlled by flags at the top:

- **Tuning** (`TUNING=True`): `model.tune()` runs short trials over a
  `SEARCH_SPACE` of key hyperparameters and saves the best configuration to
  `runs/tune/best_hyperparameters.yaml`.
- **Final training**: loads a fresh, untuned model and trains with
  the tuned hyperparameters spliced in via `**best_hyp`.

**Model choice (pretrained vs un-pretrained):** `training.py` loads the model
in two places (the tuning phase and the final-training phase), each via a
commented `YOLO(...)` line:

```python
model = YOLO("yolo11m-obb.pt")   # pretrained weights (transfer learning)
model = YOLO("yolo11m-obb.yaml") # architecture only – un-pretrained (from scratch)
```

The two lines sit next to each other with one commented out and the other
active. To switch modes, un-comment the one you want to use and comment out the
other. Using the `.yaml` version (`yolo11m-obb.yaml`) trains from scratch with
no pretrained weights; the `.pt` version starts from pretrained weights.


---

## 7. Model architecture note: OBB loss

The YOLO11-OBB training loss is computed by the `v8OBBLoss` criterion
(`ultralytics/utils/loss.py`). It sums **four** weighted terms:

| Term | Component                    | Default gain |
|------|------------------------------|--------------|
| box  | Rotated IoU (`probiou`)      | 7.5          |
| cls  | BCE classification loss      | 0.5          |
| dfl  | Distribution Focal Loss      | 1.5          |
| angle| Oriented angle loss          | 1.0          |

The gains can be adjusted through `model.train(..., box=..., cls=..., dfl=...,
angle=...)`.

---

## 8. Evaluation / metrics

`model.val()` (run during training and via `training.py`) reports:
- Precision, recall, **mAP50**, **mAP50-95**
- Per-class metric table
- Confusion matrix
- PR / F1 curves

Adding additional metrics is future work (`todo.txt`); the above is what the
current pipeline produces.

---

## 9. Smoke-test reference

A quick smoke test validates the whole pipeline *before* the expensive real
run, so wiring problems surface early instead of hours into training.

**Primary option — `training.py` `SMOKE_TEST` mode:**

Set `SMOKE_TEST = True` near the top of `training.py`, then run
`python training.py`. In this mode the script runs a short 2-trial
hyperparameter-tune sanity check (a few epochs per trial) and then exits,
without doing the full tuning or final training. It verifies that the model
loads, the tuning pipeline runs, and `runs/tune/best_hyperparameters.yaml` is
produced.

```bash
python training.py        # with SMOKE_TEST = True set first
```

After it passes, set `SMOKE_TEST = False` and rerun for the full tune + final
training.

**Alternative — `provider/06smoke_test.py`:**

A legacy standalone script that trains an untrained `yolo11n-obb.yaml` on CPU
for 2 epochs to validate data loading and label format. It can be used as a
lightweight, offline check independent of tuning.

---

## 10. References

- MAR20 dataset: https://gcheng-nwpu.github.io/
- Ultralytics YOLO (OBB): https://docs.ultralytics.com/tasks/obb/
