# dl4rs-MAR20

Deep learning detection of military aircraft in SAR (Synthetic Aperture Radar)
imagery using **YOLO11-OBB** on the **MAR20** dataset.

The project converts MAR20's PASCAL-VOC oriented annotations into the YOLO-OBB
format, splits the data into train/val/test, and trains a YOLO11 Oriented
Bounding Box model to detect the (rotated) aircraft.

---

## 1. Dataset: MAR20 – Military Aircraft

- **Source:** https://gcheng-nwpu.github.io/ (MAR20.zip) — military aircraft in RGB images.
  See `Datenquelle` for all links (including the pan.baidu mirror with code `d2uw`).
- **Size:** 3842 JPG images, each paired with a PASCAL-VOC XML annotation.
- **Annotations:** each image has both an Oriented Bounding Box (OBB) and a
  Horizontal Bounding Box (HBB). This project uses the **oriented** boxes.
- **Classes (20):** `A1 A2 A3 A4 A5 A6 A7 A8 A9 A10 A11 A12 A13 A14 A15 A16 A17 A18 A19 A20`
- **Known data quirk:** 4 XMLs (ids 334, 338, 409, 443) have a broken `0x0`
  `<size>` element, so image dimensions are read from the JPEG via PIL instead
  of trusting the XML.

---

## 2. Why Oriented Bounding Boxes (OBB)

The original `base code` was a UNet *segmentation* pipeline. This project trains
a *detection* model using oriented boxes. Rotating aircraft in SAR imagery fit an
OBB tightly and reduce background compared with a horizontal box.

---

## 3. Requirements / setup

- Python **3.13** (see `pyproject.toml`, managed with `uv`).
- Key dependencies:
  - `ultralytics>=8.4.132`
  - `torch`, `torchvision`
  - `numpy`, `matplotlib`, `scikit-image`, `albumentations`, `tqdm`

```bash
uv sync          # or: pip install -e .
```

---

## 4. Repository layout

```
dl4rs-MAR20/
├── provider/                 # data-preparation pipeline scripts
│   ├── 01split_ids.py        # train/val/test split (70/15/15, seed 0)
│   ├── convert_one.py        # OBB-VOC -> YOLO .txt for one image
│   ├── 02convert_all.py      # batch conversion of all images
│   ├── 03organize_splits.py  # move images+labels into images/ and labels/
│   ├── 04collect_classes.py  # build classes.txt
│   ├── 05make_yaml.py        # generate data.yaml
│   ├── 06smoke_test.py       # quick CPU pipeline validation
│   └── dataset_provider.py   # old UNet loader (obsolete for YOLO)
├── data/MAR20/               # the dataset (see section 6)
├── training.py               # real YOLO11-OBB training (GPU)
├── result_script.py          # (base-code helper)
├── tests.py                  # (base-code tests)
├── colab/                    # base-code notebook converted for Google Colab
├── base code/                # original UNet pipeline (historical)
├── runs/                     # training/smoke-test outputs (gitignored)
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## 5. Pipeline: from raw data to YOLO format

All data-prep steps live in `provider/` and are meant to be run **once** in the
correct order. The pipeline is expressed as a series of short, heavily-commented
scripts, one concept each.

| Step | Script                 | What it does |
|------|------------------------|--------------|
| 1    | `01split_ids.py`       | 70/15/15 train/val/test split by image id (seed 0); overwrites `ImageSets/Main/{train,val,test}.txt`. |
| 2    | `convert_one.py`       | Converts one oriented XML annotation to YOLO-OBB `.txt` (stdlib `xml.etree.ElementTree`; reads JPEG size via PIL to handle the broken-`<size>` bug). |
| 3    | `02convert_all.py`     | Applies `convert_one.py` to every image. |
| 4    | `03organize_splits.py` | Moves images and labels into the standard `images/{train,val,test}` + `labels/{...}` layout. |
| 5    | `04collect_classes.py` | Collects the class list into `data/MAR20/classes.txt`. |
| 6    | `05make_yaml.py`       | Generates the relocatable `data.yaml` (no `path:` line, see note below). |
| 7    | `06smoke_test.py`      | Sanity test: trains an untrained `yolo11n-obb.yaml` on CPU for 2 epochs to validate the whole pipeline end-to-end. |

**Important `data.yaml` note:** the YAML deliberately omits the `path:` field.
In ultralytics, `path` is resolved relative to the **current working directory**
(not the YAML file), which breaks when the project is moved. By omitting it,
ultralytics resolves `images/train` relative to the YAML's own directory, making
the whole project relocatable between machines.

---

## 6. Final data layout

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
class cx cy width height angle
```

- `class` = class index (0–19)
- `cx, cy` = normalized center
- `width, height` = normalized box size
- `angle` = orientation angle

**Split summary (70/15/15, seed 0):**

| Split | Images |
|-------|--------|
| train | 2689   |
| val   | 576    |
| test  | 577    |
| **Total** | **3842** |

---

## 7. Training

`training.py` is the real training script, meant to run on a GPU machine
(NVIDIA RTX 3060). It uses a **pretrained** model.

```bash
python training.py
```

Key settings inside `training.py`:

| Parameter      | Value                   | Reason |
|----------------|-------------------------|--------|
| `model`        | `yolo11m-obb.pt`        | Pretrained Medium OBB; ultralytics auto-rebuilds the head to our 20 classes. |
| `data`         | `data/MAR20/data.yaml`  | Our config. |
| `epochs`       | 50                      | Full run. |
| `imgsz`        | 640                     | Image size YOLO resizes to. |
| `batch`        | 16                      | Fits the 3060's 12 GB. |
| `device`       | 0                       | First CUDA GPU. |
| `project`      | `runs`                  | Output dir. |
| `seed`         | 0                       | Reproducibility. |

Results (weights, curves, `results.csv`, prediction images) are written to
`runs/`, which is gitignored (see section 9).

---

## 8. Model architecture note: OBB loss

The YOLO11-OBB training loss is computed by the `v8OBBLoss` criterion
(`ultralytics/utils/loss.py`). It sums **four** weighted terms:

| Term | Component                    | Default gain |
|------|------------------------------|--------------|
| box  | Rotated IoU (`probiou`)      | 7.5          |
| cls  | BCE classification loss      | 0.5          |
| dfl  | Distribution Focal Loss      | 1.5          |
| angle| Oriented angle loss          | 1.0          |

The gains can be adjusted through `model.train(..., box=..., cls=..., dfl=...,
angle=...)`. Implementing a **custom loss function** (e.g. signal-analytics) is
planned future work and requires subclassing the criterion — see `todo.txt`.

---

## 9. Evaluation / metrics

`model.val()` (run during training and via `training.py`) reports:
- Precision, recall, **mAP50**, **mAP50-95**
- Per-class metric table
- Confusion matrix
- PR / F1 curves

Adding additional metrics is future work (`todo.txt`); the above is what the
current pipeline produces.

---

## 10. Smoke-test reference

A full pipeline validation was completed successfully (untrained
`yolo11n-obb.yaml`, CPU, 2 epochs) at:

```
runs/obb/runs/smoke_test-4/
├── weights/          best.pt, last.pt
├── results.csv       per-epoch metrics
├── PR_curve.png, F1_curve.png, confusion_matrix.png
└── val_batch*.jpg    prediction vs ground truth
```

The low metrics from the smoke test are expected (a from-scratch, 2-epoch CPU
run) — its purpose was to validate the pipeline, not accuracy.

---

## 11. Data provenance / license

- All dataset links are recorded in `Datenquelle` (MAR20.zip, SAR-aircraft-data
  repo, etc.).
- Research datasets such as these are typically **research-only**; confirm terms
  before redistribution.

---

## 12. Status / roadmap

**Done:**
- [x] OBB-VOC → YOLO-OBB conversion pipeline (`provider/`)
- [x] 70/15/15 split, standard `images`/`labels` layout, `data.yaml`
- [x] YOLO11-OBB smoke test passing end-to-end

**Next:**
- [ ] Real GPU training on the RTX 3060 (`python training.py`)
- [ ] Additional metrics (recall / precision / F1 emphasis) — `todo.txt`
- [ ] Custom signal-analytics loss function — `todo.txt`

---

## 13. References

- MAR20 dataset: https://gcheng-nwpu.github.io/
- SAR aircraft data repo: https://github.com/hust-rslab/SAR-aircraft-data.git
- Ultralytics YOLO (OBB): https://docs.ultralytics.com/tasks/obb/
