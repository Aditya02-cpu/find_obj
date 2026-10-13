# find_obj — Object Detection for Images & Videos

Detect and localise the **80 everyday object classes** (COCO) in **any image or video you
provide**, using the **ImageAI** library on its **PyTorch** backend.

```bash
python detect.py image photo.jpg
python detect.py video clip.mp4
```

Each run produces an annotated image/video with bounding boxes **plus a JSON report** of
every detection (class, confidence, pixel coordinates, per-frame counts for video).

---

## What is in this repository

| File / folder      | What it is                                                             |
| ------------------ | ---------------------------------------------------------------------- |
| **`detect.py`**    | **The object detector** — image & video inference via ImageAI + PyTorch |
| `app.py`           | Flask web app: upload an image → single-label *classification* + history |
| `app1.py`          | Minimal drag & drop upload demo (reuses `app.py`'s ML layer)            |
| `models/`          | Pretrained weights (`*.pt` / `*.pth`) live here — download required      |
| `output/`          | Default destination for annotated images/videos + JSON reports           |
| `ARCHITECTURE.md`  | How the whole system is put together (diagrams, data flow)               |
| `docs/pytorch-backend.md` | Deep dive: **why PyTorch is mandatory** for ImageAI 3.x           |

> **Classification vs detection.** `app.py` answers *"what is in this picture?"* with one
> label (MobileNetV2). `detect.py` answers *"which objects are where?"* with bounding
> boxes for every object found — in images **and** videos. This project's requirement
> (objects in images/videos via ImageAI) is fulfilled by **`detect.py`**.

---

## Why PyTorch is required

ImageAI **3.0.2 (December 2022) replaced TensorFlow/Keras with PyTorch as its backend**.
PyTorch is therefore not optional — it *is* the engine under ImageAI:

1. **Hard import.** `imageai/Detection/__init__.py` executes `import torch` and
   `import torchvision` at module load. Without them, `from imageai.Detection import
   ObjectDetection` fails immediately with `ModuleNotFoundError`.
2. **Model architecture.** The YOLOv3 / TinyYOLOv3 networks are `torch.nn.Module`
   classes defined inside ImageAI; RetinaNet comes from `torchvision.models.detection`.
3. **Weights.** Pretrained parameters are loaded with `torch.load(...)` into the
   architecture as a PyTorch `state_dict` — this is why only `.pt` / `.pth` files are
   accepted and **`.h5` (TensorFlow) files are explicitly rejected**:
   > *"You are trying to use a Tensorflow model with ImageAI. ImageAI now uses PyTorch
   > as backend as from version 3.0.2..."*
4. **Inference.** Every image/frame is converted to a `torch.Tensor`, moved to
   CPU or CUDA, run through the network under `torch.no_grad()`, then post-processed
   (sigmoid, non-maximum suppression, coordinate scaling) with tensor operations.
5. **Device selection.** ImageAI automatically uses CUDA when available
   (`torch.cuda.is_available()`), otherwise CPU — you can force CPU with `--cpu`.

Full details, with file/line references into the installed library, are in
**[docs/pytorch-backend.md](docs/pytorch-backend.md)**.

---

## Setup

The project ships with a ready-to-use virtual environment at `.venv/`
(Python 3.14, `torch 2.14.1+cpu`, `torchvision`, `imageai 3.0.3`, `opencv-python`).

```powershell
# 1. (only if .venv is missing) create it
python -m venv .venv

# 2. activate it
.venv\Scripts\activate

# 3. install dependencies
python -m pip install -r requirements.txt

# 4. download pretrained weights (~34 MB, one time)
python detect.py download tiny-yolov3
```

**GPU instead of CPU?** Install a CUDA build of PyTorch into the environment, then drop
`--cpu` (ImageAI picks CUDA up automatically):

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Pretrained models

| `--model`   | Weight file                                | Size     | Speed    | Accuracy |
| ----------- | ------------------------------------------ | -------- | -------- | -------- |
| `tiny-yolov3` (default) | `tiny-yolov3.pt`              | ~34 MB   | fastest  | good     |
| `yolov3`    | `yolov3.pt`                                 | ~248 MB  | medium   | high     |
| `retinanet` | `retinanet_resnet50_fpn_coco-eeacb38b.pth`  | ~130 MB  | slower   | high     |

```powershell
python detect.py models               # see which weights are present
python detect.py download all         # fetch every model
```

All three detect the **same 80 classes** (`python detect.py classes`).

---

## Usage

### Images

```powershell
python detect.py image path\to\photo.jpg
python detect.py image path\to\photo.jpg --min-confidence 30
python detect.py image path\to\photo.jpg --only person,dog,car
python detect.py image path\to\photo.jpg --model retinanet --output-dir D:\results
```

```
Detecting objects in photo.jpg [model=tiny-yolov3, min-confidence=30%, objectness=0.5] ...
  [1] cat   91.20%   box=(18, 81) -> (307, 384)

  annotated image : D:\find_obj\output\photo_detected.jpg
  report          : D:\find_obj\output\photo_detected.jpg.json
```

Accepted inputs: `.jpg`, `.jpeg`, `.png` (other formats such as `.webp` are converted
automatically before detection).

### Videos

```powershell
python detect.py video path\to\clip.mp4
python detect.py video path\to\clip.mp4 --min-confidence 40 --objectness 0.3
python detect.py video path\to\clip.mp4 --interval 3          # detect every 3rd frame (faster)
python detect.py video path\to\clip.mp4 --max-seconds 30      # only first 30 seconds
python detect.py video path\to\clip.mp4 --only person,car --log-progress
```

```
Detecting objects in clip.mp4 (640x480, 10 fps, ~12 frames) ...
  model=tiny-yolov3  min-confidence=30%  objectness=0.3  every 1 frame(s)

  frames analysed : 12
  top objects     : cat (10)
  annotated video : D:\find_obj\output\clip_detected.mp4
  report          : D:\find_obj\output\clip_detected.mp4.json
```

Accepted inputs: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` (anything OpenCV can open).

### Options

| Option                | Applies to | Default | Meaning |
| --------------------- | ---------- | ------- | ------- |
| `--model`             | both       | `tiny-yolov3` | backbone: `tiny-yolov3`, `yolov3`, `retinanet` |
| `--model-path`        | both       | —       | explicit path to a `.pt`/`.pth` weights file |
| `--min-confidence`    | both       | `40`    | reporting threshold in % (what gets listed/annotated) |
| `--objectness`        | both       | `0.5`   | **raw model gate** applied before reporting (see below) |
| `--only`              | both       | —       | restrict to given classes, e.g. `person,dog` |
| `--cpu`               | both       | off     | force CPU inference |
| `--output-dir`        | both       | `output/` | where results are written |
| `--interval`          | video      | `1`     | run detection every N frames |
| `--max-seconds`       | video      | —       | stop after N seconds of video |
| `--fps`               | video      | source  | FPS of the output video |
| `--log-progress`      | video      | off     | print each processed frame number |

> **`--min-confidence` vs `--objectness`.** ImageAI applies a raw *objectness* gate
> (hard-coded `0.5`) inside its post-processing — if an object scores below it, it is
> discarded before your `--min-confidence` is ever consulted. On heavily compressed
> videos real objects can land at ~0.45 and produce **zero** detections. If that
> happens, lower `--objectness` (e.g. `0.3`) and keep `--min-confidence` where you want
> the reporting cut-off.

### Output files

* **Image** → `<name>_detected.jpg` + `<name>_detected.jpg.json`

  ```json
  {
    "model": "tiny-yolov3",
    "min_confidence": 30,
    "objects_detected": 1,
    "detections": [
      { "name": "cat", "percentage_probability": 91.2,
        "box_points": [18, 81, 307, 384] }
    ]
  }
  ```

* **Video** → `<name>_detected.mp4` + `<name>_detected.mp4.json` with
  `frames_analysed`, `top_objects`, `average_instances_per_frame` and a `per_frame`
  array (`frame`, `counts`, full `objects` with boxes/confidences).

---

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `RuntimeError: You are trying to use a Tensorflow model with ImageAI...` | a `.h5` file was passed | ImageAI 3.x needs `.pt`/`.pth` → `python detect.py download tiny-yolov3` |
| `ModuleNotFoundError: No module named 'cv2'` | OpenCV not installed | `python -m pip install opencv-python` |
| `ModuleNotFoundError: No module named 'torch'` / DLL load failed | PyTorch missing or broken | `python -m pip install -r requirements.txt` |
| `ModuleNotFoundError: No module named 'tkinter'` | ImageAI imports tkinter | reinstall Python with the *tcl/tk* component |
| `ImportError: DLL load failed ... An Application Control policy has blocked this file` | Windows **Smart App Control** blocks the *unsigned* Cython `.pyd` files shipped with SQLAlchemy 2.1 | Already applied in `.venv`: the 8 `sqlalchemy\**\*_cy*.pyd` files were renamed to `*.pyd.blocked`, so Python falls back to the identical pure-Python `.py` sources. If it returns after reinstalling SQLAlchemy, re-run the rename (see below) |
| Image works, video reports **0 objects** | objectness gate (0.5) vs compression | `--objectness 0.3`, then tune `--min-confidence` |
| Nothing detected at all | object not in the 80 COCO classes, or threshold too high | `python detect.py classes`; lower `--min-confidence` |
| `OpenCV: FFMPEG: tag 'MP4V' is not supported ...` | harmless warning — OpenCV falls back to `mp4v` | ignore |
| Output video has **no audio** | OpenCV re-encodes frames only | re-mux audio if needed: `ffmpeg -i detected.mp4 -i in.mp4 -c copy -map 0:v -map 1:a out.mp4` |
| Old `*.h5` files in the project root | legacy TensorFlow **classification** weights (ImageAI ≤ 2.1.6 era) | not usable with ImageAI 3.x — safe to archive/delete |

---

### Re-applying the Smart App Control fix

If `pip install --force-reinstall sqlalchemy` (or a future upgrade) brings the
`Application Control policy has blocked this file` error back, disable the blocked
binaries again — every one of them has a pure-Python twin, so Python just uses those:

```powershell
Get-ChildItem .venv\Lib\site-packages\sqlalchemy -Recurse -Filter *.pyd |
    Rename-Item -NewName { $_.Name + '.blocked' }
python -c "import sqlalchemy; print(sqlalchemy.__version__)"
```

The permanent alternative is allowlisting the files/folder in Windows Security
→ *App & browser control* (Smart App Control can no longer be re-enabled once turned
off, so the rename is the recommended, fully reversible route).

---

## Project structure

```
find_obj/
├── detect.py                 # image + video object detection (ImageAI + PyTorch)
├── app.py                    # Flask web app: image classification + SQLite history
├── app1.py                   # minimal drag & drop upload demo
├── requirements.txt
├── README.md
├── ARCHITECTURE.md           # system architecture & data flow diagrams
├── docs/
│   └── pytorch-backend.md    # why PyTorch is required by ImageAI
├── models/                   # pretrained weights (download via detect.py)
│   ├── tiny-yolov3.pt        # detection model (~34 MB, ready to use)
│   ├── sample_coco.jpg       # sample image  (detect.py image models\sample_coco.jpg)
│   └── cats.mp4              # sample video  (detect.py video models\cats.mp4)
├── output/                   # annotated images/videos + JSON reports
├── templates/, static/       # web app assets
└── predictions.db            # web app history (SQLite)
```

## Further reading

* **[ARCHITECTURE.md](ARCHITECTURE.md)** — components, pipelines and data flow
* **[docs/pytorch-backend.md](docs/pytorch-backend.md)** — the PyTorch requirement explained
* [ImageAI documentation](https://imageai.readthedocs.io/) ·
  [ImageAI on GitHub](https://github.com/OlafenwaMoses/ImageAI) ·
  [COCO class list](https://cocodataset.org/#home)
