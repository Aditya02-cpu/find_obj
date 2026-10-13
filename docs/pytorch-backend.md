# Why PyTorch is required (ImageAI 3.x backend)

This document explains, with references into the actually installed library
(`.venv/Lib/site-packages/imageai/`), what PyTorch does inside this project and why it
cannot be removed.

---

## 1. Short answer

**ImageAI 3.0.2 (December 2022) switched its backend from TensorFlow/Keras to
PyTorch.** Every supported detection model (YOLOv3, TinyYOLOv3, RetinaNet) is now a
PyTorch `nn.Module`, every weight file is a PyTorch `state_dict` (`.pt`/`.pth`), and
every forward pass, tensor transfer and post-processing step runs on PyTorch. If `torch`
(or `torchvision`) is not importable, `imageai.Detection` does not import at all.

```
pip show imageai          →  Version: 3.0.3   (PyTorch backend)
torch                     →  torch 2.14.1+cpu
torchvision               →  torchvision 0.29.1+cpu
```

---

## 2. Evidence inside the library

### 2.1 PyTorch is imported at module load — it is a hard dependency

`imageai/Detection/__init__.py`, lines 6–16:

```python
import torchvision          # line 6
import torch                # line 10
import cv2                  # line 11
from ..yolov3.yolov3 import YoloV3                # torch.nn.Module
from ..yolov3.tiny_yolov3 import YoloV3Tiny       # torch.nn.Module
```

So `from imageai.Detection import ObjectDetection` fails with `ModuleNotFoundError`
before a single line of user code runs if PyTorch is missing. (The same applies to
`cv2` — which is why `opencv-python` is in `requirements.txt`.)

### 2.2 TensorFlow models are explicitly rejected

`imageai/backend_check/model_extension.py` (called from `setModelPath()`):

```python
def extension_check(file_path: str) -> None:
    if file_path.endswith(".h5"):
        raise RuntimeError("You are trying to use a Tensorflow model with ImageAI.
        ImageAI now uses PyTorch as backend as from version 3.0.2 . If you want to use
        the Tensorflow models or a customly trained '.h5' model, install ImageAI 2.1.6
        or earlier. To use the latest Pytorch models, see the documentation in
        https://imageai.readthedocs.io/ ")
    elif not file_path.endswith(".pt") and not file_path.endswith(".pth"):
        raise ValueError(... "Please parse in a '.pt' and '.pth' model file.")
```

> The two `*_weights_tf_dim_ordering_tf_kernels.h5` files in the project root are
> legacy **TensorFlow/Keras classification** weights (ImageAI ≤ 2.1.6 era). They cannot
> be used for detection with ImageAI 3.x and are not needed by `detect.py`.

### 2.3 The model is built, loaded and run entirely in PyTorch

`imageai/Detection/__init__.py → ObjectDetection.loadModel()`:

```python
if self.__model_type == "yolov3":
    self.__model = YoloV3(anchors=..., num_classes=80, device=...)      # nn.Module
elif self.__model_type == "tiny-yolov3":
    self.__model = YoloV3Tiny(anchors=..., num_classes=80, device=...)   # nn.Module
elif self.__model_type == "retinanet":
    self.__model = torchvision.models.detection.retinanet_resnet50_fpn(
        pretrained=False, num_classes=91, pretrained_backbone=False)     # torchvision

state_dict = torch.load(self.__model_path, map_location=self.__device)  # .pt/.pth
self.__model.load_state_dict(state_dict)
self.__model.to(self.__device).eval()                                   # cpu / cuda
```

---

## 3. What PyTorch actually does, stage by stage

| Stage | What happens | PyTorch API involved |
| --- | --- | --- |
| **1. Device pick** | GPU if present, else CPU (override with `detect.py --cpu`) | `torch.cuda.is_available()`, `.to(device)` |
| **2. Architecture** | the network itself is a `nn.Module` (layers, weights, `forward()`) | `torch.nn`, `nn.Module` |
| **3. Weight load** | pretrained parameters read from `.pt`/`.pth` into the architecture | `torch.load()`, `load_state_dict()` |
| **4. Preprocessing** | decoded image → normalized float tensor with a batch dimension | `torch.from_numpy()`, `torch.FloatTensor`, `.div(255.0)`, `torch.cat()` |
| **5. Forward pass** | the actual inference, run with gradients disabled | `model.eval()`, `torch.no_grad()` |
| **6. Decoding** | raw conv outputs → box centres/sizes + objectness + class scores (sigmoid, exponentials against anchors) | `torch.sigmoid()`, `torch.exp()`, tensor broadcasting (`imageai/yolov3/utils.py → transform_prediction()`) |
| **7. Filtering** | objectness threshold, per-class **non-maximum suppression**, IoU maths, confidence sorting | `torch.max`, `torch.sort`, `torch.unique`, `torch.nonzero`, `torch.clamp` (`utils.py → get_predictions()`) |
| **8. Scale back** | boxes resized from the 416×416 network grid to original pixel coordinates, clipped to the image | `torch.index_select`, `torch.min`, arithmetic ops |
| **9. Result** | confidences/coordinates extracted to plain Python for JSON output | `.item()`, `float()` |
| **10. Video loop** | repeat 4–9 per frame; frames held as tensors, results collected per second/minute | same tensor ops, `VideoObjectDetection.detectObjectsFromVideo()` |

For RetinaNet the pipeline is even more concentrated: `torchvision` supplies the whole
detection model (backbone + FPN + head + built-in NMS) and ImageAI just forwards frames
through `model(images)` under `torch.no_grad()`.

---

## 4. Who does what (PyTorch vs the rest)

| Library | Role |
| --- | --- |
| **PyTorch (`torch`)** | tensor engine, `nn.Module` networks, weight loading, CPU/CUDA execution, all inference and post-processing maths |
| **torchvision** | RetinaNet architecture + image codec helpers; also the MobileNetV2/ResNet classification models used by `app.py` |
| **ImageAI** | orchestration: model-type selection, COCO class lists, thresholds, drawing boxes, the video frame loop and callback/analysis API |
| **OpenCV (`cv2`)** | media I/O only — read/write images and video frames, colour-space conversion, drawing rectangles/text |
| **Pillow / NumPy** | upload handling, array conversions, JSON-friendly outputs |

In short: **OpenCV moves pixels in and out; PyTorch thinks about them.**

---

## 5. Practical consequences

1. **Install order matters.** `torch` + `torchvision` must be a *matched pair*
   (this project: `torch 2.14.1` ↔ `torchvision 0.29.1`). Mismatched wheels cause
   `ImportError: ... incompatible function`.
2. **Weights must be PyTorch files.** Use `python detect.py download <model>`; the
   official links are embedded in `detect.py` (`MODEL_REGISTRY`). `.h5` is rejected by
   design.
3. **CPU vs GPU.** The installed environment is CPU-only (`torch+cpu`). ImageAI decides
   via `torch.cuda.is_available()`; `detect.py --cpu` forces CPU. GPU needs a CUDA build
   of PyTorch (see README §Setup).
4. **Memory.** The model is loaded once per process and kept in memory
   (`eval()` mode, inference under `no_grad()`), so repeated detections cost no extra
   load time and no autograd graph is built.
5. **Two independent PyTorch workloads exist in this repo:**
   * `detect.py` → ImageAI → YOLOv3/RetinaNet — **object detection** (boxes, many objects);
   * `app.py` → torchvision MobileNetV2 — optional **image classification** (one label).
   Both share the same installed PyTorch, so installing it serves the whole project.

---

## 6. Verifying your environment

```powershell
.venv\Scripts\activate
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available())"
python -c "from imageai.Detection import ObjectDetection; print('imageai OK')"
python detect.py models        # weights present?
python detect.py image models\sample_coco.jpg --min-confidence 30
```

Expected: version string, `imageAI OK`, `tiny-yolov3 … ready`, and a detection of
`cat ≈ 91%` for the sample image.

---

## 7. If you must keep using TensorFlow `.h5` models

That path is frozen in history: ImageAI 2.1.6 + Python 3.7 + `tensorflow==2.4.0` +
`keras==2.4.3` (see ImageAI's
[BACKEND_MIGRATION.md](https://github.com/OlafenwaMoses/ImageAI/blob/master/BACKEND_MIGRATION.md)).
It is not compatible with this project's environment (Python 3.14, torch 2.14) and not
needed for detection — the PyTorch models referenced in this README detect the 80 COCO
classes out of the box.
