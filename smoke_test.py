#!/usr/bin/env python3
r"""
smoke_test.py — verify that the object-detection stack actually works.

    python smoke_test.py                     # uses models/sample_coco.jpg
    python smoke_test.py path\to\image.jpg   # use your own image

Checks performed:
  1. dependencies import (cv2, torch, torchvision, imageai)
  2. pretrained weights for tiny-yolov3 are present
  3. image detection finds at least one object in the sample image
  4. video detection runs on a short clip built from the sample image

Exit code 0 = everything works.
"""

from __future__ import annotations

import os
import sys
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(BASE_DIR, "models", "tiny-yolov3.pt")
SAMPLE = os.path.join(BASE_DIR, "models", "sample_coco.jpg")


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main() -> int:
    results = []

    # 1. dependencies -------------------------------------------------------
    try:
        import cv2
        import torch
        import torchvision
        from imageai.Detection import ObjectDetection, VideoObjectDetection
        results.append(check(
            "dependencies import",
            True,
            f"torch {torch.__version__}, torchvision {torchvision.__version__}, "
            f"cv2 {cv2.__version__}, cuda={torch.cuda.is_available()}",
        ))
    except Exception:
        traceback.print_exc()
        results.append(check("dependencies import", False))
        return _finish(results)

    # 2. weights ------------------------------------------------------------
    results.append(check(
        "pretrained weights present",
        os.path.isfile(MODEL),
        MODEL if os.path.isfile(MODEL) else
        "run:  python detect.py download tiny-yolov3",
    ))
    if not os.path.isfile(MODEL):
        return _finish(results)

    # 3. image detection ----------------------------------------------------
    src = sys.argv[1] if len(sys.argv) > 1 else SAMPLE
    if not os.path.isfile(src):
        results.append(check("sample image exists", False, src))
        return _finish(results)

    try:
        det = ObjectDetection()
        det.setModelTypeAsTinyYOLOv3()
        det.setModelPath(MODEL)
        det.loadModel()
        out = os.path.join(BASE_DIR, "output", "_smoke_image.jpg")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        detections = det.detectObjectsFromImage(
            input_image=src, output_image_path=out, minimum_percentage_probability=30
        )
        results.append(check(
            "image detection", True,
            f"{len(detections)} object(s): "
            + ", ".join(f"{d['name']} {d['percentage_probability']}%" for d in detections),
        ))
        results.append(check("annotated image written", os.path.isfile(out), out))
    except Exception:
        traceback.print_exc()
        results.append(check("image detection", False))

    # 4. video detection ----------------------------------------------------
    try:
        import cv2
        import numpy as np

        img = cv2.imread(src)
        h, w = img.shape[:2]
        clip = os.path.join(BASE_DIR, "output", "_smoke_clip.mp4")
        vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
        for i in range(8):
            vw.write(np.roll(img, i * 4, axis=1))
        vw.release()

        vdet = VideoObjectDetection()
        vdet.setModelTypeAsTinyYOLOv3()
        vdet.setModelPath(MODEL)
        vdet.loadModel()
        # objectness 0.3: mp4 re-encoding can push a real object just below the
        # library's hard-coded 0.5 gate (see README troubleshooting table).
        inner = vdet._VideoObjectDetection__detector
        inner._ObjectDetection__objectness_score = 0.3

        counts: dict = {}
        result = vdet.detectObjectsFromVideo(
            input_file_path=clip,
            output_file_path=os.path.join(BASE_DIR, "output", "_smoke_video"),
            frames_per_second=10,
            minimum_percentage_probability=30,
            save_detected_video=True,
            per_frame_function=lambda n, objs, c: counts.update(c),
        )
        results.append(check(
            "video detection", bool(result) and os.path.isfile(result),
            f"{result} — counts: {counts}",
        ))
    except Exception:
        traceback.print_exc()
        results.append(check("video detection", False))

    return _finish(results)


def _finish(results: list) -> int:
    ok = all(results)
    print("\n" + ("SMOKE TEST PASSED" if ok else "SMOKE TEST FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
