#!/usr/bin/env python3
"""
detect.py — Object detection on images and videos using ImageAI (PyTorch backend).

This is the tool that delivers the core requirement of the project:
point it at any image or video file you provide and it will detect the
80 COCO object classes, draw bounding boxes and save both the annotated
media and a JSON report of every detection.

Subcommands
-----------
    python detect.py image <path> [options]     detect objects in one image
    python detect.py video <path> [options]     detect objects in every frame of a video
    python detect.py download <model|all>       fetch pretrained weights
    python detect.py models                     show available models + status
    python detect.py classes                    list the 80 detectable object classes

Examples
--------
    python detect.py image photo.jpg
    python detect.py image photo.jpg --min-confidence 40 --only person,dog
    python detect.py video clip.mp4 --interval 1 --max-seconds 30
    python detect.py video clip.mp4 --model retinanet --cpu
    python detect.py download tiny-yolov3

Outputs (written to ./output by default)
----------------------------------------
    <name>_detected.jpg      annotated image with boxes + labels
    <name>_detected.jpg.json detection report (boxes, confidences, counts)
    <name>_detected.mp4      annotated video
    <name>_detected.mp4.json per-frame + aggregate detection report
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from collections import Counter
from datetime import datetime, timezone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

_RELEASE = "https://github.com/OlafenwaMoses/ImageAI/releases/download/3.0.0-pretrained/"

# Supported backbones. All three detect the same 80 COCO classes; they differ
# in speed/accuracy trade-off (see `python detect.py models`).
MODEL_REGISTRY: dict[str, dict] = {
    "tiny-yolov3": {
        "file": "tiny-yolov3.pt",
        "url": _RELEASE + "tiny-yolov3.pt",
        "setter": "setModelTypeAsTinyYOLOv3",
        "speed": "fastest",
        "accuracy": "good",
        "size": "~34 MB",
    },
    "yolov3": {
        "file": "yolov3.pt",
        "url": _RELEASE + "yolov3.pt",
        "setter": "setModelTypeAsYOLOv3",
        "speed": "medium",
        "accuracy": "high",
        "size": "~248 MB",
    },
    "retinanet": {
        "file": "retinanet_resnet50_fpn_coco-eeacb38b.pth",
        "url": _RELEASE + "retinanet_resnet50_fpn_coco-eeacb38b.pth",
        "setter": "setModelTypeAsRetinaNet",
        "speed": "slower",
        "accuracy": "high",
        "size": "~130 MB",
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}  # what ImageAI's image loader accepts


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def _import_imageai():
    """Import ImageAI with actionable errors when a dependency is missing."""
    try:
        from imageai.Detection import ObjectDetection, VideoObjectDetection
    except ModuleNotFoundError as exc:
        hint = {
            "cv2": "opencv-python is not installed  ->  python -m pip install opencv-python",
            "torch": "PyTorch is not installed  ->  python -m pip install torch torchvision",
            "torchvision": "torchvision is not installed  ->  python -m pip install torchvision",
            "tkinter": "tkinter is missing  ->  reinstall Python with the 'tcl/tk' option enabled",
        }.get(exc.name, "run:  python -m pip install -r requirements.txt")
        sys.exit(f"ERROR: missing module '{exc.name}'. {hint}")
    except Exception as exc:  # broken torch install, DLL problems, ...
        sys.exit(f"ERROR: could not import imageai.Detection: {exc}\n"
                 f"Reinstall the AI stack:  python -m pip install -r requirements.txt")
    return ObjectDetection, VideoObjectDetection


def _import_cv2():
    try:
        import cv2
    except ModuleNotFoundError:
        sys.exit("ERROR: opencv-python is not installed  ->  "
                 "python -m pip install opencv-python")
    return cv2


def coco_class_names() -> list[str]:
    """Read the 80 COCO class names shipped inside the ImageAI package."""
    import imageai

    path = os.path.join(os.path.dirname(imageai.__file__), "Detection", "coco_classes.txt")
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------------
def resolve_model_path(key: str, override: str | None) -> str:
    if override:
        path = os.path.abspath(override)
        if not os.path.isfile(path):
            sys.exit(f"ERROR: model file not found: {path}")
        if path.endswith(".h5"):
            sys.exit(
                "ERROR: '.h5' files are TensorFlow/Keras models and are NOT supported by "
                "ImageAI 3.x (PyTorch backend).\n"
                "Download PyTorch weights:  python detect.py download tiny-yolov3"
            )
        return path

    path = os.path.join(MODELS_DIR, MODEL_REGISTRY[key]["file"])
    if not os.path.isfile(path):
        sys.exit(
            f"ERROR: pretrained weights not found:\n  {path}\n\n"
            f"Download them with:   python detect.py download {key}\n"
            f"or directly from:     {MODEL_REGISTRY[key]['url']}"
        )
    return path


def _set_objectness(detector, value: float) -> None:
    """Override ImageAI's internal objectness gate (hard-coded 0.5).

    This gate is applied to the raw model score BEFORE `--min-confidence`
    filtering. On heavily compressed video the raw score of a real object can
    sit just below 0.5, which makes ImageAI return nothing at all. Lowering it
    (e.g. 0.3) trades a few false positives for recall; `--min-confidence`
    still filters what gets reported.
    """
    if not 0.0 < value <= 1.0:
        sys.exit("ERROR: --objectness must be in (0, 1].")
    # VideoObjectDetection wraps a private ObjectDetection instance; both use
    # Python name-mangled private attributes.
    inner = getattr(detector, "_VideoObjectDetection__detector", detector)
    setattr(inner, "_ObjectDetection__objectness_score", float(value))


def build_detector(key: str, args) -> object:
    """Create a ready-to-use ImageAI ObjectDetection instance."""
    ObjectDetection, _ = _import_imageai()
    path = resolve_model_path(key, getattr(args, "model_path", None))

    detector = ObjectDetection()
    getattr(detector, MODEL_REGISTRY[key]["setter"])()
    detector.setModelPath(path)
    if getattr(args, "cpu", False):
        detector.useCPU()  # must run before loadModel so weights land on CPU
    detector.loadModel()
    _set_objectness(detector, getattr(args, "objectness", 0.5))
    return detector


def build_custom_objects(only: list[str] | None) -> dict | None:
    """Turn --only person,dog into ImageAI's CustomObjects dictionary."""
    if not only:
        return None
    valid = {name.replace(" ", "_") for name in coco_class_names()}
    requested = {name.strip().lower().replace(" ", "_") for name in only if name.strip()}
    unknown = sorted(requested - valid)
    if unknown:
        sys.exit(f"ERROR: unknown object class(es): {', '.join(unknown)}\n"
                 f"Run 'python detect.py classes' to see all 80 class names.")
    ObjectDetection, _ = _import_imageai()
    # Build the dict via a throw-away instance so validation stays inside ImageAI.
    probe = ObjectDetection()
    return probe.CustomObjects(**{name: True for name in requested})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _write_json(payload: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def _stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _print_objects(detections: list[dict]) -> None:
    if not detections:
        print("  (no objects detected above the confidence threshold)")
        return
    width = max(len(d["name"]) for d in detections)
    for i, d in enumerate(detections, 1):
        x1, y1, x2, y2 = d["box_points"]
        print(f"  [{i}] {d['name']:<{width}}  {d['percentage_probability']:6.2f}%   "
              f"box=({x1}, {y1}) -> ({x2}, {y2})")


def _as_jpg_if_needed(src: str) -> tuple[str, str | None]:
    """ImageAI's loaders only accept .jpg/.jpeg/.png; convert anything else.

    Returns (path_to_use, temp_file_or_None).
    """
    if os.path.splitext(src)[1].lower() in IMAGE_EXTS:
        return src, None
    try:
        from PIL import Image
    except ModuleNotFoundError:
        sys.exit("ERROR: Pillow is required to convert this image  ->  "
                 "python -m pip install Pillow")
    fd, tmp = tempfile.mkstemp(suffix=".jpg", prefix="detect_")
    os.close(fd)
    with Image.open(src) as img:
        img.convert("RGB").save(tmp, format="JPEG")
    return tmp, tmp


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_image(args) -> int:
    src = os.path.abspath(args.input)
    if not os.path.isfile(src):
        sys.exit(f"ERROR: input image not found: {src}")

    out_dir = _ensure_output_dir(os.path.abspath(args.output_dir))
    stem = os.path.splitext(os.path.basename(src))[0]
    ext = os.path.splitext(src)[1].lower()
    out_ext = ext if ext in IMAGE_EXTS else ".jpg"
    out_path = os.path.join(out_dir, f"{stem}_detected{out_ext}")

    work, temp = _as_jpg_if_needed(src)
    # Pre-copy: if the model finds nothing, ImageAI skips writing the output,
    # so we make sure an (unmodified) annotated copy always exists.
    shutil.copyfile(work, out_path)

    detector = build_detector(args.model, args)
    custom = build_custom_objects(args.only)

    print(f"Detecting objects in {os.path.basename(src)} "
          f"[model={args.model}, min-confidence={args.min_confidence}%, "
          f"objectness={args.objectness}] ...")
    try:
        detections = detector.detectObjectsFromImage(
            input_image=work,
            output_image_path=out_path,
            minimum_percentage_probability=args.min_confidence,
            custom_objects=custom,
        )
    finally:
        if temp and os.path.exists(temp):
            os.remove(temp)

    report = {
        "input": src,
        "output": out_path,
        "model": args.model,
        "min_confidence": args.min_confidence,
        "objectness_gate": args.objectness,
        "objects_detected": len(detections),
        "detections": detections,
        "processed_at": _stamp(),
    }
    report_path = out_path + ".json"
    _write_json(report, report_path)

    _print_objects(detections)
    print(f"\n  annotated image : {out_path}")
    print(f"  report          : {report_path}")
    return 0


def cmd_video(args) -> int:
    src = os.path.abspath(args.input)
    if not os.path.isfile(src):
        sys.exit(f"ERROR: input video not found: {src}")

    cv2 = _import_cv2()
    out_dir = _ensure_output_dir(os.path.abspath(args.output_dir))
    stem = os.path.splitext(os.path.basename(src))[0]
    out_base = os.path.join(out_dir, f"{stem}_detected")  # ImageAI appends '.mp4'

    # Read source properties so the output video keeps the original speed.
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"ERROR: OpenCV could not open the video: {src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    fps = args.fps if args.fps else (src_fps if 1.0 <= src_fps <= 240.0 else 20.0)

    _, VideoObjectDetection = _import_imageai()
    path = resolve_model_path(args.model, args.model_path)
    detector = VideoObjectDetection()
    getattr(detector, MODEL_REGISTRY[args.model]["setter"])()
    detector.setModelPath(path)
    if args.cpu:
        detector.useCPU()
    detector.loadModel()
    _set_objectness(detector, args.objectness)

    custom = build_custom_objects(args.only)

    per_frame: list[dict] = []

    def on_frame(frame_no, objects, counts):
        per_frame.append({"frame": frame_no, "counts": counts, "objects": objects})

    print(f"Detecting objects in video {os.path.basename(src)} "
          f"({width}x{height}, {src_fps:.6g} fps, ~{src_frames} frames) ...")
    print(f"  model={args.model}  min-confidence={args.min_confidence}%  "
          f"objectness={args.objectness}  every {args.interval} frame(s)")

    try:
        output_video = detector.detectObjectsFromVideo(
            input_file_path=src,
            output_file_path=out_base,
            frames_per_second=fps,
            frame_detection_interval=args.interval,
            minimum_percentage_probability=args.min_confidence,
            log_progress=args.log_progress,
            save_detected_video=True,
            per_frame_function=on_frame,
            detection_timeout=args.max_seconds,
            custom_objects=custom,
        )
    except ValueError as exc:
        sys.exit(f"ERROR: video detection failed: {exc}")

    if not output_video or not os.path.isfile(output_video):
        sys.exit("ERROR: no output video was produced (was the input a valid video?).")

    # Aggregate statistics across all analysed frames.
    totals_counter: Counter = Counter()
    for rec in per_frame:
        totals_counter.update(rec["counts"])
    top = totals_counter.most_common(10)

    # ImageAI's own "average instances per frame" truncates to int (10 cats over
    # 12 frames would report 0), so we compute a precise average ourselves.
    frames_n = len(per_frame) or 1
    precise_avg = {name: round(total / frames_n, 3)
                   for name, total in sorted(totals_counter.items())}

    report = {
        "input": src,
        "output": output_video,
        "model": args.model,
        "min_confidence": args.min_confidence,
        "objectness_gate": args.objectness,
        "frame_interval": args.interval,
        "output_fps": fps,
        "frames_analysed": len(per_frame),
        "top_objects": [{"name": n, "total_instances": c} for n, c in top],
        "average_instances_per_frame": precise_avg,
        "per_frame": per_frame,
        "processed_at": _stamp(),
    }
    report_path = output_video + ".json"
    _write_json(report, report_path)

    print(f"\n  frames analysed : {len(per_frame)}")
    if top:
        print("  top objects     : " +
              ", ".join(f"{n} ({c})" for n, c in top))
    else:
        print("  (no objects detected above the confidence threshold)")
    print(f"  annotated video : {output_video}")
    print(f"  report          : {report_path}")
    return 0


def cmd_download(args) -> int:
    keys = list(MODEL_REGISTRY) if args.model == "all" else [args.model]
    _ensure_output_dir(MODELS_DIR)
    for key in keys:
        meta = MODEL_REGISTRY[key]
        dest = os.path.join(MODELS_DIR, meta["file"])
        if os.path.isfile(dest):
            print(f"[skip] {key}: already present -> {dest}")
            continue
        print(f"[get ] {key}: {meta['url']}")
        tmp = dest + ".part"
        try:
            urllib.request.urlretrieve(meta["url"], tmp)
        except Exception as exc:
            if os.path.exists(tmp):
                os.remove(tmp)
            sys.exit(f"ERROR: download failed for {key}: {exc}")
        shutil.move(tmp, dest)
        print(f"[done] {key} -> {dest} ({os.path.getsize(dest):,} bytes)")
    return 0


def cmd_models(_args) -> int:
    print(f"{'name':<14} {'file':<46} {'speed':<9} {'accuracy':<8} status")
    print("-" * 92)
    for key, meta in MODEL_REGISTRY.items():
        dest = os.path.join(MODELS_DIR, meta["file"])
        status = "ready" if os.path.isfile(dest) else f"missing  (python detect.py download {key})"
        print(f"{key:<14} {meta['file']:<46} {meta['speed']:<9} {meta['accuracy']:<8} {status}")
    return 0


def cmd_classes(_args) -> int:
    names = coco_class_names()
    print(f"{len(names)} detectable COCO classes:\n")
    for i in range(0, len(names), 4):
        print("  " + "".join(f"{n:<24}" for n in names[i:i + 4]))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model", choices=list(MODEL_REGISTRY), default="tiny-yolov3",
                     help="detection backbone (default: tiny-yolov3)")
    sub.add_argument("--model-path", default=None,
                     help="explicit path to a .pt/.pth weights file (overrides --model)")
    sub.add_argument("--min-confidence", type=int, default=40,
                     help="minimum detection confidence in %% (default: 40)")
    sub.add_argument("--output-dir", default=OUTPUT_DIR,
                     help=f"where results are written (default: {OUTPUT_DIR})")
    sub.add_argument("--cpu", action="store_true", help="force CPU inference")
    sub.add_argument("--objectness", type=float, default=0.5,
                     help="raw model objectness gate, 0..1 (default: 0.5; lower it "
                          "if a compressed video yields no detections)")
    sub.add_argument("--only", default=None,
                     help="comma-separated classes to detect, e.g. 'person,dog,car'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detect.py",
        description="Object detection on images and videos (ImageAI + PyTorch).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples")[1] if "Examples" in __doc__ else None,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_img = sub.add_parser("image", help="detect objects in an image")
    p_img.add_argument("input", help="path to the image (.jpg/.jpeg/.png/.webp/...)")
    _add_common(p_img)
    p_img.set_defaults(func=cmd_image)

    p_vid = sub.add_parser("video", help="detect objects in every frame of a video")
    p_vid.add_argument("input", help="path to the video (.mp4/.avi/.mov/...)")
    _add_common(p_vid)
    p_vid.add_argument("--interval", type=int, default=1,
                       help="run detection every N frames (default: 1 = every frame)")
    p_vid.add_argument("--max-seconds", type=int, default=None,
                       help="stop after N seconds of video")
    p_vid.add_argument("--fps", type=float, default=None,
                       help="output video FPS (default: source FPS)")
    p_vid.add_argument("--log-progress", action="store_true",
                       help="print each processed frame number")
    p_vid.set_defaults(func=cmd_video)

    p_dl = sub.add_parser("download", help="download pretrained model weights")
    p_dl.add_argument("model", choices=[*MODEL_REGISTRY, "all"])
    p_dl.set_defaults(func=cmd_download)

    p_md = sub.add_parser("models", help="list model backbones and their status")
    p_md.set_defaults(func=cmd_models)

    p_cl = sub.add_parser("classes", help="list the detectable object classes")
    p_cl.set_defaults(func=cmd_classes)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
