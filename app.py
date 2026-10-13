"""
app.py — Computer Vision Web Application
========================================

Flask + SQLAlchemy(SQLite) + Pillow/NumPy + modular ML inference.

Flow:
    GET  /           -> renders the single-page dashboard (templates/index.html)
    POST /predict    -> validates + saves the upload (uuid-prefixed name),
                        runs model inference, stores a row in SQLite,
                        returns JSON {status, prediction, confidence, image_url}
    GET  /history    -> returns all rows ordered by created_at DESC as JSON

The ML layer is written as standalone functions (load_model / preprocess_image /
predict_image) so it can be extracted into its own module (e.g. ml/model.py)
without touching the web layer.
"""

from __future__ import annotations

import io
import logging
import mimetypes
import os
import uuid
import zlib
from datetime import datetime

import numpy as np
from PIL import Image, UnidentifiedImageError
from flask import Flask, jsonify, render_template, request
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------------
# 1. Application configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

# Keep every generated filename short enough that the full path stays well
# inside the classic Windows MAX_PATH limit (260) — a 300-char upload name
# would otherwise produce an unopenable path.
MAX_NAME_LEN = max(40, 240 - len(UPLOAD_DIR))

# Only these image extensions are accepted (case-insensitive).
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# Reject payloads larger than 8 MB early (returns HTTP 413).
MAX_CONTENT_LENGTH = 8 * 1024 * 1024

# Windows registry often lacks .webp -> without this it is served as
# application/octet-stream and some browsers refuse to render it.
mimetypes.add_type("image/webp", ".webp")

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(
    BASE_DIR, "predictions.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
app.config["UPLOAD_DIR"] = UPLOAD_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# SQLAlchemy is initialised against the Flask app here (Flask-SQLAlchemy 3.x).
db = SQLAlchemy(app=app)


# ---------------------------------------------------------------------------
# 2. Database model (SQLite via SQLAlchemy ORM)
# ---------------------------------------------------------------------------
class Prediction(db.Model):
    """One row per uploaded image + its inference result."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255), nullable=False)  # original upload name
    saved_filepath = db.Column(db.String(255), nullable=False)  # where it lives on disk
    predicted_label = db.Column(db.String(100), nullable=False)  # model output
    confidence_score = db.Column(db.Float, nullable=True)  # 0.0 - 1.0
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> dict:
        """Serialise a row for the /history JSON payload."""
        # saved_filepath is stored as "static/uploads/<name>" (relative, portable).
        # The browser needs a URL path, so prefix it with "/".
        image_url = "/" + str(self.saved_filepath).replace("\\", "/").lstrip("/")
        return {
            "id": self.id,
            "filename": self.filename,
            "predicted_label": self.predicted_label,
            "confidence_score": (
                round(self.confidence_score, 4)
                if self.confidence_score is not None
                else None
            ),
            "image_url": image_url,
            "created_at": (self.created_at.isoformat() + "Z") if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# 3. ML layer (modular: preprocessing, model loading, prediction)
# ---------------------------------------------------------------------------
IMG_SIZE = 224  # MobileNetV2 / ImageNet input resolution
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Labels used by the deterministic fallback classifier (no torch required).
DUMMY_LABELS = [
    "golden retriever",
    "house cat",
    "sports car",
    "sailing boat",
    "airplane",
    "mountain landscape",
    "city skyline",
    "pizza",
    "coffee mug",
    "backpack",
    "desk chair",
    "butterfly",
]

# Lazy singleton state for the pretrained model.
_MODEL: object | None = None
_CATEGORIES: list[str] | None = None
_MODEL_LOADED = False


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Pillow -> float32 CHW tensor-input, ImageNet normalised.

    Kept torch-free on purpose: it produces a plain NumPy array so the same
    preprocessing works for both the pretrained model and the fallback.
    """
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0          # HWC, [0..1]
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD                 # normalise
    arr = np.ascontiguousarray(arr.transpose(2, 0, 1))         # HWC -> CHW
    return arr


def load_model() -> tuple[object | None, list[str] | None]:
    """Try to load a pretrained MobileNetV2 from torchvision.

    Returns (model, categories) on success, or (None, None) when torch /
    torchvision (or the pretrained weights) are unavailable — in that case the
    caller falls back to the bundled dummy classifier.
    """
    try:
        from torchvision import models  # heavy import, only when available
    except Exception as exc:  # ImportError, missing DLLs, ...
        app.logger.warning("torchvision unavailable (%s) — using fallback classifier.", exc)
        return None, None

    try:
        weights = models.MobileNet_V2_Weights.DEFAULT  # ImageNet-1K weights
        model = models.mobilenet_v2(weights=weights)
        model.eval()  # inference mode (disables dropout, uses running stats)
        categories = list(weights.meta["categories"])  # 1000 ImageNet class names
        app.logger.info("Pretrained MobileNetV2 loaded (%d classes).", len(categories))
        return model, categories
    except Exception as exc:
        app.logger.warning("Could not load pretrained weights (%s) — using fallback classifier.", exc)
        return None, None


def _get_model() -> tuple[object | None, list[str] | None]:
    """Lazy-load the model exactly once (keeps app startup fast)."""
    global _MODEL, _CATEGORIES, _MODEL_LOADED
    if not _MODEL_LOADED:
        _MODEL, _CATEGORIES = load_model()
        _MODEL_LOADED = True
    return _MODEL, _CATEGORIES


def get_backend_name() -> str:
    """Human-readable name of the active inference backend (shown in the UI)."""
    model, _ = _get_model()
    return "MobileNetV2 · ImageNet" if model is not None else "Fallback Classifier · demo"


def _label_prototype(label: str, dim: int) -> np.ndarray:
    """Deterministic pseudo-random unit vector per label (seeded by crc32).

    Same label => same vector on every run/process (np.random.default_rng with
    a fixed integer seed is reproducible, unlike Python's hash()).
    """
    rng = np.random.default_rng(zlib.crc32(label.encode("utf-8")))
    proto = rng.standard_normal(dim).astype(np.float32)
    return proto / (float(np.linalg.norm(proto)) + 1e-8)


def dummy_predict(image: Image.Image) -> tuple[str, float]:
    """Deterministic, torch-free fallback "model".

    Reduces the image to a 16x16 colour descriptor, compares it against one
    fixed prototype vector per candidate label and softmaxes the similarities.
    The output is stable for the same image but genuinely image-dependent.
    """
    thumb = image.convert("RGB").resize((16, 16), Image.BILINEAR)
    feats = np.asarray(thumb, dtype=np.float32).ravel()
    feats = (feats - feats.mean()) / (feats.std() + 1e-6)  # z-scored descriptor

    logits = np.array(
        [float(feats @ _label_prototype(lbl, feats.size)) for lbl in DUMMY_LABELS],
        dtype=np.float32,
    )
    logits = (logits - logits.max()) * 4.0  # sharpen -> decisive top-1 score
    probs = np.exp(logits) / np.exp(logits).sum()

    best = int(np.argmax(probs))
    return DUMMY_LABELS[best], float(probs[best])


def predict_image(image: Image.Image) -> tuple[str, float]:
    """Run inference on a Pillow image. Returns (label, confidence 0..1)."""
    model, categories = _get_model()

    # --- Fallback path (no torch installed) ---------------------------------
    if model is None:
        return dummy_predict(image)

    # --- Pretrained MobileNetV2 path ----------------------------------------
    import torch  # safe: model only exists when torch imported successfully

    arr = preprocess_image(image)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # CHW -> NCHW batch of 1

    with torch.inference_mode():  # no autograd bookkeeping => faster + less RAM
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    confidence, index = torch.max(probs, dim=0)
    label = categories[int(index.item())]
    return label, float(confidence.item())


# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
def ext_of(filename: str) -> str:
    """Lowercased extension without the dot ('' when there is none)."""
    return filename.rsplit(".", 1)[1].lower() if "." in filename else ""


def allowed_file(filename: str) -> bool:
    """True when the filename carries one of the whitelisted extensions."""
    return ext_of(filename) in ALLOWED_EXTENSIONS


def split_name(raw_name: str) -> tuple[str, str]:
    """Split an upload name into a safe (stem, '.ext') pair.

    The stem goes through secure_filename() (non-Latin stems fall back to
    'image') and is truncated so the DB String(255) columns and filesystem
    path limits are never overflowed. The extension is preserved as-is.
    """
    stem, dot, ext = raw_name.rpartition(".")
    safe_stem = secure_filename(stem) if dot else ""
    if not safe_stem:
        safe_stem = "image"
    if len(safe_stem) > 200:  # 32 (uuid) + 1 ('_') + 200 + ext < 255
        safe_stem = safe_stem[:200]
    return safe_stem, f".{ext}" if dot else ""


def make_unique_name(stem: str, suffix: str) -> str:
    """uuid-prefixed name -> guarantees no overwrite / path traversal.

    Example: ('beach_photo', '.JPG') -> '3f1c9d...e41_beach_photo.JPG'

    The stem is further truncated so uuid + stem + ext fits MAX_NAME_LEN,
    keeping the resolved path below Windows MAX_PATH.
    """
    budget = MAX_NAME_LEN - 32 - 1 - len(suffix)  # uuid + '_' + extension
    return f"{uuid.uuid4().hex}_{stem[:max(budget, 1)]}{suffix}"


def delete_saved_file(path: str) -> None:
    """Best-effort cleanup of a file we saved but could not use."""
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError as exc:  # pragma: no cover
        app.logger.warning("Could not remove %s: %s", path, exc)


# ---------------------------------------------------------------------------
# 5. Routes
# ---------------------------------------------------------------------------
@app.get("/")
def index():
    """Render the single-page dashboard."""
    return render_template("index.html", model_backend=get_backend_name())


@app.post("/predict")
def predict():
    """Validate -> save (uuid name) -> infer -> persist -> return JSON."""
    # ---- 5.1 Basic multipart validation ------------------------------------
    if "image" not in request.files:
        return jsonify(status="error", message="No 'image' field in the request."), 400

    file = request.files["image"]
    if file is None or not file.filename or not file.filename.strip():
        return jsonify(status="error", message="No file selected — please choose an image."), 400

    # Keep the raw name for validation; a secure/display-safe version is
    # derived below (secure_filename() blanks out non-Latin characters).
    raw_name = file.filename.strip()

    # ---- 5.2 Extension whitelist -------------------------------------------
    if not allowed_file(raw_name):
        return (
            jsonify(
                status="error",
                message="Unsupported file type. Allowed: .jpg, .jpeg, .png, .webp",
            ),
            400,
        )

    # Safe, bounded name for the DB column + uuid-prefixed name for the disk.
    stem, suffix = split_name(raw_name)
    original_name = f"{stem}{suffix}"  # -> Prediction.filename

    # ---- 5.3 Read + verify the image bytes ---------------------------------
    file_bytes = file.read()
    if not file_bytes:
        return jsonify(status="error", message="The uploaded file is empty."), 400

    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.load()  # raises for truncated / corrupt payloads
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        # Also catches crafted "decompression bomb" images (Pillow safety limit).
        return jsonify(status="error", message="The file is not a valid image."), 400

    # ---- 5.4 Save to disk with a unique, secure name ------------------------
    unique_name = make_unique_name(stem, suffix)
    saved_path = os.path.join(app.config["UPLOAD_DIR"], unique_name)   # OS-native path
    relative_path = f"static/uploads/{unique_name}"                    # DB + URL path

    try:
        with open(saved_path, "wb") as fh:
            fh.write(file_bytes)
    except OSError as exc:
        app.logger.exception("Could not persist upload")
        return jsonify(status="error", message=f"Could not save file: {exc}"), 500

    # ---- 5.5 Inference -------------------------------------------------------
    try:
        label, confidence = predict_image(image)
    except Exception:
        app.logger.exception("Inference failed")
        delete_saved_file(saved_path)  # don't keep orphan files
        return jsonify(status="error", message="Model inference failed."), 500

    # ---- 5.6 Persist metadata in SQLite -------------------------------------
    record = Prediction(
        filename=original_name,
        saved_filepath=relative_path,
        predicted_label=label,
        confidence_score=confidence,
    )
    try:
        db.session.add(record)
        db.session.commit()
    except Exception:
        db.session.rollback()
        app.logger.exception("Database commit failed")
        delete_saved_file(saved_path)
        return jsonify(status="error", message="Could not save prediction to database."), 500

    app.logger.info("Predicted %r (%.2f) for %s", label, confidence, unique_name)

    # ---- 5.7 Response --------------------------------------------------------
    return jsonify(
        status="success",
        prediction=label,
        confidence=round(float(confidence), 4),
        image_url=f"/static/uploads/{unique_name}",
    )


@app.get("/history")
def history():
    """Every prediction, newest first (JSON consumed by main.js gallery)."""
    stmt = db.select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc())
    rows = db.session.execute(stmt).scalars().all()
    items = [row.to_dict() for row in rows]
    return jsonify(status="success", count=len(items), predictions=items)


# ---------------------------------------------------------------------------
# 6. Error handlers + bootstrap
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def payload_too_large(_error):
    """MAX_CONTENT_LENGTH exceeded — return JSON instead of an HTML error page."""
    return jsonify(status="error", message="Image too large (max 8 MB)."), 413


def init_app() -> None:
    """Create the upload folder and the database tables (idempotent)."""
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    with app.app_context():
        db.create_all()
    app.logger.info("Upload dir ready: %s", UPLOAD_DIR)


init_app()  # runs on import, so gunicorn/waitress work too


if __name__ == "__main__":
    # Development server. For production use: gunicorn -w 2 "app:app"
    app.run(host="127.0.0.1", port=5000, debug=True)
