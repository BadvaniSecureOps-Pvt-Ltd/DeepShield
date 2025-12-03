# inference.py — DeepShield inference + explainability (stable version)
import os
import io
import base64
import traceback
from typing import Tuple, Optional, Any
from PIL import Image

from app.models.model import predict_with_explain

# optional HEIC support
try:
    import pillow_heif
except ImportError:
    pillow_heif = None

# optional PDF → images
try:
    from extract_images import extract_images_from_pdf
except ImportError:
    extract_images_from_pdf = None


# -------------------------------------------------------
# Utility: verify bytes == valid image
# -------------------------------------------------------
def _is_valid_image_bytes(b: bytes) -> bool:
    try:
        Image.open(io.BytesIO(b)).convert("RGB")
        return True
    except Exception:
        return False


# -------------------------------------------------------
# NORMALIZER — the heart of fixing dict issues
# Always returns:
#   label, confidence, model_version, explanation_or_None
# -------------------------------------------------------
def normalize_output(raw: Any) -> Tuple[str, float, str, Optional[Any]]:
    try:
        # ------------------ CASE 1: Dictionary ------------------
        if isinstance(raw, dict):

            # unwrap { "result": {...} }
            if len(raw) == 1 and next(iter(raw)) in ("result", "data", "output"):
                raw = list(raw.values())[0] or raw

            label = (
                raw.get("label")
                or raw.get("prediction")
                or raw.get("class")
                or raw.get("pred")
            )

            confidence = (
                raw.get("confidence")
                or raw.get("prob")
                or raw.get("score")
            )

            model_version = (
                raw.get("model_version")
                or raw.get("version")
                or raw.get("model")
            )

            explanation = (
                raw.get("explainability")
                or raw.get("explanation")
                or raw.get("explanation_path")
                or raw.get("explain")
                or raw.get("heatmap")
                or raw.get("cam")
            )

            try:
                confidence = float(confidence) if confidence is not None else 0.0
            except:
                confidence = 0.0

            return (
                str(label) if label else "unknown",
                confidence,
                str(model_version) if model_version else "unknown",
                explanation,
            )

        # ------------------ CASE 2: tuple/list ------------------
        if isinstance(raw, (list, tuple)):
            padded = list(raw) + [None] * (4 - len(raw))
            label, conf, model_version, explanation = padded

            try:
                conf = float(conf) if conf is not None else 0.0
            except:
                conf = 0.0

            return (
                str(label) if label else "unknown",
                conf,
                str(model_version) if model_version else "unknown",
                explanation,
            )

        # ------------------ CASE 3: raw bytes (explainability image) ------------------
        if isinstance(raw, (bytes, bytearray)):
            if _is_valid_image_bytes(bytes(raw)):
                return "unknown", 0.0, "unknown", bytes(raw)
            else:
                return "unknown", 0.0, "unknown", None

        # ------------------ CASE 4: string ------------------
        if isinstance(raw, str):
            s = raw.strip()

            # base64 data:image/.....
            if s.startswith("data:image"):
                return "unknown", 0.0, "unknown", s

            # maybe JSON?
            try:
                import json
                return normalize_output(json.loads(s))
            except:
                pass

            # else treat as path string
            return "unknown", 0.0, "unknown", s

    except Exception:
        traceback.print_exc()

    print("[DEBUG] normalize_output failed, RAW:", repr(raw)[:300])
    return "unknown", 0.0, "unknown", None


# -------------------------------------------------------
# PREPROCESS — load file → bytes (HEIC + PDF supported)
# -------------------------------------------------------
def preprocess_file(file_path: str) -> bytes:
    filename = file_path.lower()

    with open(file_path, "rb") as f:
        content = f.read()

    # PDF
    if filename.endswith(".pdf"):
        if not extract_images_from_pdf:
            raise ValueError("PDF extraction not available")
        imgs = extract_images_from_pdf(io.BytesIO(content))
        if not imgs:
            raise ValueError("PDF contains no images")
        content = imgs[0]

    # HEIC
    elif filename.endswith((".heic", ".heif", ".hif")):
        if pillow_heif is None:
            raise ValueError("Install pillow_heif to read HEIC")
        heif = pillow_heif.read_heif(content)
        img = Image.frombytes(heif.mode, heif.size, heif.data)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        content = buf.getvalue()

    # Validate image
    try:
        Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid or corrupted image file: {e}")

    return content


# -------------------------------------------------------
# INFERENCE (no explainability return)
# -------------------------------------------------------
def run_inference(file_path: str) -> Tuple[str, float, str, str]:
    basename = os.path.basename(file_path)
    try:
        file_bytes = preprocess_file(file_path)
        raw_out = predict_with_explain(file_bytes, filename=basename)
        label, conf, model_version, _ = normalize_output(raw_out)
        return label, conf, model_version, ""
    except Exception as e:
        print("[ERROR] run_inference:", e)
        return "error", 0.0, "unknown", ""


# -------------------------------------------------------
# INFERENCE + EXPLAINABILITY
# Explanation saved into /explanations/
# -------------------------------------------------------
def run_inference_with_explain(file_path: str) -> Tuple[str, float, str, str]:
    basename = os.path.basename(file_path)
    try:
        file_bytes = preprocess_file(file_path)
        raw_out = predict_with_explain(file_bytes, filename=basename)
    except Exception as e:
        print("[ERROR] predict_with_explain:", e)
        traceback.print_exc()
        return "error", 0.0, "unknown", ""

    label, confidence, model_version, explanation = normalize_output(raw_out)

    explanation_path = ""

    try:
        # ------------- CASE A: base64 "data:image" -------------
        if isinstance(explanation, str) and explanation.startswith("data:image"):
            os.makedirs("explanations", exist_ok=True)
            safe = basename.replace(" ", "_")
            explanation_path = os.path.join("explanations", f"exp_{safe}.jpg")
            with open(explanation_path, "wb") as f:
                f.write(base64.b64decode(explanation.split(",")[1]))

        # ------------- CASE B: raw bytes -------------
        elif isinstance(explanation, (bytes, bytearray)) and _is_valid_image_bytes(explanation):
            os.makedirs("explanations", exist_ok=True)
            safe = basename.replace(" ", "_")
            explanation_path = os.path.join("explanations", f"exp_{safe}.jpg")
            with open(explanation_path, "wb") as f:
                f.write(explanation)

        # ------------- CASE C: existing local file path -------------
        elif isinstance(explanation, str) and os.path.exists(explanation):
            explanation_path = explanation

        # else: explanation_path stays empty

    except Exception as e:
        print("[WARNING] Explanation save error:", e)
        traceback.print_exc()

    return (
        label or "unknown",
        float(confidence or 0.0),
        model_version or "unknown",
        explanation_path or "",
    )
