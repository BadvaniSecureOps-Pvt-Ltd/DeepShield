# inference.py - DeepShield inference + explainability handler
import os
import io
import base64
from typing import Tuple
from PIL import Image

from app.models.model import predict_with_explain

# optional: HEIC support
try:
    import pillow_heif
except ImportError:
    pillow_heif = None

# optional: PDF image extraction
try:
    from extract_images import extract_images_from_pdf
except ImportError:
    extract_images_from_pdf = None

def preprocess_file(file_path: str) -> bytes:
    """Read and preprocess a file (image/PDF/HEIC) into raw bytes."""
    filename = file_path.lower()
    with open(file_path, "rb") as f:
        content = f.read()

    # PDF
    if filename.endswith(".pdf"):
        if not extract_images_from_pdf:
            raise ValueError("PDF extraction module missing")
        img_bytes_list = extract_images_from_pdf(io.BytesIO(content))
        if not img_bytes_list:
            raise ValueError("No images found in PDF")
        content = img_bytes_list[0]

    # HEIC
    elif filename.endswith(".heic"):
        if pillow_heif is None:
            raise ValueError("HEIC support missing, install pillow_heif")
        heif_file = pillow_heif.read_heif(content)
        img = Image.frombytes(
            mode=heif_file.mode,
            size=heif_file.size,
            data=heif_file.data,
        )
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        content = buf.getvalue()

    # Validate image
    try:
        Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Unsupported or corrupted image: {e}")

    return content

def run_inference(file_path: str) -> Tuple[str, float, str, str]:
    """
    Standard inference (without returning explainability snapshot).
    Returns: label, confidence, model_version, explanation_path (empty)
    """
    basename = os.path.basename(file_path)
    explanation_path = ""

    try:
        file_bytes = preprocess_file(file_path)
        result = predict_with_explain(file_bytes, filename=basename)
        label = result.get("label", "unknown")
        confidence = result.get("confidence", 0.0)
        model_version = result.get("model_version", "v1.0.0")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        return "error", 0.0, "v1.0.0", ""

    return label, confidence, model_version, explanation_path


def run_inference_with_explain(file_path: str) -> Tuple[str, float, str, str]:
    """
    Inference + Grad-CAM explainability snapshot.
    Returns: label, confidence, model_version, explanation_path (JPEG saved file)
    """
    basename = os.path.basename(file_path)
    explanation_path = ""

    try:
        file_bytes = preprocess_file(file_path)
        result = predict_with_explain(file_bytes, filename=basename)
        label = result.get("label", "unknown")
        confidence = result.get("confidence", 0.0)
        model_version = result.get("model_version", "v1.0.0")
        explanation_base64 = result.get("explainability", None)
    except Exception as e:
        print(f"[ERROR] predict_with_explain failed: {e}")
        return "error", 0.0, "v1.0.0", ""

    # Save explainability snapshot
    if explanation_base64 and explanation_base64.startswith("data:image"):
        try:
            explanation_dir = "explanations"
            os.makedirs(explanation_dir, exist_ok=True)
            safe_name = basename.replace(" ", "_")
            explanation_path = os.path.join(explanation_dir, f"exp_{safe_name}")
            if not explanation_path.lower().endswith(".jpg"):
                explanation_path += ".jpg"
            with open(explanation_path, "wb") as f:
                f.write(base64.b64decode(explanation_base64.split(",")[1]))
        except Exception as e:
            print(f"[WARNING] Saving explainability snapshot failed: {e}")
            explanation_path = ""

    return label, confidence, model_version, explanation_path
