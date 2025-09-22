import os
from typing import Tuple
from app.models.model import predict_with_explain


def run_inference(file_path: str) -> Tuple[str, float, str, str]:
    """
    Run ML inference on the uploaded file using DeepShield model.
    
    Returns:
        label (str): "real" or "deepfake"
        confidence (float): confidence score between 0 and 1
        model_version (str): version string
        explanation_path (str): path to saved heatmap image for explainability
    """

    basename = os.path.basename(file_path)

    # --- Read file bytes safely ---
    try:
        with open(file_path, "rb") as f:
            file_bytes = f.read()
    except Exception as e:
        # Return default values if reading fails
        return "unknown", 0.0, "v1.0.0", ""

    # --- Call the ML model ---
    try:
        result = predict_with_explain(file_bytes, filename=basename)
    except Exception as e:
        # In case the model fails, return safe defaults
        return "error", 0.0, "v1.0.0", ""

    # --- Extract results ---
    label = result.get("label", "unknown")
    confidence = result.get("confidence", 0.0)
    model_version = result.get("model_version", "v1.0.0")
    explanation_base64 = result.get("explainability", None)

    # --- Save explanation snapshot as a file for API response ---
    explanation_dir = "explanations"
    os.makedirs(explanation_dir, exist_ok=True)
    explanation_path = os.path.join(explanation_dir, f"exp_{basename}.jpg")

    if explanation_base64:
        try:
            import base64
            with open(explanation_path, "wb") as f:
                f.write(base64.b64decode(explanation_base64.split(",")[1]))
        except Exception:
            # If saving fails, keep explanation_path empty
            explanation_path = ""

    return label, confidence, model_version, explanation_path
