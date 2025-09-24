# model.py - DeepShield ML inference module (robust weight loading + reliable Grad-CAM)
# - Supports multiple checkpoint formats
# - Auto-detects a valid node for feature extraction (Grad-CAM)
# - Graceful fallbacks, extensive debug logging

import io
import os
import time
import base64
import traceback

from typing import Optional

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

# ---------------- CONFIG ----------------
MODEL_VERSION = "deepshield-efficientnet-b0-v1.1"
WEIGHTS_PATH_DEFAULT = "app/models/weights/deepfake_detector.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device: {device}, MODEL_VERSION: {MODEL_VERSION}")

# ---------------- MODEL DEFINITION ----------------
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = False):
        super().__init__()
        # torchvision EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=None if not pretrained else "DEFAULT")
        in_features = self.backbone.classifier[1].in_features
        # Match training checkpoint: Dropout + Linear only
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------- PREPROCESSING ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- SAFE LOAD IMPLEMENTATION ----------------
def _extract_state_dict(checkpoint) -> Optional[dict]:
    """
    Normalize checkpoint formats to a plain state_dict mapping parameter names -> tensors.
    Accepts:
      - plain state_dict (dict of tensors)
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - any dict that maps to tensor values
    Returns None on invalid format.
    """
    if checkpoint is None:
        return None

    # If the checkpoint is a dict with state keys, prefer common keys:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "state"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        # if dict looks like a state dict (values are tensors), return it
        # Heuristic: check if values are tensors
        sample_vals = list(checkpoint.values())[:5]
        if all(hasattr(v, "dtype") for v in sample_vals if v is not None):
            return checkpoint

    # Unknown format
    return None

def load_model(weights_path: str = WEIGHTS_PATH_DEFAULT) -> DeepfakeDetector:
    """
    Create model and attempt to load weights from weights_path.
    Supports several checkpoint formats and strips 'module.' prefixes automatically.
    Falls back to ImageNet-pretrained backbone if no valid weights are found.
    """
    model = DeepfakeDetector(num_classes=2, pretrained=True)

    try:
        if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
            print(f"[INFO] Attempting to load weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            state = _extract_state_dict(checkpoint)
            if state is None:
                print("[WARNING] Checkpoint format not recognised; attempting to use it directly as state_dict.")
                state = checkpoint if isinstance(checkpoint, dict) else None

            if isinstance(state, dict):
                # strip module. prefixes
                state = {k.replace("module.", ""): v for k, v in state.items()}

                # load only matching keys (handles small head differences)
                model_state = model.state_dict()
                matched = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
                missed_keys = [k for k in model_state if k not in matched]
                extra_keys = [k for k in state if k not in model_state]

                if len(matched) == 0:
                    print("[WARNING] No matching parameter keys between checkpoint and model. Checkpoint may be incompatible.")
                else:
                    model.load_state_dict(matched, strict=False)
                    print(f"[INFO] Loaded {len(matched)}/{len(model_state)} parameter tensors from checkpoint.")
                    if missed_keys:
                        print(f"[INFO] Example missing keys: {missed_keys[:5]} ...")
                    if extra_keys:
                        print(f"[INFO] Example extra keys in checkpoint: {extra_keys[:5]} ...")
            else:
                print("[WARNING] Could not normalize checkpoint into a state_dict. Using ImageNet-pretrained backbone.")
        else:
            print("[INFO] No valid weights file found (missing or empty). Using ImageNet-pretrained backbone.")
    except Exception as e:
        print(f"[ERROR] Exception while loading weights: {e}")
        traceback.print_exc()
        print("[FALLBACK] Continuing with ImageNet-pretrained backbone.")

    model.to(device).eval()
    return model

# instantiate once
model = load_model()

# ---------------- GRAD-CAM / FEATURE EXTRACTOR (robust) ----------------
feature_extractor = None
_explainability_node_name = None

def _find_candidate_from_graph(backbone: nn.Module) -> Optional[str]:
    """
    Use get_graph_node_names() to inspect eval nodes and pick a node.
    Strategy:
      1) prefer nodes that contain 'conv'
      2) else prefer nodes that contain 'features.' (like 'features.8')
      3) else prefer a node named 'features' (rare)
    """
    try:
        train_nodes, eval_nodes = get_graph_node_names(backbone)
    except Exception as e:
        print(f"[DEBUG] get_graph_node_names failed: {e}")
        return None

    # Debug: report node sample (last up to 40)
    if len(eval_nodes) > 0:
        print(f"[DEBUG] Eval nodes count: {len(eval_nodes)}; sample tail: {eval_nodes[-40:]}")

    # prefer nodes containing 'conv'
    conv_candidates = [n for n in eval_nodes if "conv" in n]
    if conv_candidates:
        return conv_candidates[-1]

    # else prefer nodes under 'features'
    features_candidates = [n for n in eval_nodes if n.startswith("features")]
    if features_candidates:
        # prefer the last concrete 'features.x' rather than 'features' or 'features.8' etc.
        return features_candidates[-1]

    # nothing found
    return None

def _find_candidate_from_named_modules(backbone: nn.Module) -> Optional[str]:
    """
    Inspect named_modules() for last Conv2d module - return its name.
    """
    conv_names = [name for name, m in backbone.named_modules() if isinstance(m, nn.Conv2d) and name]
    if conv_names:
        return conv_names[-1]
    return None

# Try multiple strategies, with debug prints
try:
    # 1) try a commonly used node for EfficientNet (fast path)
    candidate = "features.8"  # many torchvision implementations expose this
    try:
        feature_extractor = create_feature_extractor(model.backbone, return_nodes={candidate: "features"})
        _explainability_node_name = candidate
        print(f"[INFO] Grad-CAM: using hardcoded candidate '{candidate}'")
    except Exception:
        # 2) try graph node detection
        candidate = _find_candidate_from_graph(model.backbone)
        if candidate:
            try:
                feature_extractor = create_feature_extractor(model.backbone, return_nodes={candidate: "features"})
                _explainability_node_name = candidate
                print(f"[INFO] Grad-CAM: using graph-detected node '{candidate}'")
            except Exception as e:
                print(f"[WARNING] create_feature_extractor failed for graph candidate '{candidate}': {e}")
                traceback.print_exc()
                feature_extractor = None

        # 3) fallback to named_modules conv search
        if feature_extractor is None:
            candidate = _find_candidate_from_named_modules(model.backbone)
            if candidate:
                try:
                    feature_extractor = create_feature_extractor(model.backbone, return_nodes={candidate: "features"})
                    _explainability_node_name = candidate
                    print(f"[INFO] Grad-CAM: using named_modules-detected conv '{candidate}'")
                except Exception as e:
                    print(f"[WARNING] create_feature_extractor failed for named_modules candidate '{candidate}': {e}")
                    traceback.print_exc()
                    feature_extractor = None

except Exception as e:
    print(f"[ERROR] Unexpected error while initializing Grad-CAM: {e}")
    traceback.print_exc()
    feature_extractor = None

if feature_extractor is None:
    print("[INFO] Explainability (Grad-CAM) disabled — snapshots won't be produced. This is non-fatal.")
else:
    print(f"[INFO] Explainability enabled. Hook node: {_explainability_node_name}")

# ---------------- PREDICT (no explain) ----------------
def predict(file_bytes: bytes, filename: str = None):
    """
    Basic forward pass returning label/confidence/latency.
    Returns consistent dict format even on errors.
    """
    # load image bytes
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        print("[ERROR] PIL open failed:", e)
        traceback.print_exc()
        return {"filename": filename, "label": "error", "confidence": 0.0, "model_version": MODEL_VERSION, "latency_ms": 0.0, "error": str(e)}

    try:
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            start = time.time()
            outputs = model(img_tensor)
            latency_ms = (time.time() - start) * 1000
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
        label = "real" if pred_idx == 0 else "deepfake"
        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(latency_ms, 2)
        }
    except Exception as e:
        print("[ERROR] Inference failed:", e)
        traceback.print_exc()
        return {"filename": filename, "label": "error", "confidence": 0.0, "model_version": MODEL_VERSION, "latency_ms": 0.0, "error": str(e)}

# ---------------- PREDICT WITH EXPLAIN (Grad-CAM) ----------------
def predict_with_explain(file_bytes: bytes, filename: str = None):
    """
    Forward pass + optional explainability snapshot (data URI).
    If explainability is disabled or fails, returns "No explainability snapshot available".
    """
    # open image
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        print("[ERROR] PIL open failed:", e)
        traceback.print_exc()
        return {"filename": filename, "label": "error", "confidence": 0.0, "model_version": MODEL_VERSION, "latency_ms": 0.0, "explainability": "No explainability snapshot available", "error": str(e)}

    try:
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            start = time.time()
            feats = feature_extractor(img_tensor) if feature_extractor is not None else {}
            outputs = model(img_tensor)
            latency_ms = (time.time() - start) * 1000
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
        label = "real" if pred_idx == 0 else "deepfake"
    except Exception as e:
        print("[ERROR] Inference failed:", e)
        traceback.print_exc()
        return {"filename": filename, "label": "error", "confidence": 0.0, "model_version": MODEL_VERSION, "latency_ms": 0.0, "explainability": "No explainability snapshot available", "error": str(e)}

    # If no feature extractor or features missing, return placeholder
    if feature_extractor is None or not feats or "features" not in feats:
        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(latency_ms, 2),
            "explainability": "No explainability snapshot available"
        }

    # Build a simple Grad-CAM-like heatmap (channel-mean) — fast and robust
    try:
        fmap = feats["features"].squeeze().cpu().numpy()  # [C,H,W]
        heatmap = np.mean(fmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        denom = heatmap.max() if heatmap.max() != 0 else 1.0
        heatmap = heatmap / (denom + 1e-8)

        # overlay
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

        _, buf = cv2.imencode(".jpg", overlay)
        b64_overlay = base64.b64encode(buf).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_overlay}"

        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(latency_ms, 2),
            "explainability": data_uri
        }
    except Exception as e:
        print("[WARNING] Failed to generate explainability snapshot:", e)
        traceback.print_exc()
        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(latency_ms, 2),
            "explainability": "No explainability snapshot available"
        }
