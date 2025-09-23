# model.py - DeepShield ML inference module (stable for EfficientNet B0)

import io
import os
import time
import base64
import traceback

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

# ---------------- CONFIG ----------------
MODEL_VERSION = "deepshield-efficientnet-b0-v1.0"
WEIGHTS_PATH_DEFAULT = "app/models/weights/deepfake_detector.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device: {device}, MODEL_VERSION: {MODEL_VERSION}")

# ---------------- MODEL ----------------
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights="DEFAULT" if pretrained else None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# ---------------- PREPROCESS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- SAFE LOAD ----------------
def load_model(weights_path: str = WEIGHTS_PATH_DEFAULT) -> DeepfakeDetector:
    model = DeepfakeDetector(num_classes=2, pretrained=True)
    try:
        if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
            state = torch.load(weights_path, map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {k.replace("module.", ""): v for k, v in state.items()}

            # Partial loading: only keys that exist
            model_dict = model.state_dict()
            state_to_load = {k: v for k, v in state.items() if k in model_dict}
            missing_keys = [k for k in model_dict if k not in state_to_load]
            if missing_keys:
                print(f"[INFO] Missing keys (will use pretrained for them): {missing_keys[:5]} ...")

            model.load_state_dict(state_to_load, strict=False)
            print(f"[INFO] Loaded custom weights from {weights_path}")
        else:
            print("[INFO] No weights found. Using ImageNet-pretrained backbone.")
    except Exception as e:
        print(f"[ERROR] Unexpected error loading weights: {e}")
        traceback.print_exc()
    model.to(device).eval()
    return model

model = load_model()

# ---------------- GRAD-CAM / FEATURE EXTRACTOR ----------------
feature_extractor = None
_explainability_node_name = None

def _get_last_conv(backbone: nn.Module):
    convs = [name for name, m in backbone.named_modules() if isinstance(m, nn.Conv2d)]
    return convs[-1] if convs else None

try:
    candidate = "features.8"  # safe for EfficientNet-B0
    feature_extractor = create_feature_extractor(model.backbone, return_nodes={candidate: "features"})
    _explainability_node_name = candidate
    print(f"[INFO] Grad-CAM: using node '{candidate}' for explainability")
except Exception:
    # fallback to auto-detect
    candidate = _get_last_conv(model.backbone)
    if candidate:
        try:
            feature_extractor = create_feature_extractor(model.backbone, return_nodes={candidate: "features"})
            _explainability_node_name = candidate
            print(f"[INFO] Grad-CAM: fallback auto-detected node '{candidate}'")
        except Exception:
            feature_extractor = None

if feature_extractor is None:
    print("[INFO] Explainability (Grad-CAM) disabled — snapshots will not be produced")

# ---------------- PREDICTION ----------------
def predict(file_bytes: bytes, filename: str = None):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            start = time.time()
            outputs = model(img_tensor)
            elapsed = (time.time() - start) * 1000
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
        label = "real" if pred_idx == 0 else "deepfake"
        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(elapsed, 2)
        }
    except Exception as e:
        print("[ERROR] Inference failed:", e)
        traceback.print_exc()
        return {"filename": filename, "label": "error", "confidence": 0.0, "error": str(e)}

# ---------------- PREDICTION WITH EXPLAIN ----------------
def predict_with_explain(file_bytes: bytes, filename: str = None):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            start = time.time()
            feats = feature_extractor(img_tensor) if feature_extractor else {}
            outputs = model(img_tensor)
            elapsed = (time.time() - start) * 1000
            probs = torch.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())
            confidence = float(probs[pred_idx].item())
        label = "real" if pred_idx == 0 else "deepfake"

        # if Grad-CAM failed or not available
        if not feature_extractor or "features" not in feats:
            return {
                "filename": filename,
                "label": label,
                "confidence": round(confidence, 4),
                "model_version": MODEL_VERSION,
                "latency_ms": round(elapsed, 2),
                "explainability": "No explainability snapshot available"
            }

        # create Grad-CAM heatmap
        fmap = feats["features"].squeeze().cpu().numpy()
        heatmap = np.maximum(np.mean(fmap, axis=0), 0)
        heatmap /= (heatmap.max() + 1e-8)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255*heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

        _, buf = cv2.imencode(".jpg", overlay)
        b64_overlay = base64.b64encode(buf).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_overlay}"

        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(elapsed, 2),
            "explainability": data_uri
        }

    except Exception as e:
        print("[WARNING] Explainability failed:", e)
        traceback.print_exc()
        return {
            "filename": filename,
            "label": label,
            "confidence": round(confidence, 4),
            "model_version": MODEL_VERSION,
            "latency_ms": round(elapsed, 2),
            "explainability": "No explainability snapshot available"
        }
