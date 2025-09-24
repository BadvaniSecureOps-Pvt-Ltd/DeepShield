# models.py - DeepShield ML inference module (fixed + Grad-CAM)
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
from torchvision.models.feature_extraction import create_feature_extractor

# ---------------- CONFIG ----------------
MODEL_VERSION = "deepshield-efficientnet-b0-v1.3"
WEIGHTS_PATH_DEFAULT = "app/models/weights/deepfake_detector.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device: {device}, MODEL_VERSION: {MODEL_VERSION}")

# ---------------- MODEL DEFINITION ----------------
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
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

# ---------------- WEIGHT LOADING ----------------
def _extract_state_dict(checkpoint) -> Optional[dict]:
    if checkpoint is None:
        return None
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "state"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
        sample_vals = list(checkpoint.values())[:5]
        if all(hasattr(v, "dtype") for v in sample_vals if v is not None):
            return checkpoint
    return None

def load_model(weights_path: str = WEIGHTS_PATH_DEFAULT) -> DeepfakeDetector:
    model = DeepfakeDetector(num_classes=2, pretrained=True)
    try:
        if os.path.exists(weights_path) and os.path.getsize(weights_path) > 0:
            print(f"[INFO] Loading weights from: {weights_path}")
            checkpoint = torch.load(weights_path, map_location=device)
            state = _extract_state_dict(checkpoint) or checkpoint
            if isinstance(state, dict):
                state = {k.replace("module.", ""): v for k, v in state.items()}
                model_state = model.state_dict()
                matched = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
                model.load_state_dict(matched, strict=False)
                print(f"[INFO] Loaded {len(matched)}/{len(model_state)} tensors.")
        else:
            print("[INFO] No valid weights found, using ImageNet-pretrained backbone.")
    except Exception as e:
        print(f"[ERROR] Failed loading weights: {e}")
        traceback.print_exc()
    model.to(device).eval()
    return model

# ---------------- PREPROCESS ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------- INSTANTIATE ----------------
model = load_model()

# ---------------- GRAD-CAM / FEATURE EXTRACTOR ----------------
feature_extractor = None
_explainability_node_name = None

def _find_last_conv(backbone: nn.Module) -> Optional[str]:
    conv_layers = [name for name, m in backbone.named_modules() if isinstance(m, nn.Conv2d)]
    print(f"[INFO] All Conv2d nodes: {conv_layers}")
    return conv_layers[-1] if conv_layers else None

try:
    # Hardcode to confirmed correct node 'features.8'
    _explainability_node_name = "features.8"
    feature_extractor = create_feature_extractor(model.backbone, return_nodes={_explainability_node_name: "features"})
    print(f"[INFO] Grad-CAM using node '{_explainability_node_name}'")
except Exception as e:
    print(f"[WARNING] Grad-CAM init failed: {e}")
    feature_extractor = None

if feature_extractor is None:
    print("[INFO] Explainability disabled — snapshots will not be generated.")
else:
    print(f"[INFO] Explainability enabled. Node: {_explainability_node_name}")

# ---------------- PREDICT ----------------
def predict(file_bytes: bytes, filename: str = None):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
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

# ---------------- PREDICT WITH EXPLAIN ----------------
def predict_with_explain(file_bytes: bytes, filename: str = None):
    result = predict(file_bytes, filename)
    if feature_extractor is None:
        result["explainability"] = "No explainability snapshot available"
        return result

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feats = feature_extractor(img_tensor)
        fmap = feats["features"].squeeze().cpu().numpy()
        heatmap = np.maximum(np.mean(fmap, axis=0), 0)
        heatmap /= heatmap.max() + 1e-8

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

        _, buf = cv2.imencode(".jpg", overlay)
        b64_overlay = base64.b64encode(buf).decode("utf-8")
        result["explainability"] = f"data:image/jpeg;base64,{b64_overlay}"
    except Exception as e:
        print("[WARNING] Grad-CAM failed:", e)
        result["explainability"] = "No explainability snapshot available"
        traceback.print_exc()
    return result
