# models.py — DeepShield Hybrid Model (CLEAN, COMPATIBLE WITH INFERENCE.PY)
import io
import os
import time
import uuid
import base64
import traceback
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor

# ---------------- CONFIG ----------------
MODEL_VERSION = "deepshield-hybrid-v1.0"

WEIGHTS_PATH_A = "app/models/weights/deepfake_detector_best.pth"
WEIGHTS_PATH_B = "app/models/weights/aiimage_resnet50.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] device: {device}, MODEL_VERSION: {MODEL_VERSION}")

TEMPERATURE = 1.5
FUSION_THRESHOLD = 0.60

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------- MODELS ----------------
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )
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


class AIImageDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def _extract(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ["state_dict", "model_state_dict"]:
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_model_a():
    model = DeepfakeDetector()
    if os.path.exists(WEIGHTS_PATH_A):
        print("[INFO] Loading Model A weights")
        ckpt = _extract(torch.load(WEIGHTS_PATH_A, map_location=device))
        model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=False)
    else:
        print("[WARN] Model A weights missing")
    return model.to(device).eval()


def load_model_b():
    model = AIImageDetector()
    if os.path.exists(WEIGHTS_PATH_B):
        print("[INFO] Loading Model B weights")
        ckpt = _extract(torch.load(WEIGHTS_PATH_B, map_location=device))
        model.load_state_dict({k.replace("module.", ""): v for k, v in ckpt.items()}, strict=False)
    else:
        print("[WARN] Model B weights missing")
    return model.to(device).eval()


model_a = load_model_a()
model_b = load_model_b()

# ---------------- GRAD-CAM FOR MODEL A ----------------
try:
    feature_extractor_a = create_feature_extractor(
        model_a.backbone, return_nodes={"features.8": "feat"}
    )
    print("[INFO] Grad-CAM enabled.")
except:
    feature_extractor_a = None
    print("[WARN] Grad-CAM disabled.")


def _preprocess(img):
    return transform(img).unsqueeze(0).to(device)


def _compute_logits(model, tensor):
    with torch.no_grad():
        out = model(tensor)
    logits = out[0].cpu().numpy()
    fake_prob = torch.softmax(out / TEMPERATURE, dim=1)[0, 0].item()
    return logits, float(fake_prob)


# ---------------- PREDICT (FLAT FORMAT) ----------------
def predict(file_bytes: bytes, filename=None):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = _preprocess(img)

        logits_a, fake_a = _compute_logits(model_a, tensor)
        logits_b, fake_b = _compute_logits(model_b, tensor)

        fused_fake = max(fake_a, fake_b)
        fused_real = 1 - fused_fake
        label = "deepfake" if fused_fake >= FUSION_THRESHOLD else "real"
        confidence = fused_fake if label == "deepfake" else fused_real

        return {
            "label": label,
            "confidence": float(confidence),
            "model_version": MODEL_VERSION,
        }

    except Exception as e:
        print("[ERROR predict]:", e)
        return {"label": "error", "confidence": 0.0, "model_version": MODEL_VERSION}


# ---------------- PREDICT WITH EXPLAIN ----------------
def predict_with_explain(file_bytes: bytes, filename=None):
    base = predict(file_bytes, filename)

    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        tensor = _preprocess(img)

        explanation_jpeg = None

        if feature_extractor_a:
            feats = feature_extractor_a(tensor)
            fmap = list(feats.values())[0].squeeze().detach().cpu().numpy()
            heat = np.maximum(fmap.mean(axis=0), 0)
            heat /= heat.max() + 1e-8

            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            heat_resized = cv2.resize(heat, (img_cv.shape[1], img_cv.shape[0]))
            heatmap = cv2.applyColorMap((heat_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_cv, 0.6, heatmap, 0.4, 0)

            _, buffer = cv2.imencode(".jpg", overlay)
            explanation_jpeg = "data:image/jpeg;base64," + base64.b64encode(buffer).decode()

        return {
            "label": base["label"],
            "confidence": base["confidence"],
            "model_version": MODEL_VERSION,
            "explainability": explanation_jpeg,
        }

    except Exception as e:
        print("[ERROR explain]:", e)
        return {
            "label": base["label"],
            "confidence": base["confidence"],
            "model_version": MODEL_VERSION,
            "explainability": None,
        }
