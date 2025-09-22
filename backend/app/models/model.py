# model.py - DeepShield ML inference module (extended with filename + explainability)

import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time, base64
import numpy as np
import cv2
from torchvision.models.feature_extraction import create_feature_extractor

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION = "deepshield-efficientnet-b0-v0.2"

# ------------------------------
# Model definition
# ------------------------------
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
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

# ------------------------------
# Preprocessing
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# Load model once at import
# ------------------------------
def load_model(weights_path="deepfake_detector.pth"):
    model = DeepfakeDetector(num_classes=2, pretrained=True)
    try:
        state = torch.load(weights_path, map_location=device)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        print(f"[INFO] Loaded weights from {weights_path}")
    except FileNotFoundError:
        print("[WARNING] No weights found, using ImageNet pretrained model only")
    model.to(device).eval()
    return model

# Global model
model = load_model()

# Grad-CAM setup
target_layers = {"features": "features.6.3"}  # last conv block
feature_extractor = create_feature_extractor(model, return_nodes=target_layers)

# ------------------------------
# Predict function
# ------------------------------
def predict(file_bytes: bytes, filename: str = None):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image input: {e}", "filename": filename}

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        outputs = model(img_tensor)
        elapsed = (time.time() - start) * 1000
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    label = "real" if pred_idx == 0 else "deepfake"

    return {
        "filename": filename,
        "label": label,
        "confidence": round(confidence, 4),
        "model_version": MODEL_VERSION,
        "latency_ms": round(elapsed, 2)
    }

# ------------------------------
# Predict + Explainability
# ------------------------------
def predict_with_explain(file_bytes: bytes, filename: str = None):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image input: {e}", "filename": filename}

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        feats = feature_extractor(img_tensor)
        outputs = model(img_tensor)
        elapsed = (time.time() - start) * 1000
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    label = "real" if pred_idx == 0 else "deepfake"

    # Grad-CAM (simple)
    fmap = feats["features"].squeeze().cpu().numpy()
    heatmap = np.mean(fmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    # Overlay heatmap on image
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (img_cv.shape[1], img_cv.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.6, heatmap_colored, 0.4, 0)

    _, buf = cv2.imencode(".jpg", overlay)
    b64_overlay = base64.b64encode(buf).decode("utf-8")

    return {
        "filename": filename,
        "label": label,
        "confidence": round(confidence, 4),
        "model_version": MODEL_VERSION,
        "latency_ms": round(elapsed, 2),
        "explainability": f"data:image/jpeg;base64,{b64_overlay}"
    }
