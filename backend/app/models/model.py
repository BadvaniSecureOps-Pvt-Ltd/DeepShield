# model.py - DeepShield ML inference module

import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import time

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION = "deepshield-efficientnet-b0-v0.1"

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

# ------------------------------
# Predict function
# ------------------------------
def predict(file_bytes: bytes):
    try:
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as e:
        return {"error": f"Invalid image input: {e}"}

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
        "label": label,
        "confidence": round(confidence, 4),
        "model_version": MODEL_VERSION,
        "latency_ms": round(elapsed, 2)
    }
