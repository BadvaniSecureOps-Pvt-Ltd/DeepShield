# app/models/video_inference.py
import torch
import cv2
import numpy as np
from torchvision.models.video import r3d_18
import torch.nn as nn
import os

def load_video_model(weight_path="app/models/weights/deepfake_video_r3d18.pth"):
    model = r3d_18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()
    return model

def predict_video(file_bytes, model):
    tmp_path = "/tmp/temp_video.mp4"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)

    cap = cv2.VideoCapture(tmp_path)
    frames = []
    while len(frames) < 16:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (128, 128))
        frames.append(frame)
    cap.release()
    os.remove(tmp_path)

    if not frames:
        return {"label": "error", "confidence": 0.0}

    frames = np.stack(frames, axis=0)
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        preds = model(frames)
        conf = torch.softmax(preds, dim=1)
        label = "real" if conf[0][0] > conf[0][1] else "deepfake"
        confidence = float(conf[0].max())

    return {"label": label, "confidence": round(confidence, 4)}
