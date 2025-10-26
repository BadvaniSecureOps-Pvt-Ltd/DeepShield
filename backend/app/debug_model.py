# debug_model.py
import os
from PIL import Image
import torch
from torchvision import transforms
import cv2
from models.model import DeepfakeDetector  # Import your model class

# ----------------------------
# CONFIG - adjust paths
# ----------------------------
MODEL_WEIGHTS = "models/weights/deepfake_detector.pth"
# Updated paths
REAL_IMAGES_DIR = "../data/train/real"
FAKE_IMAGES_DIR = "../data/train/fake"
FACE_DETECT = True  # True if your model expects face crops
# ----------------------------

# ----------------------------
# Load model
# ----------------------------
model = DeepfakeDetector()  # Create model instance
state_dict = torch.load(MODEL_WEIGHTS, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully.\n")

# Preprocessing (adjust to match your training!)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_face(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    face_img = cv2.cvtColor(img[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_img)

def predict(image_path):
    if FACE_DETECT:
        img = extract_face(image_path)
        if img is None:
            print(f"No face detected in {image_path}")
            return None, None
    else:
        img = Image.open(image_path).convert("RGB")
    img = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
    return pred_class.item(), confidence.item()

# ----------------------------
# Test images
# ----------------------------
for folder, expected_label in [(REAL_IMAGES_DIR, "real"), (FAKE_IMAGES_DIR, "deepfake")]:
    print(f"\nTesting images in {folder} (expected: {expected_label})")
    for img_file in os.listdir(folder)[:10]:  # test first 10 images
        img_path = os.path.join(folder, img_file)
        pred, conf = predict(img_path)
        if pred is not None:
            print(f"{img_file}: Predicted={'real' if pred==0 else 'deepfake'}, Confidence={conf:.2f} (Expected={expected_label})")
