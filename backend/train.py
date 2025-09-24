# train.py - DeepShield training script (updated for tensor load success)
import os
import fitz  # PyMuPDF
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Import your DeepfakeDetector model
from app.models.model import DeepfakeDetector

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real_pdf = "real_images.pdf"
fake_pdf = "fake_images.pdf"
data_dir = "data"
train_ratio = 0.8
num_epochs = 5
batch_size = 32
learning_rate = 1e-4

# ---------------- EXTRACT IMAGES ----------------
def extract_images_from_pdf(pdf_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    pdf = fitz.open(pdf_path)
    count = 0
    for page_number in range(len(pdf)):
        page = pdf[page_number]
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            image_filename = os.path.join(output_folder, f"{page_number+1}_{img_index+1}.jpg")
            with open(image_filename, "wb") as f:
                f.write(image_bytes)
            count += 1
    print(f"Extracted {count} images from {pdf_path} to {output_folder}")

def prepare_dataset(real_pdf, fake_pdf, data_dir, train_ratio=0.8):
    folders = ["train/real", "train/fake", "val/real", "val/fake"]
    for f in folders:
        os.makedirs(os.path.join(data_dir, f), exist_ok=True)

    temp_real = os.path.join(data_dir, "temp_real")
    temp_fake = os.path.join(data_dir, "temp_fake")
    if os.path.exists(temp_real): shutil.rmtree(temp_real)
    if os.path.exists(temp_fake): shutil.rmtree(temp_fake)

    extract_images_from_pdf(real_pdf, temp_real)
    extract_images_from_pdf(fake_pdf, temp_fake)

    def split_and_move(temp_folder, train_folder, val_folder):
        images = [f for f in os.listdir(temp_folder) if f.lower().endswith('.jpg')]
        train_imgs, val_imgs = train_test_split(images, train_size=train_ratio, random_state=42)
        for img in train_imgs:
            shutil.move(os.path.join(temp_folder, img), os.path.join(train_folder, img))
        for img in val_imgs:
            shutil.move(os.path.join(temp_folder, img), os.path.join(val_folder, img))

    split_and_move(temp_real, os.path.join(data_dir, "train/real"), os.path.join(data_dir, "val/real"))
    split_and_move(temp_fake, os.path.join(data_dir, "train/fake"), os.path.join(data_dir, "val/fake"))

    shutil.rmtree(temp_real)
    shutil.rmtree(temp_fake)
    print("Dataset prepared successfully.")

# ---------------- PREPARE DATA ----------------
prepare_dataset(real_pdf, fake_pdf, data_dir, train_ratio)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=transform)
val_dataset   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ---------------- MODEL ----------------
model = DeepfakeDetector(num_classes=2, pretrained=True).to(device)

# ---------------- TRAINING ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: Train Loss={total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = correct / total * 100
    print(f"Validation Accuracy: {acc:.2f}%")

# ---------------- SAVE WEIGHTS ----------------
weights_path = "app/models/weights/deepfake_detector.pth"
os.makedirs(os.path.dirname(weights_path), exist_ok=True)
torch.save(model.state_dict(), weights_path)
print(f"✅ Weights saved to {weights_path}")
