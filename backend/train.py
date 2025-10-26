# train.py - DeepShield training script (merged dataset)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Import your DeepfakeDetector model
from app.models.model import DeepfakeDetector

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

num_epochs = 5
batch_size = 4
learning_rate = 1e-4
image_size = 224

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------- DATASETS ----------------
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
val_dataset   = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"✅ Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

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
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")

    # ---------------- VALIDATION ----------------
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
