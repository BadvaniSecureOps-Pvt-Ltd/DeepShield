# train.py - DeepShield GPU-optimized training script
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import your DeepfakeDetector model
from app.models.model import DeepfakeDetector

# ---------------- CONFIG ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INFO] GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] CUDA Version: {torch.version.cuda}")

data_dir = "data"
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

num_epochs = 10           # can increase later
batch_size = 32           # safe for H100 (adjust if OOM)
learning_rate = 1e-4
image_size = 224
num_workers = 8           # multi-threaded data loading

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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

print(f"✅ Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# ---------------- MODEL ----------------
model = DeepfakeDetector(num_classes=2, pretrained=True).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ---------------- TRAINING LOOP ----------------
scaler = torch.cuda.amp.GradScaler()  # for mixed precision
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

    for imgs, labels in progress:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():  # AMP context
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"\n[TRAIN] Epoch {epoch+1}/{num_epochs} | Avg Loss: {avg_loss:.4f}")

    # ---------------- VALIDATION ----------------
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"[VAL] Accuracy: {acc:.2f}%")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        weights_path = "app/models/weights/deepfake_detector_best.pth"
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"✅ New best model saved! Accuracy: {best_acc:.2f}%")

print(f"\n🏁 Training complete. Best validation accuracy: {best_acc:.2f}%")
