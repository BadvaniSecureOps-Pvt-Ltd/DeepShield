# train_video.py
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.models.video import r3d_18
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ---------------- Dataset ----------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, transform=None):
        self.samples = []
        self.clip_len = clip_len
        self.transform = transform
        for label, cls in enumerate(['real', 'fake']):
            folder = os.path.join(root_dir, cls)
            for file in os.listdir(folder):
                if file.endswith('.mp4'):
                    self.samples.append((os.path.join(folder, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._load_video(path)
        if self.transform:
            frames = torch.stack([self.transform(f) for f in frames])
        frames = frames.permute(1, 0, 2, 3)  # (C, T, H, W)
        return frames, label

    def _load_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while len(frames) < self.clip_len:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        cap.release()
        if len(frames) < self.clip_len and frames:
            frames += [frames[-1]] * (self.clip_len - len(frames))
        return frames[:self.clip_len]

# ---------------- Train Function ----------------
def train_video_model(data_dir="data", epochs=5, batch_size=4, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_ds = VideoDataset(os.path.join(data_dir, "train"), transform=transform)
    val_ds = VideoDataset(os.path.join(data_dir, "val"), transform=transform)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = r3d_18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for clips, labels in train_dl:
            clips, labels = clips.to(device), labels.to(device)
            out = model(clips)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_dl):.4f}")

    os.makedirs("app/models/weights", exist_ok=True)
    torch.save(model.state_dict(), "app/models/weights/deepfake_video_r3d18.pth")
    print("✅ Model saved: app/models/weights/deepfake_video_r3d18.pth")

if __name__ == "__main__":
    train_video_model()
