"""
Melatih model klasifikasi sampah elektronik menggunakan MobileNetV3-Small.
Dataset: modified-dataset/ dengan 10 kelas.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from pathlib import Path

# ─── Konfigurasi ─────────────────────────────────────────────────────────────
DATASET_DIR = Path("../modified-dataset")
MODEL_DIR   = Path("model")
MODEL_PATH  = MODEL_DIR / "ewaste_model.pth"
META_PATH   = MODEL_DIR / "model_meta.json"

BATCH_SIZE  = 32
NUM_EPOCHS  = 10
LR          = 1e-3
IMG_SIZE    = 224
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR.mkdir(exist_ok=True)

# ─── Augmentasi & Transform ───────────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ─── Dataset ─────────────────────────────────────────────────────────────────
train_ds = datasets.ImageFolder(DATASET_DIR / "train", transform=train_transform)
val_ds   = datasets.ImageFolder(DATASET_DIR / "val",   transform=val_transform)
test_ds  = datasets.ImageFolder(DATASET_DIR / "test",  transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

CLASS_NAMES = train_ds.classes
NUM_CLASSES = len(CLASS_NAMES)
print(f"Kelas ({NUM_CLASSES}): {CLASS_NAMES}", flush=True)
print(f"Perangkat: {DEVICE}", flush=True)

# ─── Model ────────────────────────────────────────────────────────────────────
model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

# Ganti classifier head
in_features = model.classifier[3].in_features
model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# ─── Training ─────────────────────────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

best_val_acc = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    model.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += inputs.size(0)

    # Validate
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total += inputs.size(0)

    t_acc = 100 * train_correct / train_total
    v_acc = 100 * val_correct   / val_total
    t_loss = train_loss / train_total
    v_loss = val_loss   / val_total
    scheduler.step()

    print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | "
          f"Train Loss: {t_loss:.4f} Acc: {t_acc:.1f}% | "
          f"Val Loss: {v_loss:.4f} Acc: {v_acc:.1f}%", flush=True)

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "num_classes": NUM_CLASSES,
            "img_size": IMG_SIZE,
            "val_acc": v_acc,
        }, MODEL_PATH)
        print(f"  ✓ Model terbaik disimpan (val acc: {v_acc:.1f}%)")

# ─── Test ─────────────────────────────────────────────────────────────────────
print("\n=== Evaluasi pada data test ===")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
test_correct, test_total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        test_total   += inputs.size(0)

test_acc = 100 * test_correct / test_total
print(f"Test Accuracy: {test_acc:.2f}%")

# Simpan metadata
meta = {
    "class_names": CLASS_NAMES,
    "num_classes": NUM_CLASSES,
    "img_size": IMG_SIZE,
    "best_val_acc": best_val_acc,
    "test_acc": test_acc,
    "architecture": "MobileNetV3-Small",
}
with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print(f"\nSelesai! Model disimpan di: {MODEL_PATH}")
print(f"Metadata: {META_PATH}")
