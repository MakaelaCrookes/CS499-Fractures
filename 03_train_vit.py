# Imports
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import random

# Config
DATA_PATH = "data/FracAtlas/images/"
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 5e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# Cust dataset
class SafeImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.classes = ['Fractured', 'Non_fractured']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_path = os.path.join(root, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Found {len(self.samples)} valid images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, target
        except Exception:
            img = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                img = self.transform(img)
            return img, target

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
print("\nLoading dataset...")
full_dataset = SafeImageFolder(DATA_PATH, transform=transform)

# Split into train/val/test
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

print(f"Train: {len(train_dataset)} images")
print(f"Val: {len(val_dataset)} images")
print(f"Test: {len(test_dataset)} images")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Create model
print("\nCreating Vision Transformer (ViT) model...")

# Use Google ViT-base model pretrained on ImageNet
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Modify classifier for binary
model.classifier = nn.Linear(model.config.hidden_size, 2)
model.to(DEVICE)

# Count params
vit_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ViT trainable parameters: {vit_params:,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# Training loop
print("\nStarting ViT training...")
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(pixel_values=images)
        logits = outputs.logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Validation {epoch+1}/{EPOCHS}"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(pixel_values=images)
            logits = outputs.logits
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            _, preds = torch.max(logits, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    val_acc = accuracy_score(all_labels, all_preds)
    val_accuracies.append(val_acc)
    
    scheduler.step()
    
    print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Evaluate
print("\nEvaluating ViT on test set...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(pixel_values=images)
        logits = outputs.logits
        _, preds = torch.max(logits, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='binary')
recall = recall_score(all_labels, all_preds, average='binary')
f1 = f1_score(all_labels, all_preds, average='binary')

print("\n" + "="*50)
print("TEST RESULTS - Vision Transformer (ViT)")
print("="*50)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Save metrics
import json

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1
}

os.makedirs("outputs/metrics", exist_ok=True)

with open("outputs/metrics/vit_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Saved ViT metrics to outputs/metrics/vit_metrics.json")

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print(f"\nConfusion Matrix:")
print(f"            Predicted")
print(f"            Neg     Pos")
print(f"Actual  Neg  {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"        Pos  {cm[1,0]:4d}  {cm[1,1]:4d}")

# Save plots
os.makedirs("outputs/figures", exist_ok=True)

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='ViT Train Loss')
plt.plot(val_losses, label='ViT Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ViT Training and Validation Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='ViT Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('ViT Validation Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('outputs/figures/vit_training_curves.png', dpi=150)
print("\nSaved ViT training curves to outputs/figures/vit_training_curves.png")

# Plot confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation='nearest', cmap='Greens')
plt.title('Confusion Matrix - Vision Transformer (ViT)')
plt.colorbar()
tick_marks = np.arange(len(full_dataset.classes))
plt.xticks(tick_marks, full_dataset.classes)
plt.yticks(tick_marks, full_dataset.classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i, j] > cm.max()/2 else 'black')

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/figures/vit_confusion_matrix.png', dpi=150)
print("Saved ViT confusion matrix to outputs/figures/vit_confusion_matrix.png")

# Save model
os.makedirs("outputs/models", exist_ok=True)
torch.save(model.state_dict(), 'outputs/models/vit_fracture.pth')
print("Saved ViT model to outputs/models/vit_fracture.pth")

print("\nViT training complete!")