#!/usr/bin/env python3
"""
U-Net Training Script for Floor Plan Room Segmentation
Trains a U-Net model on CubiCasa5K dataset for semantic segmentation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import cv2
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CubiCasaSemanticDataset(Dataset):
    """
    PyTorch Dataset for CubiCasa5K semantic segmentation
    Converts COCO instance annotations to semantic masks
    """
    
    def __init__(self, annotation_file, image_root, transforms=None):
        self.image_root = Path(image_root)
        self.transforms = transforms
        
        # Load COCO data
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create lookup dictionaries
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.annotations_by_image = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations_by_image:
                self.annotations_by_image[img_id] = []
            self.annotations_by_image[img_id].append(ann)
        
        # Filter images that have annotations
        self.image_ids = [img_id for img_id in self.images.keys() 
                         if img_id in self.annotations_by_image]
        
        print(f"Dataset loaded: {len(self.image_ids)} images with annotations")
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        
        # Load image
        img_path = self.image_root / img_info['file_name']
        if not img_path.exists():
            # Handle path format issues
            parts = img_info['file_name'].split('/')
            if len(parts) >= 3:
                img_path = self.image_root / parts[-3] / parts[-2] / parts[-1]
        
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Create semantic mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Get annotations for this image
        annotations = self.annotations_by_image.get(img_id, [])
        
        for ann in annotations:
            category_id = ann['category_id']
            bbox = ann['bbox']  # [x, y, width, height]
            
            x, y, w, h = map(int, bbox)
            x = max(0, min(x, width-1))
            y = max(0, min(y, height-1))
            w = max(1, min(w, width-x))
            h = max(1, min(h, height-y))
            
            # Fill bounding box with category ID
            # 1 = wall, 2 = room, 0 = background
            mask[y:y+h, x:x+w] = category_id
        
        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        
        return image, mask.long()

def get_transforms(is_train=True):
    """Get data augmentation transforms"""
    if is_train:
        return A.Compose([
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=5, p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(512, 512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

def dice_loss(pred, target, smooth=1):
    """Dice loss for segmentation"""
    pred = torch.softmax(pred, dim=1)
    target_one_hot = torch.zeros_like(pred)
    target_one_hot.scatter_(1, target.unsqueeze(1), 1)
    
    intersection = (pred * target_one_hot).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
    
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate pixel accuracy
            pred_masks = torch.argmax(outputs, dim=1)
            correct_pixels += (pred_masks == masks).sum().item()
            total_pixels += masks.numel()
            
            total_loss += loss.item()
            accuracy = correct_pixels / total_pixels
            pbar.set_postfix({'loss': loss.item(), 'acc': accuracy})
    
    return total_loss / len(dataloader), correct_pixels / total_pixels

def main():
    """Main training function"""
    print("U-Net Training for Floor Plan Segmentation")
    print("=" * 50)
    
    # Configuration
    config = {
        'batch_size': 4,
        'learning_rate': 0.002,
        'num_epochs': 30,
        'num_classes': 3,  # background, wall, room
        'encoder_name': 'resnet34',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Device: {config['device']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Learning rate: {config['learning_rate']}")
    
    # Create datasets
    train_dataset = CubiCasaSemanticDataset(
        annotation_file="dataset cubicasa/cubicasa5k_coco/train_coco_pt.json",
        image_root="dataset cubicasa/cubicasa5k/cubicasa5k",
        transforms=get_transforms(is_train=True)
    )
    
    val_dataset = CubiCasaSemanticDataset(
        annotation_file="dataset cubicasa/cubicasa5k_coco/val_coco_pt.json",
        image_root="dataset cubicasa/cubicasa5k/cubicasa5k",
        transforms=get_transforms(is_train=False)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create model
    model = smp.Unet(
        encoder_name=config['encoder_name'],
        encoder_weights="imagenet",
        in_channels=3,
        classes=config['num_classes']
    )
    model = model.to(config['device'])
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    # Training loop
    best_val_acc = 0
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, config['device'])
        
        # Update learning rate
        scheduler.step()
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_unet_model.pth')
            print(f"✓ New best model saved (accuracy: {best_val_acc:.4f})")
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.show()
    
    print(f"\n✅ Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved as: best_unet_model.pth")

if __name__ == "__main__":
    main() 