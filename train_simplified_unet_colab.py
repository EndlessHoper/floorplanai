# 🚀 FloorPlan AI - Colab Optimized Training Script
# Converted from train_simplified_unet.ipynb for Google Colab
# Optimized for T4/V100/A100 GPUs with faster training

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
from datetime import datetime
import warnings
import json
warnings.filterwarnings('ignore')

# Colab-specific imports
try:
    from google.colab import drive, files
    COLAB_ENV = True
    print("🔍 Google Colab environment detected")
except ImportError:
    COLAB_ENV = False
    print("🔍 Local environment detected")

# Install dependencies if needed
if COLAB_ENV:
    try:
        import kornia.augmentation as K
    except ImportError:
        print("📦 Installing Kornia...")
        os.system('pip install -q kornia')
        import kornia.augmentation as K
    
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        print("📦 Installing segmentation-models-pytorch...")
        os.system('pip install -q segmentation-models-pytorch')
        import segmentation_models_pytorch as smp
else:
    import kornia.augmentation as K

def setup_colab_environment():
    """Setup Google Colab environment and mount drive"""
    if not COLAB_ENV:
        print("⚠️ Not in Colab environment, skipping drive mount")
        return None
    
    print("📁 Mounting Google Drive...")
    drive.mount('/content/drive')
    
    # Define paths
    drive_base = '/content/drive/MyDrive/FloorPlanAI'
    dataset_path = f'{drive_base}/dataset cubicasa/cubicasa5k/cubicasa5k'
    dataset_module = f'{drive_base}/cubicasa_dataset_v2.py'
    
    print(f"📂 Dataset path: {dataset_path}")
    print(f"🐍 Module path: {dataset_module}")
    
    # Check if files exist
    if os.path.exists(dataset_path):
        print("✅ Dataset found!")
    else:
        print("❌ Dataset not found! Please upload to Google Drive")
        return None
    
    if os.path.exists(dataset_module):
        print("✅ Dataset module found!")
    else:
        print("❌ Dataset module not found! Please upload cubicasa_dataset_v2.py")
        return None
    
    # Add to Python path
    import sys
    sys.path.append(drive_base)
    
    return {
        'drive_base': drive_base,
        'dataset_path': dataset_path,
        'dataset_module': dataset_module
    }

def detect_gpu_and_optimize():
    """Detect GPU and set optimal training parameters"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎯 GPU: {gpu_name}")
        print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
        
        # Optimize based on GPU
        if "T4" in gpu_name:
            config = {
                'batch_size': 16,
                'num_workers': 2,
                'epochs': 10
            }
            print("📊 T4 GPU detected - using optimized settings")
        elif "V100" in gpu_name:
            config = {
                'batch_size': 24,
                'num_workers': 4,
                'epochs': 8
            }
            print("📊 V100 GPU detected - using high-performance settings")
        elif "A100" in gpu_name:
            config = {
                'batch_size': 32,
                'num_workers': 4,
                'epochs': 6
            }
            print("📊 A100 GPU detected - using maximum performance settings")
        else:
            config = {
                'batch_size': 12,
                'num_workers': 2,
                'epochs': 10
            }
            print("📊 Standard GPU detected - using conservative settings")
        
        config['gpu_name'] = gpu_name
        config['gpu_memory'] = gpu_memory
        return config
    else:
        print("❌ No GPU available!")
        return {
            'batch_size': 4,
            'num_workers': 0,
            'epochs': 2,
            'gpu_name': 'CPU',
            'gpu_memory': 0
        }

class SimplifiedDataset(Dataset):
    """
    Converts CubiCasa5K's many classes into just 3:
    - 0: Background
    - 1: Wall  
    - 2: Room (all room types combined)
    """
    def __init__(self, split_file, dataset_root, image_size=512, augment=False):
        # Import here to avoid issues if module not found
        from cubicasa_dataset_v2 import CubiCasa5KDatasetV2
        
        self.original_dataset = CubiCasa5KDatasetV2(
            split_file=split_file,
            dataset_root=dataset_root, 
            image_size=(image_size, image_size),
            augment=augment
        )
        
    def __len__(self):
        return len(self.original_dataset)
    
    def __getitem__(self, idx):
        # Get original sample (returns dict)
        sample = self.original_dataset[idx]
        image = sample['image']
        mask = sample['mask']
        
        # Convert to simplified 3-class mask
        simple_mask = torch.zeros_like(mask)
        
        # Mapping: 
        # 0 = Background -> 0
        # 1 = Outdoor -> 0 (treat as background)
        # 2 = Wall -> 1  
        # 3+ = All rooms -> 2
        simple_mask[mask == 0] = 0  # Background
        simple_mask[mask == 1] = 0  # Outdoor -> Background
        simple_mask[mask == 2] = 1  # Wall
        simple_mask[mask >= 3] = 2  # All rooms -> Room
        
        return image, simple_mask

def create_model_and_optimizer(config, device):
    """Create U-Net model with optimizations"""
    print("🧠 Creating U-Net model...")
    
    # Create U-Net model
    model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet', 
        classes=3,  # Background, Wall, Room
        activation=None
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 Total parameters: {total_params:,}")
    print(f"🔧 Trainable parameters: {trainable_params:,}")
    
    # GPU-based augmentations (more aggressive for Colab)
    gpu_augmentation = nn.Sequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=10.0),  # More aggressive than local
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1)
    ).to(device)
    
    # Loss and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    print("✅ Model setup complete!")
    
    return model, gpu_augmentation, criterion, optimizer, scheduler

def load_datasets(paths, config):
    """Load and prepare datasets with Colab optimizations"""
    print("📂 Loading datasets...")
    
    dataset_root = paths['dataset_path']
    train_split = os.path.join(dataset_root, 'train.txt')
    val_split = os.path.join(dataset_root, 'val.txt')
    
    # Create datasets
    train_dataset = SimplifiedDataset(train_split, dataset_root, 512, augment=False)
    val_dataset = SimplifiedDataset(val_split, dataset_root, 512, augment=False)
    
    print(f"📊 Train: {len(train_dataset)} samples")
    print(f"📊 Val: {len(val_dataset)} samples")
    
    # Colab-optimized DataLoaders
    dataloader_kwargs = {
        'batch_size': config['batch_size'],
        'num_workers': config['num_workers'],
        'pin_memory': True,
    }
    
    # Add persistent_workers for multi-worker setups
    if config['num_workers'] > 0:
        dataloader_kwargs['persistent_workers'] = True
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    
    print(f"🚀 Train batches: {len(train_loader)}")
    print(f"🚀 Val batches: {len(val_loader)}")
    print(f"⚡ Using {config['num_workers']} workers")
    
    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                gpu_augmentation, device, config):
    """Colab-optimized training loop"""
    print(f"🚀 Starting training for {config['epochs']} epochs...")
    print(f"📊 Batch size: {config['batch_size']}")
    print(f"⚡ Workers: {config['num_workers']}")
    
    history = {'train_loss': [], 'val_loss': [], 'lr': [], 'pixel_accuracy': []}
    best_loss = float('inf')
    
    # Create checkpoint directory
    checkpoint_dir = '/content/checkpoints' if COLAB_ENV else 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(config['epochs']):
        print(f"\n{'='*60}")
        print(f"🔄 Epoch {epoch+1}/{config['epochs']}")
        print(f"📈 Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Training phase
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f'🏋️ Training', 
                           bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (images, masks) in enumerate(progress_bar):
            # Move to device
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, dtype=torch.long, non_blocking=True)
            
            # Apply GPU augmentations
            images = gpu_augmentation(images)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg': f'{train_loss/(batch_idx+1):.4f}'
            })
            
            # GPU monitoring every 50 batches
            if batch_idx % 50 == 0 and torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated()/1e9
                gpu_max = torch.cuda.max_memory_allocated()/1e9
                print(f"  💾 GPU Memory: {gpu_mem:.1f}GB / Max: {gpu_max:.1f}GB")
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc='🔍 Validation',
                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
            
            for images, masks in progress_bar:
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, dtype=torch.long, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                # Calculate pixel accuracy
                pred_masks = torch.argmax(outputs, dim=1)
                correct_pixels += (pred_masks == masks).sum().item()
                total_pixels += masks.numel()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{val_loss/(progress_bar.n+1):.4f}'
                })
        
        val_loss /= len(val_loader)
        pixel_accuracy = correct_pixels / total_pixels * 100
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        history['pixel_accuracy'].append(pixel_accuracy)
        
        # Print epoch summary
        print(f"📊 Epoch {epoch+1} Results:")
        print(f"  🏋️ Train Loss: {train_loss:.4f}")
        print(f"  🔍 Val Loss: {val_loss:.4f}")
        print(f"  🎯 Pixel Accuracy: {pixel_accuracy:.2f}%")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'pixel_accuracy': pixel_accuracy,
                'config': config
            }
            torch.save(checkpoint, f'{checkpoint_dir}/best_model.pth')
            print(f"  ✅ New best model saved! (Val Loss: {val_loss:.4f})")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
                'config': config
            }
            torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{epoch+1}.pth')
            print(f"  💾 Checkpoint saved at epoch {epoch+1}")
    
    print(f"\n🎉 Training complete!")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    
    return history, best_loss

def visualize_results(history, model, val_dataset, device, config):
    """Create visualizations of training results"""
    print("📈 Creating visualizations...")
    
    # Create results plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0,0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0,0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0,0].set_title('Training & Validation Loss')
    axes[0,0].set_xlabel('Epoch')
    axes[0,0].set_ylabel('Loss')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Learning rate
    axes[0,1].plot(epochs, history['lr'], 'g-')
    axes[0,1].set_title('Learning Rate Schedule')
    axes[0,1].set_xlabel('Epoch')
    axes[0,1].set_ylabel('Learning Rate')
    axes[0,1].grid(True)
    
    # Pixel accuracy
    axes[1,0].plot(epochs, history['pixel_accuracy'], 'm-')
    axes[1,0].set_title('Pixel Accuracy')
    axes[1,0].set_xlabel('Epoch')
    axes[1,0].set_ylabel('Accuracy (%)')
    axes[1,0].grid(True)
    
    # Sample prediction
    model.eval()
    with torch.no_grad():
        sample_img, sample_mask = val_dataset[0]
        pred = model(sample_img.unsqueeze(0).to(device))
        pred_mask = torch.argmax(pred, dim=1).cpu().squeeze()
        
        axes[1,1].imshow(pred_mask.numpy(), cmap='viridis', vmin=0, vmax=2)
        axes[1,1].set_title('Sample Prediction')
        axes[1,1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return pred_mask

def save_and_download_results(history, config, training_time):
    """Save training summary and download results"""
    if not COLAB_ENV:
        print("📁 Saving results locally...")
        return
    
    print("📦 Preparing files for download...")
    
    # Create training summary
    training_summary = {
        'config': config,
        'history': history,
        'best_val_loss': min(history['val_loss']),
        'best_pixel_accuracy': max(history['pixel_accuracy']),
        'training_time': str(training_time),
        'total_epochs': len(history['train_loss']),
        'final_lr': history['lr'][-1],
        'model_architecture': 'U-Net with ResNet34 encoder',
        'classes': ['Background', 'Wall', 'Room']
    }
    
    checkpoint_dir = '/content/checkpoints'
    with open(f'{checkpoint_dir}/training_summary.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    # List available files
    print("📁 Available files for download:")
    for file in os.listdir(checkpoint_dir):
        file_path = f'{checkpoint_dir}/{file}'
        file_size = os.path.getsize(file_path) / (1024*1024)  # MB
        print(f"  📄 {file} ({file_size:.1f} MB)")
    
    # Download main files
    print("\n🔽 Downloading results...")
    files.download(f'{checkpoint_dir}/best_model.pth')
    files.download(f'{checkpoint_dir}/training_summary.json')
    
    print("✅ Downloads complete!")

def main():
    """Main training function"""
    print("🚀 FloorPlan AI - Colab Training Starting...")
    
    # Setup Colab environment
    if COLAB_ENV:
        paths = setup_colab_environment()
        if paths is None:
            print("❌ Failed to setup Colab environment")
            return
    else:
        # Local paths
        paths = {
            'dataset_path': 'dataset cubicasa/cubicasa5k/cubicasa5k'
        }
    
    # Detect GPU and optimize
    config = detect_gpu_and_optimize()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_loader, val_loader, train_dataset, val_dataset = load_datasets(paths, config)
    
    # Create model
    model, gpu_augmentation, criterion, optimizer, scheduler = create_model_and_optimizer(config, device)
    
    # Train model
    start_time = datetime.now()
    history, best_loss = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        gpu_augmentation, device, config
    )
    end_time = datetime.now()
    training_time = end_time - start_time
    
    print(f"⏱️ Total training time: {training_time}")
    
    # Visualize results
    pred_mask = visualize_results(history, model, val_dataset, device, config)
    
    # Print final summary
    print("\n🎊 TRAINING COMPLETE!")
    print(f"🏆 Best validation loss: {best_loss:.4f}")
    print(f"🎯 Best pixel accuracy: {max(history['pixel_accuracy']):.2f}%")
    print(f"⏱️ Training time: {training_time}")
    print(f"🖥️ GPU used: {config['gpu_name']}")
    
    # Show predicted classes
    class_names = {0: 'Background', 1: 'Wall', 2: 'Room'}
    print("\n🔍 Prediction classes in sample:")
    unique_pred = torch.unique(pred_mask)
    for cls in unique_pred:
        print(f"  ✅ {class_names[cls.item()]}")
    
    # Save and download results
    save_and_download_results(history, config, training_time)
    
    print("\n🎯 Next steps:")
    print("  1. Download the trained model")
    print("  2. Use connected components for room separation")
    print("  3. Add OCR for room labels")
    print("  4. Test on Dutch floor plans")

if __name__ == "__main__":
    main() 