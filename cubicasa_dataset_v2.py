"""
CubiCasa5K Dataset for PyTorch - Floor Plan Semantic Segmentation (v2)

This improved version uses the original CubiCasa5K SVG parsing approach
to generate proper semantic segmentation masks from the model.svg files.

Classes:
- 0: Background
- 1: Outdoor  
- 2: Wall
- 3: Kitchen
- 4: Living/Dining/Lounge  
- 5: Bedroom
- 6: Bath/Sauna
- 7: Entry/Hall
- 8: Railing
- 9: Storage/Closet
- 10: Garage/CarPort
- 11: Other Rooms
"""

import os
import sys
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add CubiCasa5k to the path to use their classes
sys.path.append('CubiCasa5k')
from floortrans.loaders.house import House


class CubiCasa5KDatasetV2(Dataset):
    """
    Improved PyTorch Dataset for CubiCasa5K using SVG-based segmentation.
    
    Args:
        split_file (str): Path to train.txt, val.txt, or test.txt containing image paths
        dataset_root (str): Root path to the CubiCasa5K dataset  
        image_size (tuple): Target image size (height, width)
        augment (bool): Whether to apply data augmentation
        use_original_image (bool): Use F1_original.png instead of F1_scaled.png
        
    Returns:
        dict: Contains 'image' tensor and 'mask' tensor for segmentation
    """
    
    def __init__(self, split_file, dataset_root, image_size=(512, 512), 
                 augment=False, use_original_image=False):
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.augment = augment
        self.use_original_image = use_original_image
        
        # Load image paths from split file
        with open(split_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # Image filename to use
        self.image_filename = 'F1_original.png' if use_original_image else 'F1_scaled.png'
        
        # Setup augmentations
        self.transform = self._get_transforms()
        
    def _get_transforms(self):
        """Get augmentation transforms based on augment setting."""
        if self.augment:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2), 
                A.Rotate(limit=5, p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.3
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            return A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
    
    def _create_segmentation_mask(self, svg_path, height, width):
        """
        Create semantic segmentation mask from SVG file using CubiCasa5K's approach.
        
        Args:
            svg_path (str): Path to the model.svg file
            height (int): Image height
            width (int): Image width
            
        Returns:
            np.ndarray: Semantic mask with shape (height, width)
        """
        try:
            # Use CubiCasa5K's House class to parse SVG
            house = House(svg_path, height, width)
            
            # Get the individual components
            walls = house.walls  # 2D array (height, width)
            icons = house.icons  # 2D array (height, width) - rooms
            
            # Create a single semantic mask by combining walls and rooms
            # Priority: rooms (icons) > walls > background
            semantic_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Set walls first (where walls > 0 but we'll map wall values to class 2)
            wall_mask = walls > 0
            semantic_mask[wall_mask] = 2  # Wall class = 2
            
            # Set rooms (where icons > 0), which will override walls if they overlap
            room_mask = icons > 0
            semantic_mask[room_mask] = icons[room_mask]
            
            return semantic_mask
            
        except Exception as e:
            print(f"Warning: Failed to create segmentation mask from {svg_path}: {e}")
            # Return empty mask (all background) if SVG parsing fails
            return np.zeros((height, width), dtype=np.uint8)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path
        rel_path = self.image_paths[idx]
        
        # Construct full paths
        full_image_path = os.path.join(
            self.dataset_root, 
            rel_path.lstrip('/'), 
            self.image_filename
        )
        
        svg_path = os.path.join(
            self.dataset_root,
            rel_path.lstrip('/'),
            'model.svg'
        )
        
        # Load image
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {full_image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Create segmentation mask from SVG
        mask = self._create_segmentation_mask(svg_path, height, width)
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        
        return {
            'image': transformed['image'],
            'mask': transformed['mask'].long(),
            'image_path': full_image_path,
            'svg_path': svg_path,
            'rel_path': rel_path
        }


def create_dataloaders_v2(dataset_root, batch_size=4, num_workers=2, image_size=(512, 512)):
    """
    Create train, validation, and test dataloaders for CubiCasa5K using SVG-based segmentation.
    
    Args:
        dataset_root (str): Path to cubicasa5k dataset
        batch_size (int): Batch size for dataloaders
        num_workers (int): Number of worker processes
        image_size (tuple): Target image size
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Define paths
    train_split = os.path.join(dataset_root, 'train.txt')
    val_split = os.path.join(dataset_root, 'val.txt')
    test_split = os.path.join(dataset_root, 'test.txt')
    
    # Create datasets
    train_dataset = CubiCasa5KDatasetV2(
        train_split, dataset_root, 
        image_size=image_size, augment=True
    )
    val_dataset = CubiCasa5KDatasetV2(
        val_split, dataset_root,
        image_size=image_size, augment=False
    )
    test_dataset = CubiCasa5KDatasetV2(
        test_split, dataset_root,
        image_size=image_size, augment=False  
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the improved dataset implementation
    dataset_root = "dataset cubicasa/cubicasa5k/cubicasa5k"
    
    print("Testing CubiCasa5K Dataset V2 with SVG-based segmentation...")
    
    # Create dataloaders
    try:
        train_loader, val_loader, test_loader = create_dataloaders_v2(
            dataset_root, batch_size=2, num_workers=0
        )
        
        print(f"Train dataset size: {len(train_loader.dataset)}")
        print(f"Val dataset size: {len(val_loader.dataset)}")
        print(f"Test dataset size: {len(test_loader.dataset)}")
        
        # Test loading a single batch
        print("\nTesting batch loading...")
        batch = next(iter(train_loader))
        print(f"Image batch shape: {batch['image'].shape}")
        print(f"Mask batch shape: {batch['mask'].shape}")
        print(f"Image dtype: {batch['image'].dtype}")
        print(f"Mask dtype: {batch['mask'].dtype}")
        print(f"Mask unique values: {torch.unique(batch['mask'])}")
        print(f"Mask value counts:")
        unique_values, counts = torch.unique(batch['mask'], return_counts=True)
        for val, count in zip(unique_values, counts):
            print(f"  Class {val}: {count} pixels")
        
        print("✅ Dataset V2 test successful!")
        
    except Exception as e:
        print(f"❌ Dataset V2 test failed: {e}")
        import traceback
        traceback.print_exc() 