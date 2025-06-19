"""
CubiCasa5K Dataset for PyTorch - Floor Plan Semantic Segmentation

This module implements a PyTorch Dataset for loading CubiCasa5K floor plan images
and creating semantic segmentation masks with 3 classes:
- 0: Background
- 1: Walls  
- 2: Rooms

The dataset supports data augmentation and is designed to work with U-Net models
for semantic segmentation.
"""

import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class CubiCasa5KDataset(Dataset):
    """
    PyTorch Dataset for CubiCasa5K floor plan data.
    
    Args:
        split_file (str): Path to train.txt, val.txt, or test.txt containing image paths
        dataset_root (str): Root path to the CubiCasa5K dataset
        coco_annotation_file (str): Path to the COCO JSON annotation file
        image_size (tuple): Target image size (height, width)
        augment (bool): Whether to apply data augmentation
        
    Returns:
        dict: Contains 'image' tensor and 'mask' tensor for segmentation
    """
    
    def __init__(self, split_file, dataset_root, coco_annotation_file, 
                 image_size=(512, 512), augment=False):
        self.dataset_root = dataset_root
        self.image_size = image_size
        self.augment = augment
        
        # Load image paths from split file
        with open(split_file, 'r') as f:
            self.image_paths = [line.strip() for line in f.readlines()]
        
        # Load COCO annotations
        with open(coco_annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        # Create mapping from image filename to annotations
        self.image_id_to_annotations = {}
        for annotation in self.coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(annotation)
        
        # Create mapping from filename to image info
        self.filename_to_image_info = {}
        for image_info in self.coco_data['images']:
            self.filename_to_image_info[image_info['file_name']] = image_info
        
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
    
    def _create_semantic_mask(self, image_info, annotations):
        """
        Create semantic segmentation mask from COCO annotations.
        
        Args:
            image_info (dict): Image metadata from COCO
            annotations (list): List of annotation objects for this image
            
        Returns:
            np.ndarray: Semantic mask with shape (height, width)
        """
        height, width = image_info['height'], image_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        for annotation in annotations:
            # Get the segmentation mask from COCO annotation
            if 'segmentation' in annotation and annotation['segmentation']:
                # COCO polygons format
                segmentation = annotation['segmentation']
                if isinstance(segmentation, list):
                    # Polygon format
                    for poly in segmentation:
                        if len(poly) >= 6:  # Need at least 3 points (6 coordinates)
                            poly_points = np.array(poly).reshape(-1, 2).astype(np.int32)
                            
                            # Determine if this is a wall or room based on category
                            category_id = annotation['category_id']
                            
                            # Based on CubiCasa5K, we'll treat all annotations as rooms (class 2)
                            # Walls will be inferred later or handled separately
                            cv2.fillPoly(mask, [poly_points], 2)  # Room class
                
        return mask
    
    def _infer_walls_from_svg(self, svg_path):
        """
        Placeholder for wall inference from SVG files.
        For now, returns empty mask - can be enhanced later.
        """
        # TODO: Implement SVG parsing to extract wall information
        # This would require parsing the model.svg files in each directory
        return None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path
        rel_path = self.image_paths[idx]
        
        # Construct full image path (using F1_scaled.png)
        full_image_path = os.path.join(
            self.dataset_root, 
            rel_path.lstrip('/'), 
            'F1_scaled.png'
        )
        
        # Load image
        image = cv2.imread(full_image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {full_image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get filename for COCO lookup
        filename = f"{rel_path.strip('/')}/F1_scaled.png"
        
        # Get mask from COCO annotations
        if filename in self.filename_to_image_info:
            image_info = self.filename_to_image_info[filename]
            image_id = image_info['id']
            annotations = self.image_id_to_annotations.get(image_id, [])
            mask = self._create_semantic_mask(image_info, annotations)
        else:
            # If no annotations found, create empty mask
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            print(f"Warning: No annotations found for {filename}")
        
        # Apply transforms
        transformed = self.transform(image=image, mask=mask)
        
        return {
            'image': transformed['image'],
            'mask': transformed['mask'].long(),
            'image_path': full_image_path,
            'rel_path': rel_path
        }


def create_dataloaders(dataset_root, coco_root, batch_size=4, num_workers=2, image_size=(512, 512)):
    """
    Create train, validation, and test dataloaders for CubiCasa5K.
    
    Args:
        dataset_root (str): Path to cubicasa5k dataset
        coco_root (str): Path to cubicasa5k_coco annotations
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
    
    train_coco = os.path.join(coco_root, 'train_coco_pt.json')
    val_coco = os.path.join(coco_root, 'val_coco_pt.json')
    test_coco = os.path.join(coco_root, 'test_coco_pt.json')
    
    # Create datasets
    train_dataset = CubiCasa5KDataset(
        train_split, dataset_root, train_coco, 
        image_size=image_size, augment=True
    )
    val_dataset = CubiCasa5KDataset(
        val_split, dataset_root, val_coco,
        image_size=image_size, augment=False
    )
    test_dataset = CubiCasa5KDataset(
        test_split, dataset_root, test_coco,
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
    # Test the dataset implementation
    dataset_root = "dataset cubicasa/cubicasa5k/cubicasa5k"
    coco_root = "dataset cubicasa/cubicasa5k_coco"
    
    print("Testing CubiCasa5K Dataset...")
    
    # Create dataloaders
    try:
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_root, coco_root, batch_size=2, num_workers=0
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
        
        print("✅ Dataset test successful!")
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc() 