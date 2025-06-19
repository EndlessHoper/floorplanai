"""
Debug script to understand how to create proper semantic segmentation mask
"""

import sys
import os
import cv2
import numpy as np

# Add CubiCasa5k to the path
sys.path.append('CubiCasa5k')
from floortrans.loaders.house import House

def debug_segmentation():
    """Debug segmentation tensor creation."""
    
    # Get a sample image path
    dataset_root = "dataset cubicasa/cubicasa5k/cubicasa5k"
    with open(os.path.join(dataset_root, 'train.txt'), 'r') as f:
        sample_path = f.readline().strip()
    
    print(f"Sample path: {sample_path}")
    
    # Construct full paths
    image_path = os.path.join(dataset_root, sample_path.lstrip('/'), 'F1_scaled.png')
    svg_path = os.path.join(dataset_root, sample_path.lstrip('/'), 'model.svg')
    
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    print(f"Image dimensions: {width}x{height}")
    
    # Create house object
    house = House(svg_path, height, width)
    
    # Examine individual components
    print(f"Walls shape: {house.walls.shape}")
    print(f"Icons shape: {house.icons.shape}")
    print(f"Walls unique values: {np.unique(house.walls)}")
    print(f"Icons unique values: {np.unique(house.icons)}")
    
    # Get the full segmentation tensor (2-channel)
    seg_tensor = house.get_segmentation_tensor()
    print(f"Segmentation tensor shape: {seg_tensor.shape}")
    
    # Extract individual channels
    walls_channel = seg_tensor[0]  # First channel (walls)
    icons_channel = seg_tensor[1]  # Second channel (icons/rooms)
    
    print(f"Walls channel unique values: {np.unique(walls_channel)}")
    print(f"Icons channel unique values: {np.unique(icons_channel)}")
    
    # Create a single semantic mask by combining both channels
    # Priority: icons (rooms) > walls > background
    semantic_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Set walls (where walls_channel > 0)
    semantic_mask[walls_channel > 0] = 2  # Wall class
    
    # Set rooms (where icons_channel > 0), overriding walls
    semantic_mask[icons_channel > 0] = icons_channel[icons_channel > 0]
    
    print(f"Final semantic mask shape: {semantic_mask.shape}")
    print(f"Final semantic mask unique values: {np.unique(semantic_mask)}")
    
    # Count pixels per class
    unique_values, counts = np.unique(semantic_mask, return_counts=True)
    for val, count in zip(unique_values, counts):
        percentage = (count / semantic_mask.size) * 100
        print(f"  Class {val}: {count} pixels ({percentage:.1f}%)")

if __name__ == "__main__":
    debug_segmentation() 