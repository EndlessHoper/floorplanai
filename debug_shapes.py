"""
Debug script to investigate image and mask shape mismatches
"""

import sys
import os
import cv2
import numpy as np

# Add CubiCasa5k to the path
sys.path.append('CubiCasa5k')
from floortrans.loaders.house import House

def debug_shapes():
    """Debug image and mask shapes."""
    
    # Get a sample image path
    dataset_root = "dataset cubicasa/cubicasa5k/cubicasa5k"
    with open(os.path.join(dataset_root, 'train.txt'), 'r') as f:
        sample_path = f.readline().strip()
    
    print(f"Sample path: {sample_path}")
    
    # Construct full paths
    image_path = os.path.join(dataset_root, sample_path.lstrip('/'), 'F1_scaled.png')
    svg_path = os.path.join(dataset_root, sample_path.lstrip('/'), 'model.svg')
    
    print(f"Image path: {image_path}")
    print(f"SVG path: {svg_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    print(f"Image shape: {image.shape}")
    print(f"Image dimensions: {width}x{height}")
    
    # Create mask using CubiCasa5K House class
    try:
        house = House(svg_path, height, width)
        mask = house.get_segmentation_tensor()
        print(f"Mask shape: {mask.shape}")
        print(f"Mask dtype: {mask.dtype}")
        print(f"Mask unique values: {np.unique(mask)}")
        
        # Check if shapes match
        if image.shape[:2] == mask.shape[:2]:
            print("✅ Shapes match!")
        else:
            print("❌ Shape mismatch!")
            print(f"Image: {image.shape[:2]}, Mask: {mask.shape[:2]}")
        
    except Exception as e:
        print(f"❌ Error creating mask: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_shapes() 