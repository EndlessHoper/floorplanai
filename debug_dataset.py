"""
Debug script to investigate CubiCasa5K COCO annotation format
and fix the dataset loading issues.
"""

import json
import os

def analyze_coco_data():
    """Analyze the COCO annotation structure."""
    
    print("=== ANALYZING COCO DATA ===")
    
    # Load train annotations
    with open('dataset cubicasa/cubicasa5k_coco/train_coco_pt.json', 'r') as f:
        train_data = json.load(f)
    
    print(f"Train - Images: {len(train_data['images'])}, Annotations: {len(train_data['annotations'])}")
    print(f"Categories: {train_data['categories']}")
    
    # Look at sample image and annotation
    sample_image = train_data['images'][0]
    print(f"\nSample image: {sample_image}")
    
    # Find annotations for this image
    image_id = sample_image['id']
    annotations_for_image = [ann for ann in train_data['annotations'] if ann['image_id'] == image_id]
    print(f"\nAnnotations for image {image_id}: {len(annotations_for_image)}")
    
    if annotations_for_image:
        sample_annotation = annotations_for_image[0]
        print(f"Sample annotation: {sample_annotation}")
        print(f"Annotation keys: {sample_annotation.keys()}")
    
    # Check file path patterns
    sample_files = train_data['images'][:5]
    print(f"\nSample file paths:")
    for img in sample_files:
        print(f"  {img['file_name']}")
    
    # Check our train.txt paths
    with open('dataset cubicasa/cubicasa5k/cubicasa5k/train.txt', 'r') as f:
        train_paths = [line.strip() for line in f.readlines()[:5]]
    
    print(f"\nOur train.txt paths:")
    for path in train_paths:
        print(f"  {path}")
    
    return train_data

def check_file_existence():
    """Check if files exist in our dataset."""
    
    print("\n=== CHECKING FILE EXISTENCE ===")
    
    # Check a few sample paths
    base_path = "dataset cubicasa/cubicasa5k/cubicasa5k"
    
    with open(os.path.join(base_path, 'train.txt'), 'r') as f:
        sample_paths = [line.strip() for line in f.readlines()[:3]]
    
    for rel_path in sample_paths:
        # Check both F1_original.png and F1_scaled.png
        for filename in ['F1_original.png', 'F1_scaled.png']:
            full_path = os.path.join(base_path, rel_path.lstrip('/'), filename)
            exists = os.path.exists(full_path)
            print(f"  {full_path}: {'✅' if exists else '❌'}")

def fix_filename_mapping():
    """Create corrected filename mapping for annotations."""
    
    print("\n=== ANALYZING FILENAME MAPPING ===")
    
    with open('dataset cubicasa/cubicasa5k_coco/train_coco_pt.json', 'r') as f:
        data = json.load(f)
    
    # Check annotation file patterns
    print("Original annotation file paths (first 5):")
    for i, img in enumerate(data['images'][:5]):
        original_path = img['file_name']
        print(f"  {original_path}")
        
        # Extract the directory part and create our expected path
        # Original: /kaggle/input/cubicasa5k/cubicasa5k/cubicasa5k/high_quality_architectural/6044/F1_original.png
        # Expected: high_quality_architectural/6044/F1_scaled.png
        
        parts = original_path.split('/')
        if len(parts) >= 3:
            # Get the last 3 parts: [subset]/[id]/[filename]
            subset_id_file = '/'.join(parts[-3:])
            # Replace F1_original.png with F1_scaled.png
            corrected = subset_id_file.replace('F1_original.png', 'F1_scaled.png')
            print(f"    -> {corrected}")

if __name__ == "__main__":
    train_data = analyze_coco_data()
    check_file_existence()
    fix_filename_mapping() 