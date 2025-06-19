#!/usr/bin/env python3
"""
COCO Dataset Test Script for U-Net Semantic Segmentation
Tests COCO dataset loading and analysis for semantic mask conversion
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

class COCODatasetAnalyzer:
    def __init__(self, annotation_file, image_root):
        """Initialize COCO dataset analyzer"""
        self.annotation_file = annotation_file
        self.image_root = Path(image_root)
        
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
    
    def get_stats(self):
        """Get dataset statistics"""
        stats = {
            'total_images': len(self.images),
            'total_annotations': len(self.coco_data['annotations']),
            'categories': list(self.categories.values()),
            'annotations_per_image': []
        }
        
        # Calculate annotations per image
        for img_id in self.images:
            ann_count = len(self.annotations_by_image.get(img_id, []))
            stats['annotations_per_image'].append(ann_count)
        
        stats['avg_annotations_per_image'] = np.mean(stats['annotations_per_image'])
        stats['max_annotations_per_image'] = np.max(stats['annotations_per_image'])
        stats['min_annotations_per_image'] = np.min(stats['annotations_per_image'])
        
        return stats
    
    def visualize_sample(self, image_id=None, save_path=None):
        """Visualize a sample image with annotations"""
        if image_id is None:
            # Pick a random image with annotations
            images_with_anns = [img_id for img_id in self.annotations_by_image.keys() 
                               if len(self.annotations_by_image[img_id]) > 0]
            image_id = random.choice(images_with_anns)
        
        # Get image info
        img_info = self.images[image_id]
        img_path = self.image_root / img_info['file_name']
        
        # Load image
        if not img_path.exists():
            print(f"Image not found: {img_path}")
            return None
        
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get annotations for this image
        annotations = self.annotations_by_image.get(image_id, [])
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image
        ax1.imshow(img_rgb)
        ax1.set_title(f"Original Image\n{img_info['file_name']}")
        ax1.axis('off')
        
        # Image with annotations
        ax2.imshow(img_rgb)
        
        # Draw bounding boxes
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for i, ann in enumerate(annotations):
            bbox = ann['bbox']  # [x, y, width, height]
            x, y, w, h = bbox
            
            cat_id = ann['category_id']
            cat_name = self.categories[cat_id]['name']
            
            # Draw rectangle
            rect = plt.Rectangle((x, y), w, h, 
                               linewidth=2, 
                               edgecolor=colors[cat_id % len(colors)], 
                               facecolor='none')
            ax2.add_patch(rect)
            
            # Add label
            ax2.text(x, y-10, f"{cat_name}", 
                    color=colors[cat_id % len(colors)], 
                    fontsize=8, 
                    weight='bold')
        
        ax2.set_title(f"With Annotations ({len(annotations)} objects)")
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
        
        return {
            'image_id': image_id,
            'file_name': img_info['file_name'],
            'annotations': len(annotations),
            'image_shape': img_rgb.shape
        }

def test_coco_datasets():
    """Test all COCO dataset splits"""
    print("Testing COCO Dataset Loading")
    print("=" * 50)
    
    # Dataset paths
    coco_root = "dataset cubicasa/cubicasa5k_coco"
    image_root = "dataset cubicasa/cubicasa5k/cubicasa5k"
    
    splits = ['train', 'val', 'test']
    analyzers = {}
    
    for split in splits:
        print(f"\n{split.upper()} SPLIT:")
        print("-" * 30)
        
        # Load dataset
        ann_file = f"{coco_root}/{split}_coco_pt.json"
        analyzer = COCODatasetAnalyzer(ann_file, image_root)
        analyzers[split] = analyzer
        
        # Get statistics
        stats = analyzer.get_stats()
        
        print(f"Images: {stats['total_images']}")
        print(f"Annotations: {stats['total_annotations']}")
        print(f"Categories: {len(stats['categories'])}")
        for cat in stats['categories']:
            print(f"  - {cat['name']} (ID: {cat['id']})")
        print(f"Avg annotations per image: {stats['avg_annotations_per_image']:.1f}")
        print(f"Min/Max annotations per image: {stats['min_annotations_per_image']}/{stats['max_annotations_per_image']}")
    
    return analyzers

def main():
    """Main function"""
    print("COCO Dataset Analysis for Floor Plan AI")
    print("=" * 60)
    
    # Test dataset loading
    analyzers = test_coco_datasets()
    
    # Visualize a sample from training set
    print("\n" + "=" * 60)
    print("SAMPLE VISUALIZATION")
    print("=" * 60)
    
    train_analyzer = analyzers['train']
    sample_info = train_analyzer.visualize_sample(save_path="sample_visualization.png")
    
    if sample_info:
        print(f"Visualized: {sample_info['file_name']}")
        print(f"Image shape: {sample_info['image_shape']}")
        print(f"Annotations: {sample_info['annotations']}")
    
    print("\n✅ COCO dataset analysis completed successfully!")
    print("Ready for model training setup (Phase 2)")

if __name__ == "__main__":
    main() 