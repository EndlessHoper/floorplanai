#!/usr/bin/env python3
"""
Environment Setup Test Script for Floor Plan AI Project
Tests all dependencies and verifies dataset structure
"""

import sys
import json
import os
from pathlib import Path

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")
    
    # Test PyTorch
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} installed")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    # Test other dependencies
    packages = [
        ('cv2', 'OpenCV'),
        ('easyocr', 'EasyOCR'),
        ('numpy', 'NumPy'),
        ('matplotlib', 'Matplotlib'),
        ('sklearn', 'Scikit-learn'),
        ('pycocotools', 'COCO Tools')
    ]
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
            return False
    
    # Test segmentation-models-pytorch (our new backbone)
    try:
        import segmentation_models_pytorch as smp
        print(f"✓ segmentation-models-pytorch installed")
        
        # Test basic U-Net creation
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=2  # room vs background
        )
        print(f"✓ U-Net model creation successful")
    except ImportError:
        print("✗ segmentation-models-pytorch not installed")
        return False
    except Exception as e:
        print(f"✗ U-Net model creation failed: {e}")
        return False
    
    return True

def test_dataset_structure():
    """Verify the CubiCasa5K dataset structure"""
    print("\nTesting dataset structure...")
    
    # Check main dataset directories
    base_path = Path("dataset cubicasa")
    if not base_path.exists():
        print("✗ Dataset directory not found")
        return False
    
    # Check COCO annotations
    coco_path = base_path / "cubicasa5k_coco"
    coco_files = ["train_coco_pt.json", "val_coco_pt.json", "test_coco_pt.json"]
    
    for file in coco_files:
        file_path = coco_path / file
        if file_path.exists():
            print(f"✓ Found {file} ({file_path.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"✗ Missing {file}")
            return False
    
    # Check image directories
    image_path = base_path / "cubicasa5k" / "cubicasa5k"
    image_dirs = ["high_quality_architectural", "high_quality", "colorful"]
    
    for dir_name in image_dirs:
        dir_path = image_path / dir_name
        if dir_path.exists():
            count = len(list(dir_path.iterdir()))
            print(f"✓ Found {dir_name} directory with {count} subdirectories")
        else:
            print(f"✗ Missing {dir_name} directory")
            return False
    
    # Check split files
    split_files = ["train.txt", "val.txt", "test.txt"]
    for file in split_files:
        file_path = image_path / file
        if file_path.exists():
            with open(file_path, 'r') as f:
                lines = len(f.readlines())
            print(f"✓ Found {file} with {lines} entries")
        else:
            print(f"✗ Missing {file}")
            return False
    
    return True

def analyze_coco_data():
    """Analyze the COCO annotation structure for semantic segmentation"""
    print("\nAnalyzing COCO annotations for U-Net training...")
    
    try:
        # Load training data
        with open("dataset cubicasa/cubicasa5k_coco/train_coco_pt.json", 'r') as f:
            train_data = json.load(f)
        
        print(f"Training set:")
        print(f"  - Images: {len(train_data['images'])}")
        print(f"  - Annotations: {len(train_data['annotations'])}")
        print(f"  - Categories: {len(train_data['categories'])}")
        
        for cat in train_data['categories']:
            print(f"    * {cat['name']} (ID: {cat['id']})")
        
        # Sample annotation
        if train_data['annotations']:
            sample_ann = train_data['annotations'][0]
            print(f"Sample annotation keys: {list(sample_ann.keys())}")
        
        print(f"\n✓ Note: Will convert COCO instances to semantic masks for U-Net training")
        
        # Load validation data
        with open("dataset cubicasa/cubicasa5k_coco/val_coco_pt.json", 'r') as f:
            val_data = json.load(f)
        
        print(f"\nValidation set:")
        print(f"  - Images: {len(val_data['images'])}")
        print(f"  - Annotations: {len(val_data['annotations'])}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error analyzing COCO data: {e}")
        return False

def test_sample_image():
    """Test loading a sample floor plan image"""
    print("\nTesting sample image loading...")
    
    try:
        import cv2
        
        # Try to load a sample image
        sample_path = "dataset cubicasa/cubicasa5k/cubicasa5k/high_quality_architectural/6044/F1_original.png"
        if os.path.exists(sample_path):
            img = cv2.imread(sample_path)
            if img is not None:
                print(f"✓ Successfully loaded sample image: {img.shape}")
                return True
            else:
                print("✗ Failed to load sample image")
                return False
        else:
            print("✗ Sample image not found")
            return False
            
    except Exception as e:
        print(f"✗ Error loading sample image: {e}")
        return False

def main():
    """Run all tests"""
    print("Floor Plan AI - Environment Setup Test (U-Net Edition)")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Dataset Structure", test_dataset_structure),
        ("COCO Analysis", analyze_coco_data),
        ("Sample Image", test_sample_image)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Environment is ready for U-Net training.")
        print("📋 Next: Convert COCO annotations to semantic masks for Phase 2")
    else:
        print("⚠ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main() 