# Floor Plan AI Project

A Mask R-CNN based system for extracting room areas from floor plan images, designed for Dutch real estate analysis.

## Project Overview

This project builds an AI system that can:
1. Take a floor plan image as input
2. Detect and segment individual rooms using U-Net semantic segmentation
3. Extract individual room instances using connected components analysis
4. Extract scale information from dimension labels using OCR
5. Calculate accurate room areas in square meters
6. Output structured JSON data with room names and areas

**Target use case**: Process floor plan images from Dutch real estate listings (funda.nl) to automatically extract room dimensions.

## Phase 1 ✅ - Environment Setup & Dataset Preparation

### Environment
- **Python**: 3.12.9
- **CUDA**: 12.9 with NVIDIA GeForce RTX 2060 SUPER
- **PyTorch**: 2.7.1+cu118 with CUDA support

### Dependencies Installed
- ✅ PyTorch 2.7.1+cu118 
- ✅ OpenCV 4.11.0
- ✅ EasyOCR 1.7.2
- ✅ segmentation-models-pytorch 0.5.0
- ✅ NumPy, Matplotlib, Scikit-learn
- ✅ pycocotools for COCO dataset handling

### Dataset Analysis (CubiCasa5K)
- **Training set**: 4,200 images, 173,023 annotations
- **Validation set**: 400 images, 15,926 annotations  
- **Test set**: 400 images, 16,818 annotations
- **Categories**: 2 types (wall, room)
- **Avg annotations per image**: ~41.2

### Files Created
- `setup_test.py` - Environment verification script
- `coco_dataset_test.py` - Dataset analysis and visualization
- `requirements.txt` - Python dependencies

## Quick Start

### 1. Environment Test
```bash
python setup_test.py
```

### 2. Dataset Analysis  
```bash
python coco_dataset_test.py
```

## Next Steps - Phase 2

1. ✅ Architecture pivot to U-Net + connected components (no Detectron2 needed!)
2. Implement data augmentation pipeline for semantic segmentation
3. Set up PyTorch Dataset for semantic masks (not COCO instances)
4. Begin U-Net model architecture setup

## Dataset Structure

```
dataset cubicasa/
├── cubicasa5k_coco/
│   ├── train_coco_pt.json (46MB)
│   ├── val_coco_pt.json (4.2MB)
│   └── test_coco_pt.json (4.5MB)
└── cubicasa5k/cubicasa5k/
    ├── train.txt (4200 entries)
    ├── val.txt (400 entries)
    ├── test.txt (400 entries)
    ├── high_quality_architectural/ (3732 subdirs)
    ├── high_quality/ (992 subdirs)
    └── colorful/ (276 subdirs)
```

## Architecture Overview

```
Input: Floor plan image → 
U-Net (semantic segmentation) → 
Connected Components (instance separation) → 
EasyOCR (scale detection) → 
Area calculation → 
JSON output
```

**ELI5**: U-Net "paints" room pixels, then we group connected blobs to separate individual rooms.

## Project Phases

- ✅ **Phase 1**: Environment Setup & Dataset Preparation
- 🔄 **Phase 2**: Data Analysis & Preprocessing
- ⏳ **Phase 3**: Model Architecture & Training  
- ⏳ **Phase 4**: Inference Pipeline Development
- ⏳ **Phase 5**: Scale Calibration System
- ⏳ **Phase 6**: Room Association & Area Calculation
- ⏳ **Phase 7**: Integration & Output Generation
- ⏳ **Phase 8**: Evaluation & Testing
- ⏳ **Phase 9**: Dutch Real Estate Adaptation
- ⏳ **Phase 10**: Deployment & Documentation

---
*Last updated: Phase 1 Complete* 