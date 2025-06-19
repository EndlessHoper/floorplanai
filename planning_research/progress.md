# Floor Plan AI Project Progress

## Project Overview
This project aims to build a custom AI pipeline to extract room areas and labels from floor plan images. The final goal is to process Dutch real estate listings from `funda.nl`. The core of the project involves training a custom semantic segmentation model (U-Net) and combining it with Optical Character Recognition (OCR) for a complete, end-to-end solution. This approach was chosen to provide full control over the pipeline and to build a strong portfolio piece demonstrating deep learning skills.

## Final Pipeline
```mermaid
graph TD
    subgraph "Our Custom Pipeline"
        A["Floor Plan Image"] --> B{"Scale Calibration <br> (EasyOCR + CV2)"};
        A --> C{"Our Trained U-Net Model"};
        C --> D["Semantic Mask <br> (Wall, Room)"];
        D -- "Select 'Room' Pixels" --> E{"Instance Separation <br> (Connected Components)"};
        E --> F["Individual Room Masks"];
        A --> G{"Text Label OCR <br> (EasyOCR for Dutch)"};
        F & G --> H{"Associate Label to Mask <br> (Centroid Logic)"};
        B & H --> I{"Calculate Area"};
        I --> J["Output JSON"];
    end
```

## Phase 1: Environment & Data Preparation ✅ **COMPLETE**
- [x] Set up Python 3.8+ environment with CUDA GPU support. ✅ **COMPLETED** - Python 3.12.9 with CUDA available on 1 GPU
- [x] Install initial dependencies: `torch`, `torchvision`, `opencv-python-headless`, `easyocr`, `segmentation-models-pytorch`. ✅ **COMPLETED** - All dependencies installed 
- [x] Verify CubiCasa5K dataset structure and analyze annotation format. ✅ **COMPLETED** - Dataset properly structured with train(4200)/val(400)/test(400) splits
- [x] **Implement PyTorch `Dataset` and `DataLoader` for the CubiCasa5K dataset.** ✅ **COMPLETED** - `CubiCasa5KDatasetV2` implemented with SVG-based segmentation
- [x] **The `Dataset` loads images and creates semantic masks with 8 classes from SVG files.** ✅ **COMPLETED** - Proper semantic segmentation with classes 0-7 
- [x] **Implement data augmentation pipeline (random flips, rotations, brightness/contrast jitter).** ✅ **COMPLETED** - Albumentations pipeline working correctly

**Phase 1 Final Status:** ✅ **COMPLETE** 
- Environment setup with CUDA support
- Working PyTorch Dataset with proper SVG-based semantic segmentation  
- Data augmentation pipeline functional
- Successfully loading batches with shape (batch_size, 3, 512, 512) for images and (batch_size, 512, 512) for masks
- 8 semantic classes detected: Background(0), Outdoor(1), Wall(2), Kitchen(3), Living/Dining(4), Bedroom(5), Bath(6), Entry/Hall(7)

## Phase 2: Model Training
- [ ] **NEXT TASK:** Configure a U-Net model with a ResNet34 encoder using the `segmentation-models-pytorch` library.
- [ ] Set loss function (e.g., Dice Loss or a combination of Dice and Cross-Entropy).
- [ ] Set hyperparameters: Learning Rate (e.g., 0.001), Batch Size (e.g., 4), Epochs (e.g., 30).
- [ ] Implement a training loop in PyTorch, including a validation step to monitor performance on the validation set.
- [ ] Track IoU (Intersection over Union) or Dice score for room and wall classes.
- [ ] Implement model checkpointing to save the best-performing model weights.

## Phase 3: Inference and Post-Processing Pipeline
- [ ] Build the inference script that takes a new floor plan image.
- [ ] Load the trained U-Net model and perform segmentation to get the `(wall, room)` mask.
- [ ] Isolate the room mask and apply `cv2.connectedComponents` to get individual room instances.
- [ ] **Scale Calibration:**
    - Use EasyOCR to find text labels on the image.
    - Use regex to find a dimension label (e.g., "5.2m").
    - Find the associated line with OpenCV and measure its pixel length to get a `meters/pixel` ratio.
- [ ] **Label Association:**
    - For each room instance mask, calculate its centroid.
    - Find the OCR'd text box that contains the centroid.
    - Assign that text as the room's label (e.g., "Woonkamer").
- [ ] **Area Calculation:**
    - For each instance mask, count its pixels.
    - Calculate the final area: `area_m2 = pixel_count * (scale_factor**2)`.
- [ ] Generate the final JSON output.

## Phase 4: Evaluation & Adaptation for Dutch Floor Plans
- [ ] Evaluate the full pipeline's accuracy on the CubiCasa5K test set.
- [ ] Collect a test set of at least 50 floor plans from `funda.nl`.
- [ ] Manually annotate these plans to create a ground truth for evaluation.
- [ ] Test the pipeline on the Dutch plans, specificsally checking the OCR performance on Dutch text and the model's generalization.
- [ ] If needed, fine-tune the model on a small, annotated set of Dutch floor plans to improve performance.

## Current Status: Phase 2 - Model Training Setup
**Data Preparation Complete!** ✅ 

**Dataset Implementation Summary:**
- ✅ `CubiCasa5KDatasetV2` successfully implemented using original CubiCasa5K SVG parsing
- ✅ Proper semantic segmentation masks generated from `model.svg` files
- ✅ 8 semantic classes: Background(0), Outdoor(1), Wall(2), Kitchen(3), Living/Dining(4), Bedroom(5), Bath(6), Entry/Hall(7)
- ✅ Data augmentation pipeline with random flips, rotations, brightness/contrast adjustments
- ✅ PyTorch DataLoaders working correctly with batch loading
- ✅ Image normalization using ImageNet stats for transfer learning compatibility

**Next Steps:** Begin implementing U-Net model with ResNet34 encoder for semantic segmentation training.
