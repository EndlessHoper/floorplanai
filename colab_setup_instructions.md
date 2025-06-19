# 🚀 Google Colab Setup Instructions for FloorPlan AI

## 📁 What to Upload to Google Drive

Create this folder structure in your Google Drive:

```
MyDrive/
└── FloorPlanAI/
    ├── cubicasa_dataset_v2.py          # Your dataset module
    ├── train_simplified_unet_colab.py  # Colab training script (see below)
    └── dataset cubicasa/               # Your dataset folder
        └── cubicasa5k/
            └── cubicasa5k/
                ├── train.txt
                ├── val.txt
                ├── test.txt
                ├── colorful/
                ├── high_quality/
                └── high_quality_architectural/
```

## 📋 Files You Need to Upload:

### 1. **cubicasa_dataset_v2.py** ✅ (You already have this)
- Upload your existing `cubicasa_dataset_v2.py` file

### 2. **Dataset** ✅ (You already have this)
- Upload your entire `dataset cubicasa/` folder
- **Size**: ~2-3GB (this is the main upload)
- **Time**: 10-15 minutes depending on internet speed

### 3. **Colab Training Script** (I'll create this for you)

## 🎯 Google Colab Optimizations vs Your Local Setup:

| Feature | Your Local (Windows) | Google Colab |
|---------|---------------------|--------------|
| **GPU** | RTX 2060 Super (8GB) | T4 (16GB) / V100 (16GB) / A100 (40GB) |
| **Batch Size** | 8 | 16-32 (2-4x larger) |
| **Workers** | 0 (Windows limit) | 2-4 (Linux, no issues) |
| **Speed per Epoch** | ~35 minutes | ~8-12 minutes (3-4x faster) |
| **Total Training (10 epochs)** | ~6 hours | ~1.5 hours |
| **Memory** | 4.2GB max | Much better utilization |

## 🔧 Colab-Specific Optimizations:

### 1. **Dynamic Batch Size Selection**
```python
# Auto-detects GPU and sets optimal batch size
if "T4" in gpu_name:
    batch_size = 16
elif "V100" in gpu_name or "A100" in gpu_name:
    batch_size = 32
```

### 2. **Multiple Workers**
```python
# Linux can handle multiple workers
num_workers = 2  # vs 0 on Windows
persistent_workers = True  # Keeps workers alive
```

### 3. **Better GPU Utilization**
```python
# More aggressive augmentations since GPU can handle it
RandomRotation(degrees=10.0)  # vs 5.0 locally
ColorJitter(brightness=0.2)   # vs 0.1 locally
```

### 4. **Enhanced Monitoring**
- Learning rate scheduling
- Pixel accuracy tracking
- Better progress bars
- Automatic checkpointing

## ⚡ Speed Comparison:

**Your Current Setup (1 epoch test):**
- 525 batches × 4.2 seconds = ~37 minutes per epoch
- 10 epochs = ~6.2 hours

**Google Colab T4 (estimated):**
- 131 batches × 3.5 seconds = ~8 minutes per epoch
- 10 epochs = ~1.3 hours

**Speed Improvement: 4.8x faster!** 🚀

## 🎁 Additional Colab Benefits:

1. **No Windows Issues**: No num_workers=0 limitations
2. **Better Hardware**: More VRAM, faster GPUs
3. **Pre-installed Libraries**: PyTorch, CUDA already setup
4. **Automatic Saves**: Downloads trained model automatically
5. **Free Tier**: 12 hours/day free GPU time
6. **Pro Tier**: $10/month for priority access to better GPUs

## 📦 What the Colab Notebook Will Do:

1. **Auto-detect GPU** and optimize settings
2. **Mount Google Drive** to access your dataset
3. **Install dependencies** automatically
4. **Train model** with optimal settings
5. **Show real-time progress** with better monitoring
6. **Save checkpoints** automatically
7. **Download trained model** when complete
8. **Generate training summary** with all metrics

## 🚀 Expected Results:

With Colab's optimizations, your model should:
- **Train 4-5x faster** than your local setup
- **Use more GPU memory** (better utilization)
- **Handle larger batch sizes** (better gradients)
- **Complete 10 epochs** in ~1-2 hours instead of 6+ hours

Ready to create the Colab notebook? The main upload will be your dataset (~2-3GB), everything else is small. 