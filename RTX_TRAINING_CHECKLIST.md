# RTX PC Training Checklist

## 📋 Required Materials

### 1. Hardware Requirements
- [ ] NVIDIA RTX 4060 Ti 16GB (or similar GPU)
- [ ] 16GB+ RAM
- [ ] 50GB+ free disk space
- [ ] Internet connection (for initial setup)

### 2. Software Requirements
- [ ] Windows 10/11 or Linux (Ubuntu 20.04+)
- [ ] Python 3.10 or higher
- [ ] NVIDIA GPU drivers (latest)
- [ ] CUDA 12.1 or higher

### 3. Code & Scripts
- [ ] Clone GitHub repository: `git clone https://github.com/mijesu/Music_ReClass.git`
- [ ] Or download ZIP from GitHub

### 4. Datasets (Choose One or Both)

#### Option A: GTZAN Dataset (Required for basic training)
- [ ] Download GTZAN: http://marsyas.info/downloads/datasets.html
- [ ] Size: ~1.2GB
- [ ] Extract to: `C:/Datasets/GTZAN/Data/` (Windows) or `/data/GTZAN/Data/` (Linux)
- [ ] Structure:
  ```
  GTZAN/Data/
  ├── blues/
  ├── classical/
  ├── country/
  ├── disco/
  ├── hiphop/
  ├── jazz/
  ├── metal/
  ├── pop/
  ├── reggae/
  └── rock/
  ```

#### Option B: FMA Medium Dataset (Optional, for better accuracy)
- [ ] Download FMA Medium: https://github.com/mdeff/fma
- [ ] Size: ~25GB
- [ ] Extract to: `C:/Datasets/FMA/Data/fma_medium/` (Windows)
- [ ] Note: Takes 2-3 hours to download

### 5. Pre-trained Models (Optional)
- [ ] Copy from Jetson if available:
  - OpenJMLA models (optional)
  - MSD features (optional)

---

## 🚀 Setup Steps

### Step 1: Install Python Packages
```bash
# Navigate to project
cd Music_ReClass

# Run setup script (Linux/Mac)
bash setup_rtx.sh

# Or manually install (Windows/Linux)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements_rtx.txt
```

### Step 2: Verify Installation
```python
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```
Expected output:
```
PyTorch: 2.x.x
CUDA: True
```

### Step 3: Update Dataset Paths
Edit the training script you want to use:

**For GTZAN:**
```python
# In training/train_gtzan_rtx.py or train_gtzan_enhanced.py
GTZAN_PATH = "C:/Datasets/GTZAN/Data"  # Windows
# or
GTZAN_PATH = "/data/GTZAN/Data"        # Linux
```

**For FMA:**
```python
# In training/train_combined_4hr.py
FMA_PATH = "C:/Datasets/FMA/Data/fma_medium"  # Windows
```

### Step 4: Choose Training Script

| Script | Time | Accuracy | Dataset | Best For |
|--------|------|----------|---------|----------|
| `train_gtzan_rtx.py` | 15-20 min | 70-85% | GTZAN | Quick test |
| `train_gtzan_enhanced.py` | 4 hours | 80-90% | GTZAN | Best single model |
| `train_combined_4hr.py` | 4 hours | 75-85% | FMA+MSD | Multi-modal |

### Step 5: Start Training
```bash
cd Music_ReClass
python training/train_gtzan_rtx.py
```

---

## 📦 Files to Transfer from Jetson (Optional)

If you want to use existing data from Jetson:

### Essential (Code only)
```
Music_ReClass/
├── training/          # All training scripts
├── utils/            # Early stopping, logger
├── requirements_rtx.txt
└── setup_rtx.sh
```

### Optional (Datasets - if already downloaded)
```
DataSets/
├── GTZAN/Data/       # 1.2GB
└── FMA/Data/         # 25GB
```

### Optional (Pre-trained models)
```
AI_models/
├── OpenJMLA/         # 1.6GB (optional)
└── MSD/Data/         # 2.6GB (optional, for combined training)
```

---

## 📊 Expected Output Files

After training completes, you'll have:

```
Music_ReClass/
├── models/
│   ├── GTZAN_best.pth              # Trained model (~50MB)
│   ├── confusion_matrix.png        # Visualization
│   └── checkpoint_*.pth            # Checkpoints
│
└── logs/
    ├── gtzan_rtx_config.json       # Training config
    ├── gtzan_rtx_metrics.csv       # Epoch metrics
    ├── gtzan_rtx.log               # Full log
    └── gtzan_rtx_summary.json      # Final results
```

---

## 🔧 Troubleshooting

### CUDA Not Available
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Out of Memory
Edit training script:
```python
batch_size=16  # Reduce from 32 to 16
# or
batch_size=8   # Further reduce if needed
```

### Slow Data Loading
Edit training script:
```python
num_workers=2  # Reduce from 4 to 2
```

### Audio Loading Errors
Install ffmpeg:
- **Windows:** Download from https://ffmpeg.org/download.html
- **Linux:** `sudo apt install ffmpeg`

---

## ✅ Quick Start Checklist

Minimum requirements to start training:

- [ ] RTX GPU with CUDA installed
- [ ] Python 3.10+ installed
- [ ] Git clone: `git clone https://github.com/mijesu/Music_ReClass.git`
- [ ] Install packages: `pip install -r requirements_rtx.txt`
- [ ] Download GTZAN dataset (~1.2GB)
- [ ] Update `GTZAN_PATH` in training script
- [ ] Run: `python training/train_gtzan_rtx.py`

**Estimated time to start training: 30 minutes**

---

## 📞 Support

- GitHub Issues: https://github.com/mijesu/Music_ReClass/issues
- Documentation: See README_RTX.md

---

## 🎯 Training Time Estimates

| Dataset | Script | Time | Accuracy |
|---------|--------|------|----------|
| GTZAN (1K) | train_gtzan_rtx.py | 15-20 min | 70-85% |
| GTZAN (1K) | train_gtzan_enhanced.py | 4 hours | 80-90% |
| FMA (25K) | train_combined_4hr.py | 4 hours | 75-85% |
| FMA (25K) | 48-hour training | 48 hours | 85-90% |

---

**Last Updated:** 2025-11-24
**Version:** 1.0
