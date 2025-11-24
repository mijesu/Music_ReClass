# Music ReClass - RTX PC Training Guide

## System Requirements

- **GPU:** NVIDIA RTX 4060 Ti 16GB (or similar)
- **OS:** Windows 10/11 or Linux
- **Python:** 3.10 or higher
- **CUDA:** 12.1 or higher
- **RAM:** 16GB+ recommended
- **Storage:** 5GB for GTZAN dataset

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/mijesu/Music_ReClass.git
cd Music_ReClass
```

### 2. Install Dependencies

**Linux/Mac:**
```bash
bash setup_rtx.sh
```

**Windows:**
```powershell
# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other packages
pip install -r requirements_rtx.txt
```

### 3. Verify Installation
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 4. Prepare Dataset

Download GTZAN dataset and organize as:
```
GTZAN/
└── Data/
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

### 5. Update Path

Edit `training/train_gtzan_rtx.py`:
```python
GTZAN_PATH = "C:/path/to/GTZAN/Data"  # Windows
# or
GTZAN_PATH = "/path/to/GTZAN/Data"    # Linux/Mac
```

### 6. Train Model
```bash
python training/train_gtzan_rtx.py
```

## Training Configuration

### RTX 4060 Ti Optimized Settings

| Parameter | Value | Notes |
|-----------|-------|-------|
| Batch Size | 32 | Optimized for 16GB VRAM |
| Workers | 4 | Parallel data loading |
| Epochs | 50 | With early stopping |
| Learning Rate | 0.001 | AdamW optimizer |
| Mixed Precision | Enabled | FP16 for speed |
| Model Depth | 4 layers | 64→128→256→512 |

### Expected Performance

- **Training Time:** 15-20 minutes
- **Accuracy:** 70-85%
- **VRAM Usage:** 4-6 GB
- **Model Size:** ~50 MB

## Output Files

After training completes:
```
models/
├── GTZAN_best.pth           # Best model weights
└── confusion_matrix.png     # Visualization
```

## Package Versions

### Current Jetson Setup
```
torch==2.8.0
torchaudio==2.8.0
librosa==0.11.0
numpy==1.24.4
matplotlib==3.5.1
```

### Recommended RTX PC Setup
```
torch>=2.0.0
torchaudio>=2.0.0
librosa>=0.10.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.13.0
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size: `batch_size=16` or `batch_size=8`
- Reduce workers: `num_workers=2`

### Slow Training
- Check GPU usage: `nvidia-smi`
- Ensure CUDA is enabled: `torch.cuda.is_available()`
- Update GPU drivers

### Audio Loading Errors
- Install ffmpeg: `sudo apt install ffmpeg` (Linux) or download from ffmpeg.org (Windows)
- Some GTZAN files may be corrupted (automatically skipped)

## Using Trained Model

After training, use the model for classification:

```python
import torch
from train_gtzan_rtx import GTZANClassifier

# Load model
model = GTZANClassifier(num_classes=10)
model.load_state_dict(torch.load('models/GTZAN_best.pth'))
model.eval()

# Use for inference
# (See classify_music_tbc.py for full example)
```

## Contact

- GitHub: https://github.com/mijesu/Music_ReClass
- Issues: https://github.com/mijesu/Music_ReClass/issues
