#!/bin/bash
# Setup script for RTX 4060 Ti PC

echo "==================================="
echo "Music ReClass - RTX PC Setup"
echo "==================================="

# Check Python version
echo -e "\nChecking Python version..."
python3 --version

# Check if CUDA is available
echo -e "\nChecking CUDA..."
nvidia-smi

# Install PyTorch with CUDA support
echo -e "\nInstalling PyTorch with CUDA 12.1..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo -e "\nInstalling other packages..."
pip3 install -r requirements_rtx.txt

# Verify installation
echo -e "\nVerifying installation..."
python3 << EOF
import torch
import librosa
import numpy as np
import sklearn
import matplotlib
import seaborn

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Librosa: {librosa.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Seaborn: {seaborn.__version__}")
EOF

echo -e "\n==================================="
echo "Setup complete!"
echo "==================================="
echo -e "\nNext steps:"
echo "1. Update GTZAN_PATH in training/train_gtzan_rtx.py"
echo "2. Run: python3 training/train_gtzan_rtx.py"
