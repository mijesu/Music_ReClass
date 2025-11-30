# Deployment Guide

## Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/mijesu/Music_ReClass.git
cd Music_ReClass

# Install package
pip install -e .

# Or install from requirements
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Download Models

Place trained models in `models/` directory:
- `msd_model.pth` (77% accuracy, 672 KB)
- `best_model.pth` (your best trained model)

### 4. Run Classification

```bash
# Using Python module
python -m src.inference.classify_and_tag --input /path/to/music

# Or direct script
python src/inference/classify_and_tag.py --input /path/to/music
```

## Production Deployment

### Option 1: Standalone Service

```bash
# Run setup script
bash scripts/setup/startup.sh

# Start classification service
python src/inference/classify_and_tag.py --watch /music/directory
```

### Option 2: Docker (Coming Soon)

```bash
docker build -t music-reclass .
docker run -v /music:/data music-reclass
```

### Option 3: API Server (Coming Soon)

```bash
python src/api/server.py
```

## Platform-Specific Setup

### NVIDIA Jetson
```bash
# Use Jetson-specific configuration
cp platform/jetson/config.env .env
bash platform/jetson/setup.sh
```

### RTX GPU
```bash
# Use RTX-specific configuration
cp platform/rtx/config.env .env
bash platform/rtx/setup.sh
```

### Apple M1/M2
```bash
# Use M1-specific configuration
cp platform/m1/config.env .env
bash platform/m1/setup.sh
```

## Performance Tuning

### Fast Mode (77% accuracy, 2 min training)
- Use FMA features only
- Model: `msd_model.pth`
- Best for: Quick deployment

### Balanced Mode (82-88% accuracy)
- Use FMA + MERT features
- Training: 30-60 minutes
- Best for: Production use

### High Accuracy Mode (85-92% accuracy)
- Use FMA + MERT + JMLA features
- Training: 8-12 hours
- Best for: Maximum accuracy

## Monitoring

```bash
# Check logs
tail -f logs/session_*.md

# Monitor GPU
nvidia-smi -l 1
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in config
- Use smaller model
- Enable gradient checkpointing

### Slow Inference
- Enable early stopping
- Use progressive voting
- Batch process files

### Import Errors
- Reinstall package: `pip install -e .`
- Check Python path: `export PYTHONPATH=$PWD/src:$PYTHONPATH`
