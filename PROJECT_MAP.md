# Music ReClass - Project Map

## 📍 Where Everything Is Now

### 🎯 Main Tasks

| What You Want To Do | Where To Go | Command |
|---------------------|-------------|---------|
| **Train a model** | `src/training/` | `python -m src.training.train_msd` |
| **Classify music** | `src/inference/` | `python -m src.inference.classify_and_tag --input /path` |
| **Extract features** | `src/extractors/` | `python -m src.extractors.extract_fma` |
| **Check model** | `src/analysis/` | `python -m src.analysis.check_model` |

### 📂 Directory Guide

```
Music_ReClass/
│
├── 🎵 src/                          # ALL YOUR CODE IS HERE
│   ├── extractors/                  # Feature extraction (FMA, MERT, JMLA)
│   ├── training/                    # Train models
│   ├── inference/                   # Classify music (was "classification/")
│   ├── models/                      # Model architectures
│   ├── utils/                       # Helper functions
│   └── analysis/                    # Analysis tools
│
├── ⚙️  config/                       # Settings
│   └── model_configs/               # Model parameters (was "json/")
│
├── 💾 models/                        # Trained model files (.pth)
│   └── (put your .pth files here)
│
├── 📊 data/                          # Data storage
│   └── features/                    # Feature files (.npy)
│
├── 🖥️  platform/                     # Platform-specific
│   ├── rtx/                         # RTX GPU code
│   ├── m1/                          # M1 Mac code (was "MBP/")
│   └── integrations/plex/           # Plex integration
│
├── 🔧 scripts/                       # Scripts
│   └── setup/                       # startup.sh, shutdown.sh
│
├── 📝 docs/                          # Documentation
│   ├── deployment/                  # How to deploy
│   └── ...
│
└── 📋 logs/                          # Log files
```

## 🚀 Quick Commands

### Training
```bash
# Fast training (2 min, 77%)
python -m src.training.train_msd

# Best accuracy (4 hrs, 80-90%)
python -m src.training.train_gtzan_enhanced

# FMA training
python -m src.training.train_fma_progressive
```

### Classification
```bash
# Classify music files
python -m src.inference.classify_and_tag --input /path/to/music

# With specific model
python -m src.inference.classify_and_tag --input /path --model models/msd_model.pth
```

### Feature Extraction
```bash
# Extract FMA features
python -m src.extractors.extract_fma_features

# Extract MERT features
python -m src.extractors.extract_mert_features

# Extract JMLA features
python -m src.extractors.extract_jmla_features
```

## 🔍 Finding Your Old Files

| Old Location | New Location |
|--------------|--------------|
| `extractors/extract_fma.py` | `src/extractors/extract_fma.py` |
| `training/train_msd.py` | `src/training/train_msd.py` |
| `classification/classify_and_tag.py` | `src/inference/classify_and_tag.py` |
| `json/fma_parameters.json` | `config/model_configs/fma_parameters.json` |
| `features/FMA_features.npy` | `data/features/FMA_features.npy` |
| `RTX/train_gtzan_rtx.py` | `platform/rtx/train_gtzan_rtx.py` |
| `MBP/` | `platform/m1/` |
| `Plex/` | `platform/integrations/plex/` |
| `utils/config.py` | `src/utils/config.py` |
| `analysis/check_model.py` | `src/analysis/check_model.py` |

## 💡 Key Changes

1. **Everything is in `src/` now** - All your Python code
2. **`classification/` → `src/inference/`** - Better name for production
3. **`json/` → `config/model_configs/`** - Clearer purpose
4. **`features/` → `data/features/`** - Organized data storage
5. **Platform code separated** - RTX, M1, Jetson in `platform/`

## 📖 Documentation

- `README.md` - Project overview
- `DEPLOYMENT.md` - How to deploy
- `IMPORT_UPDATES.md` - What changed in imports
- `docs/deployment/DEPLOYMENT_GUIDE.md` - Full deployment guide

## ❓ Need Help?

- See what's in a folder: `ls src/training/`
- Find a file: `find . -name "train_msd.py"`
- Read this guide: `cat PROJECT_MAP.md`
