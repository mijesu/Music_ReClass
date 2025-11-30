# Music ReClass - Deployment Ready

## Quick Deploy

```bash
# 1. Install
pip install -e .

# 2. Configure
cp .env.example .env

# 3. Run
python -m src.inference.classify_and_tag --input /path/to/music
```

## New Structure

```
Music_ReClass/
├── src/                    # Source code
│   ├── extractors/        # Feature extraction
│   ├── inference/         # Classification
│   ├── training/          # Model training
│   ├── models/            # Model architectures
│   ├── utils/             # Utilities
│   └── analysis/          # Analysis tools
│
├── config/                 # Configuration
│   ├── model_configs/     # Model parameters
│   └── deployment_configs/# Deployment settings
│
├── models/                 # Trained weights
├── data/features/          # Extracted features
├── scripts/setup/          # Setup scripts
├── platform/               # Platform-specific
│   ├── rtx/
│   ├── jetson/
│   ├── m1/
│   └── integrations/
│
└── docs/deployment/        # Deployment docs
```

## What Changed

### Moved
- `extractors/` → `src/extractors/`
- `training/` → `src/training/`
- `classification/` → `src/inference/`
- `json/` → `config/model_configs/`
- `features/` → `data/features/`
- `RTX/` → `platform/rtx/`
- `MBP/` → `platform/m1/`
- `Plex/` → `platform/integrations/plex/`
- `utils/` → `src/utils/`
- `analysis/` → `src/analysis/`

### Created
- `setup.py` - Package installation
- `requirements.txt` - Dependencies
- `.env.example` - Configuration template
- `docs/deployment/` - Deployment guides
- `models/` - Model storage
- `tests/` - Test directory

## Import Changes

Old:
```python
from extractors.extract_fma import extract_features
from training.train_msd import train_model
```

New:
```python
from src.extractors.extract_fma import extract_features
from src.training.train_msd import train_model
```

Or install as package:
```python
from music_reclass.extractors import extract_features
from music_reclass.training import train_model
```

## Next Steps

1. Update imports in all Python files
2. Test training pipeline
3. Test inference pipeline
4. Create Docker configuration (optional)
5. Set up CI/CD (optional)

See `docs/deployment/DEPLOYMENT_GUIDE.md` for full details.
