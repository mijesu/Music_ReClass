# Import Updates Complete

## Changes Made

### 1. Updated Module Imports
- `from extractors.` → `from src.extractors.`
- `from training.` → `from src.training.`
- `from classification.` → `from src.inference.`
- `from utils.` → `from src.utils.`
- `from analysis.` → `from src.analysis.`
- `from config import` → `from src.utils.config import`

### 2. Removed sys.path Hacks
- Removed all `sys.path.append('..')` lines
- No longer needed with proper package structure

### 3. Fixed Double Imports
- `from src.src.` → `from src.`

## Usage

### Option 1: Run from project root
```bash
cd /home/mijesu_970/Music_ReClass
python -m src.training.train_msd
python -m src.inference.classify_and_tag --input /path/to/music
```

### Option 2: Install as package
```bash
pip install -e .
python -c "from music_reclass.extractors import extract_fma"
```

### Option 3: Add to PYTHONPATH
```bash
export PYTHONPATH=/home/mijesu_970/Music_ReClass:$PYTHONPATH
python src/training/train_msd.py
```

## Testing

```bash
# Test imports
python3 -c "import sys; sys.path.insert(0, '.'); from src.utils.config import *"
python3 -c "import sys; sys.path.insert(0, '.'); from src.extractors.extract_fma import extract_fma_features"
```

## Files Updated
- All Python files in `src/`
- All Python files in `platform/`
