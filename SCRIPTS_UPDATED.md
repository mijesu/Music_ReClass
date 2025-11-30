# Scripts Updated ✅

## Changes Made

### startup.sh
- ✅ Fixed project directory navigation (now goes to root)
- ✅ Updated feature files path: `features/` → `data/features/`
- ✅ Updated all commands to use new structure:
  - `python3 training/train_msd.py` → `python3 -m src.training.train_msd`
  - `python3 classification/classify_and_tag.py` → `python3 -m src.inference.classify_and_tag`
  - `python3 extractors/extract_fma.py` → `python3 -m src.extractors.extract_fma`

### shutdown.sh
- ✅ Fixed project directory navigation (now goes to root)

## How to Use

```bash
# From anywhere in the project
bash scripts/setup/startup.sh

# Or make executable and run
chmod +x scripts/setup/startup.sh
./scripts/setup/startup.sh
```

## What They Do

**startup.sh** - Shows you:
- Git status
- GPU status
- Python environment
- Feature files in `data/features/`
- Quick commands with correct paths

**shutdown.sh** - Helps you:
- Sync documentation
- Check uncommitted changes
- Create session logs
- Clean up before ending work

Both scripts now work correctly with the new directory structure!
