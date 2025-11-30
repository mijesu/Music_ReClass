# Music Reclassification Project - History

## Session 1: November 22, 2025 (Initial Setup)

### Objectives
- Set up music AI project for genre classification
- Identify and organize datasets
- Configure AI models

### Completed Tasks
1. **Created project structure**
   - Identified GTZAN dataset location
   - Set up dataset organization (Data/ and Misc/ folders)

2. **Initial scripts created**
   - `pytorch_example.py` - PyTorch basics

## Session 3: November 23, 2025 (16:55 - 17:55)

### Major Accomplishments

#### 1. Script Organization
- Created organized directory structure at `/media/mijesu_970/SSD_Data/Python/Music_Reclass/`
- Categorized 14 Python scripts into 4 folders:
  - `training/` - 6 training scripts
  - `analysis/` - 4 analysis scripts  
  - `utils/` - 1 utility script
  - `examples/` - 3 example scripts
- Separated executable scripts from project documentation

#### 2. New Training Scripts Created
- **train_gtzan_openjmla.py** - Transfer learning with GPU monitoring and adaptive batch sizing
- **train_xgboost_fma.py** - XGBoost baseline using FMA pre-computed features
- **quick_baseline.py** - Fast 2-5 minute baseline for rapid experimentation
- **compare_models.py** - Automated comparison between XGBoost and deep learning
- **train_gtzan_v2.py** ⭐ - Enhanced training with all improvements (RECOMMENDED)

#### 3. Analysis & Utility Scripts
- **analyze_data.py** - FMA/GTZAN genre distribution, mel-spectrogram visualization, class imbalance check
- **gpu_monitor.py** - GPU memory monitoring and batch size suggestions

#### 4. Key Features Implemented

**GPU Memory Management:**
- Auto-detect free GPU memory
- Dynamic batch size adjustment (2-8 based on available memory)
- Memory cleanup every 20 batches
- Real-time GPU usage display

**Training Infrastructure (V2):**
- OpenJMLA transfer learning (86M frozen + 200K trainable params)
- Data augmentation (time stretch, pitch shift, noise)
- Validation loop with metrics
- Confusion matrix visualization
- Classification report (precision, recall, F1)
- Learning rate scheduler (ReduceLROnPlateau)
- Early stopping (patience=5)
- Best model saving
- Checkpoint system for resume

#### 5. Comparative Analysis
- Created **Kaggle_vs_JMLA.md.md** documenting:
  - Kaggle notebook (XGBoost) vs custom script (CNN)
  - Strengths and weaknesses of each approach
  - 10 specific improvement suggestions
  - Expected accuracy ranges for different methods
  - Recommended 3-phase workflow

#### 6. Package Installations
- ✓ xgboost 3.1.2 - For traditional ML baseline
- ✗ jukebox - Abandoned due to Python 2 compatibility issues

### Technical Decisions

**Jukebox Installation:**
- Attempted installation failed (old dependencies: Django 1.4.5, mutagen 1.21)
- Decision: Abandoned as it's for music generation, not classification
- Alternative: Focus on OpenJMLA for audio understanding

**Script Organization:**
- Moved from `/Kiro_Projects/Music_Reclass/scripts/` to `/Python/Music_Reclass/`
- Rationale: Separate executable code from project documentation

### Project Structure (Current)

```
/media/mijesu_970/SSD_Data/
├── Python/Music_Reclass/          # Executable scripts
│   ├── training/
│   │   ├── train_gtzan_v2.py      ⭐ RECOMMENDED
│   │   ├── train_gtzan_openjmla.py
│   │   ├── train_xgboost_fma.py
│   │   ├── quick_baseline.py
│   │   ├── compare_models.py
│   │   └── [2 legacy scripts]
│   ├── analysis/
│   │   ├── analyze_data.py
│   │   └── [3 more]
│   ├── utils/
│   │   └── gpu_monitor.py
│   └── examples/ [3 scripts]
│
├── Kiro_Projects/Music_Reclass/   # Documentation
│   ├── PROJECT_HISTORY.md
│   ├── PROJECT_PRESENTATION.md
│   ├── APPROACH_COMPARISON.md     ⭐ NEW
│   ├── REFERENCES.md
│   └── music_project_info.md
│
├── datasets/
│   ├── GTZAN/Data/genres_original/    (1,000 tracks, 10 genres)
│   └── FMA/Data/fma_medium/           (25,000 tracks, 16 genres)
│
└── AI_models/OpenJMLA/
    ├── epoch_20.pth                   (330MB)
    └── epoch_4-step_8639-allstep_60000.pth (1.3GB)
```

### Performance Expectations

| Method | Dataset | Expected Accuracy | Time | GPU |
|--------|---------|------------------|------|-----|
| XGBoost | FMA | 50-60% | 5 min | No |
| CNN | GTZAN | 60-70% | 30 min | Yes |
| OpenJMLA V2 | GTZAN | 70-80% | 45 min | Yes |
| OpenJMLA + FMA | FMA | 75-85% | 2 hrs | Yes |

### Next Steps
1. Run `quick_baseline.py` for immediate results
2. Train `train_gtzan_v2.py` for best accuracy
3. Analyze confusion matrix to identify problem genres
4. Consider ensemble approach for production

### Files Created This Session
- train_gtzan_openjmla.py
- train_xgboost_fma.py
- quick_baseline.py
- compare_models.py
- train_gtzan_v2.py ⭐
- analyze_data.py
- gpu_monitor.py
- APPROACH_COMPARISON.md
- scripts/README.md

---

   - Created `music_project_info.md` with project overview
   - Documented recommended Python libraries

---

## Session 2: November 23, 2025 (Environment & Dataset Setup)

### Morning: Python Environment Setup
1. **Package verification and installation**
   - Checked existing packages: torch 2.8.0, torchaudio 2.8.0, matplotlib 3.5.1
   - Installed librosa 0.11.0 with dependencies
   - Attempted essentia (failed due to missing fftw3f)
   - Downgraded numpy from 2.2.6 → 1.26.4 → 1.24.4 for scipy compatibility

### AI Model Setup
1. **OpenJMLA model organization**
   - Moved from `~/OpenJMLA/` to `/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/`
   - Identified Git LFS pointers (some files only 135 bytes)
   - Successfully downloaded actual models:
     - `epoch_20.pth` (330MB) - Early checkpoint
     - `epoch_4-step_8639-allstep_60000.pth` (1.3GB) - Main model
   - Verified model structure: Vision Transformer (ViT), 150 parameters, 768 embedding dim

### Dataset Expansion
1. **GTZAN Dataset**
   - Location: `/media/mijesu_970/SSD_Data/DataSets/GTZAN/`
   - Structure: Data/ (10 genre folders with audio), Misc/ (spectrograms)
   - 1,000 audio files total (100 per genre)

2. **FMA Medium Dataset** (Major Download)
   - Downloaded: 22.7GB audio files (~2 hours)
   - Downloaded: 342MB metadata
   - Location: `/media/mijesu_970/SSD_Data/DataSets/FMA/`
   - Contents: 25,000 tracks, 16 genres
   - Extracted and organized into Data/ and Misc/ folders

3. **Future datasets documented**
   - MagnaTagATune: 25,863 clips, 188 tags (~50GB)
   - Million Song Dataset: 1M songs metadata (~280GB)

### Afternoon: Training Script Development
1. **Created `Classifed_JMLA_GTZAN.py`**
   - Transfer learning approach: OpenJMLA → Classification layer
   - AudioDataset class with mel-spectrogram conversion
   - GenreClassifier CNN model
   - Training loop with 10 epochs

2. **Script optimizations**
   - Added memory management (`clear_memory()` function)
   - Added GPU memory monitoring (`show_gpu_memory()`)
   - Fixed audio length inconsistency (padding/cropping to 30 seconds)
   - Configured for GPU-only training with batch size 2
   - Aggressive memory clearing after each batch and epoch

3. **Troubleshooting**
   - Fixed numpy version compatibility (scipy warning)
   - Fixed tensor size mismatch in DataLoader
   - Addressed CUDA memory allocation issues

### Project Organization
1. **Created project folder structure**
   - Main folder: `/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass/`
   - Moved all project materials to centralized location
   - Updated project summary with new location

2. **Final project files**
   - `music_project_info.md` - Complete project documentation
   - `Classifed_JMLA_GTZAN.py` - Main training script
   - `check_model.py` - Model inspection
   - `load_jmla_model.py` - JMLA loader
   - `music_genre_classifier.py` - Classifier
   - `pytorch_example.py` - Examples
   - `download_fma_medium.sh` - FMA download script

---

## Current Status

### ✅ Completed
- Python environment configured with all required libraries
- OpenJMLA models downloaded and verified (1.63GB total)
- GTZAN dataset ready (1,000 tracks, 10 genres)
- FMA Medium dataset downloaded (25,000 tracks, 16 genres)
- Training script created with GPU optimization
- Project organized in dedicated folder

### 🔄 In Progress
- Training script ready to run (pending execution)
- Model training on GTZAN/FMA datasets

### 📋 Next Steps
1. Execute training script on GTZAN dataset
2. Evaluate model performance
3. Fine-tune hyperparameters if needed
4. Test on FMA Medium dataset
5. Apply trained model to Music_TBC folder for classification
6. Consider adding MagnaTagATune for multi-label classification

---

## Technical Stack

**Hardware:** Jetson (ARM64 with CUDA)
**OS:** Linux
**Python:** 3.10.12
**Key Libraries:**
- PyTorch 2.8.0 (with CUDA support)
- torchaudio 2.8.0
- librosa 0.11.0
- numpy 1.24.4
- matplotlib 3.5.1

**Datasets:**
- GTZAN: 1,000 tracks, 10 genres
- FMA Medium: 25,000 tracks, 16 genres

**Models:**
- OpenJMLA Vision Transformer (330MB + 1.3GB checkpoints)
- Custom CNN classifier for genre classification

---

## Lessons Learned

1. **Git LFS Management:** Model files from Git repos may be pointers; need `git lfs pull`
2. **Memory Management:** Jetson requires aggressive memory clearing for GPU training
3. **Audio Processing:** Must pad/crop audio to consistent length for batch processing
4. **Package Compatibility:** numpy/scipy version conflicts require careful management
5. **Dataset Organization:** Consistent Data/Misc structure improves project organization

---

*Last Updated: November 23, 2025, 14:12*


## Session 4: November 24, 2025 (MSD Feature Training & FMA RTX Setup)

### Objectives
- Understand music classification features
- Train using MSD-style pre-computed features
- Create RTX-optimized FMA training script

### Achievements
- **Completed MSD feature-based training**: 77.09% accuracy in 2 minutes
- Created comprehensive feature documentation (CLASSIFICATION_FEATURES.md)
- Built RTX-optimized FMA training script (train_fma_rtx.py)
- Established model storage policy (all .pth in AI_models folder)
- Investigated Tagtraum dataset (no longer available)

### Models Trained
- **msd_model.pth**: 77% accuracy, 16 genres, 17,000 FMA tracks, 518 features
- Location: `/media/mijesu_970/SSD_Data/AI_models/msd_model.pth`
- Size: 672 KB

### Scripts Created
1. **train_msd.py** - Feature-based training using FMA pre-computed features
2. **train_fma_rtx.py** - RTX-optimized FMA audio training (ready to run)
3. **download_tagtraum.py** - Tagtraum download helper (sources unavailable)

### Documentation Created
1. **CLASSIFICATION_FEATURES.md** - Complete guide on music classification features
2. **SESSION_4_SUMMARY.md** - Detailed session summary

### Key Insights
- Feature-based training is 15x faster than audio-based (2 min vs 30 min)
- FMA pre-computed features achieve 77% accuracy without audio processing
- 518 features include: chroma, MFCC, spectral, tonnetz, zero-crossing
- Tagtraum dataset no longer available, FMA built-in labels work better
- Simple MLP (256→128→16) sufficient for feature-based classification

### Next Steps
1. Run FMA RTX audio training (train_fma_rtx.py)
2. Create classification script for msd_model.pth
3. Build ensemble model combining features + audio
4. Test on Musics_TBC folder
