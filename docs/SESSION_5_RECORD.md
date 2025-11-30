# Session 5 Record - Model Evaluation & Dataset Expansion

**Date:** November 25, 2025  
**Time:** 05:30 - 09:07 (GMT+8)  
**Duration:** ~3.5 hours

---

## Summary

Evaluated additional pre-trained models for the project and expanded dataset coverage with MagnaTagATune (MTT). Successfully processed 25,863 audio clips and integrated multi-label annotations.

---

## Tasks Completed

### 1. Project Organization ✅

**File Renaming:**
- `music_project_info.md` → `Project_Info.md`

**File Organization:**
- Moved .pth model files to `/AI_models/trained_models/`
- Moved .log files to `/AI_models/training_logs/`
- Updated SUMMARY.md with storage guidelines

**Python Scripts:**
- Organized classification scripts into `classification/` folder
- Moved training scripts to `training/` folder
- Cleaned up duplicate files

### 2. Dataset Analysis ✅

**Genre Lists Documented:**

**GTZAN (10 genres):**
1. Blues, 2. Classical, 3. Country, 4. Disco, 5. Hip-Hop, 6. Jazz, 7. Metal, 8. Pop, 9. Reggae, 10. Rock

**FMA Medium (16 genres):**
1. Blues, 2. Classical, 3. Country, 4. Easy Listening, 5. Electronic, 6. Experimental, 7. Folk, 8. Hip-Hop, 9. Instrumental, 10. International, 11. Jazz, 12. Old-Time/Historic, 13. Pop, 14. Rock, 15. Soul-RnB, 16. Spoken

**MSD - Million Song Dataset (13 genres):**
1. Blues, 2. Country, 3. Electronic, 4. Folk, 5. International, 6. Jazz, 7. Latin, 8. New Age, 9. Pop_Rock, 10. Rap, 11. Reggae, 12. RnB, 13. Vocal

**MTT - MagnaTagATune (29 genre tags + 159 other tags):**
- Genres: rock, pop, jazz, classical, blues, country, folk, reggae, metal, punk, disco, techno, trance, hip hop, rap, funk, electronic, electronica, ambient, new age, opera, world, hard rock, soft rock, heavy metal, industrial, dance, house, jungle
- Multi-label: Each clip can have multiple tags

### 3. MagnaTagATune (MTT) Dataset Processing ✅

**Input:** 25,837 Echonest XML files

**Script Created:** `utils/convert_mtt_features.py`

**Features Extracted (36 dimensions):**
- Duration, loudness, tempo, tempo confidence
- Time signature, key, mode (with confidence)
- Timbre mean/std (12 dimensions each)
- Pitch mean/std (12 dimensions each)
- Loudness statistics

**Output Files:**
- `MTT.npy` - 3.55 MB (features only)
- `MTT.pth` - 5.33 MB (features + metadata)

**Annotation Integration:**

**Scripts Created:**
- `utils/combine_mtt_annotations.py` (PyTorch format)
- `utils/combine_mtt_npy.py` (NumPy format)

**Output Files:**
- `MTT_combined.pth` - 26 MB (features + 188 tags)
- `MTT_combined.npz` - 2.92 MB (compressed)

**Processing Time:** ~10 minutes for 25,837 files

### 4. Model Evaluation ✅

**VGGish (Google AudioSet):**
- ✅ Downloaded: 276 MB
- ✅ Location: `/AI_models/VGGish/`
- ✅ PyTorch compatible
- Output: 128-dim embeddings
- Use: General audio feature extraction

**OpenJMLA:**
- ✅ Already available: 1.3 GB
- ✅ Location: `/AI_models/OpenJMLA/`
- Output: 768-dim embeddings
- Use: Music-specific feature extraction

**CLMR (Contrastive Learning):**
- ❌ No pre-trained weights available
- ❌ PyTorch 1.9.0 required (incompatible with 2.8.0)
- 📚 Repository cloned for reference
- Status: Reference code only

**musicnn:**
- ❌ Skipped by user decision
- Alternative to CLMR
- MTT pre-trained available but not needed

**OpenAI Jukebox:**
- ❌ Models removed from public access (404 errors)
- ❌ Old package incompatible with Python 3.10+
- Not suitable: Designed for generation, not classification
- Status: Unavailable

### 5. Reference Materials Reviewed ✅

**musicnn_example.ipynb:**
- Demonstrates musicnn as feature extractor
- MTT_musicnn model architecture
- Features: timbral (408), temporal (153), CNN layers
- Output: taggram predictions

**genre-classification-with-fma-data.ipynb:**
- XGBoost approach with FMA features
- Feature selection + PCA (518 → 60 components)
- Results: 55-60% accuracy
- Fast baseline without deep learning

**Records Created:**
- `MTT_Dataset_Record.md` - MTT processing details
- `FMA_Kaggle_Notebook_Record.md` - Kaggle analysis

---

## Files Created/Modified

### New Scripts (3)
1. `utils/convert_mtt_features.py` - Extract Echonest XML features
2. `utils/combine_mtt_annotations.py` - Combine PTH with annotations
3. `utils/combine_mtt_npy.py` - Combine NPZ with annotations

### New Documentation (4)
1. `MTT_Dataset_Record.md` - MTT processing record
2. `FMA_Kaggle_Notebook_Record.md` - Kaggle notebook analysis
3. `AI_models/CLMR/README.md` - CLMR status
4. `AI_models/Jukebox/README.md` - Jukebox status

### Updated Files (2)
1. `Project_Info.md` - Updated with complete dataset info
2. `SUMMARY.md` - Added model storage guidelines

---

## Dataset Status

| Dataset | Status | Clips | Genres | Features | Size |
|---------|--------|-------|--------|----------|------|
| GTZAN | ✅ Ready | 1,000 | 10 | Mel-spec | 1.2 GB |
| FMA Medium | ✅ Ready | 25,000 | 16 | 518 pre-computed | 22 GB |
| MSD | ✅ Ready | 10,000 | 13 | Echonest | 2.6 GB |
| MTT | ✅ Ready | 25,863 | 29 + 159 tags | 36 Echonest | 50 GB |

---

## Model Status

| Model | Status | Size | Output | Use Case |
|-------|--------|------|--------|----------|
| OpenJMLA | ✅ Available | 1.3 GB | 768-dim | Music features |
| VGGish | ✅ Available | 276 MB | 128-dim | General audio |
| CLMR | 📚 Reference | - | - | Code reference |
| musicnn | ❌ Skipped | - | - | Not needed |
| Jukebox | ❌ Unavailable | - | - | Removed by OpenAI |

---

## Key Insights

### MTT Dataset
- **Multi-label classification:** 188 tags per clip
- **Diverse tags:** Genres, instruments, moods, characteristics
- **Echonest features:** 36 dimensions (tempo, key, timbre, pitch)
- **Efficient storage:** NPZ format (2.92 MB) vs PTH (26 MB)

### Model Availability
- **VGGish:** Easy to install, auto-downloads
- **OpenJMLA:** Already available, music-specific
- **CLMR/Jukebox:** Not practically available
- **Conclusion:** VGGish + OpenJMLA sufficient for project

### Feature Extraction Approaches
1. **Pre-computed features** (FMA, MSD, MTT): Fast, small files
2. **Deep learning embeddings** (VGGish, OpenJMLA): Better accuracy
3. **Hybrid approach**: Combine both for ensemble

---

## Next Steps

### Immediate
1. Create VGGish feature extraction script for datasets
2. Train multi-label classifier on MTT dataset
3. Compare VGGish vs OpenJMLA embeddings

### Short-term
1. Build ensemble model (VGGish + OpenJMLA + XGBoost)
2. Test on Music_TBC folder
3. Generate classification reports

### Long-term
1. Multi-label classification (MTT 188 tags)
2. Deploy as REST API
3. Web interface

---

## Statistics

**Time Spent:** 3.5 hours

**Files Created:** 7
- Scripts: 3
- Documentation: 4

**Data Processed:** 25,863 audio clips

**Models Evaluated:** 5
- Available: 2 (VGGish, OpenJMLA)
- Unavailable: 3 (CLMR, musicnn, Jukebox)

**Storage Used:**
- MTT features: 8.8 MB (npy + pth + npz + combined)
- VGGish model: 276 MB
- Total new: ~285 MB

---

**Session Status:** ✅ Complete  
**Next Session:** VGGish feature extraction and model comparison
