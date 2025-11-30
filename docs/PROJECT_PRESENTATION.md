# Music Genre Classification with AI
## Transfer Learning Project

---

## 📋 Project Overview

**Objective:** Automatic music genre classification using deep learning

**Approach:** Transfer learning with OpenJMLA + GTZAN/FMA datasets

**Platform:** Jetson (ARM64 with CUDA)

**Timeline:** November 22-23, 2025

---

## 🎯 Project Goals

1. **Feature Extraction**
   - Use pre-trained OpenJMLA Vision Transformer
   - Extract meaningful audio features from spectrograms

2. **Classification**
   - Train custom classifier on music genres
   - Support 10-16 genre categories

3. **Application**
   - Classify unlabeled music in Music_TBC folder
   - Organize music library automatically

---

## 🗂️ Datasets

### GTZAN Dataset ✅
- **Size:** 1,000 tracks
- **Genres:** 10 (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock)
- **Use:** Baseline training and validation

### FMA Medium ✅
- **Size:** 25,000 tracks (~22GB)
- **Genres:** 16
- **Use:** Enhanced training with more diversity

### Future Datasets 📋
- **MagnaTagATune:** 25,863 clips, 188 tags (~50GB)
- **Million Song Dataset:** 1M songs metadata (~280GB)

---

## 🤖 AI Models

### OpenJMLA Vision Transformer
- **Type:** Pre-trained audio feature extractor
- **Architecture:** Vision Transformer (ViT)
- **Size:** 1.63GB (2 checkpoints)
- **Parameters:** 150 layers, 768 embedding dimension
- **Author:** MMSelfSup

### Custom Classifier
- **Type:** CNN-based genre classifier
- **Input:** Mel-spectrograms (128 bands)
- **Output:** Genre probabilities
- **Training:** Transfer learning approach

---

## 🛠️ Technical Stack

### Hardware
- **Platform:** NVIDIA Jetson
- **GPU:** CUDA-enabled
- **Memory:** Optimized for embedded system

### Software
- **Python:** 3.10.12
- **PyTorch:** 2.8.0 (CUDA)
- **Audio:** librosa 0.11.0, torchaudio 2.8.0
- **Visualization:** matplotlib 3.5.1

---

## 📊 Project Structure

```
/media/mijesu_970/SSD_Data/
├── Kiro_Projects/Music_Reclass/    # Project files
│   ├── Classifed_JMLA_GTZAN.py    # Main training script
│   ├── music_project_info.md       # Documentation
│   └── PROJECT_HISTORY.md          # Development log
├── DataSets/                        # Training data
│   ├── GTZAN/                      # 1,000 tracks
│   └── FMA/                        # 25,000 tracks
├── AI_models/OpenJMLA/             # Pre-trained models
└── Music_TBC/                      # Music to classify
```

---

## 🔄 Workflow

### 1. Data Preparation
- Load audio files (WAV format)
- Convert to mel-spectrograms
- Normalize and pad to consistent length

### 2. Feature Extraction
- Use OpenJMLA to extract audio features
- Generate high-level representations

### 3. Classification
- Train custom classifier on extracted features
- 10 epochs with Adam optimizer
- Batch size: 2 (GPU memory optimized)

### 4. Evaluation
- Measure accuracy on validation set
- Fine-tune hyperparameters

---

## ✅ Completed Milestones

### Session 1 (Nov 22)
- ✓ Project planning and structure
- ✓ Initial script development
- ✓ Documentation framework

### Session 2 (Nov 23)
- ✓ Python environment setup
- ✓ OpenJMLA models downloaded (1.63GB)
- ✓ GTZAN dataset organized
- ✓ FMA Medium downloaded (22GB, 2 hours)
- ✓ Training script with GPU optimization
- ✓ Memory management implementation

---

## 🚀 Key Features

### Memory Optimization
- Garbage collection after each batch
- CUDA cache clearing
- Batch size tuned for Jetson

### GPU Monitoring
- Real-time memory usage tracking
- Allocated vs Reserved memory display
- Performance optimization

### Audio Processing
- Consistent 30-second clips
- Padding/cropping for uniform size
- Mel-spectrogram conversion (128 bands)

---

## 🔧 Technical Challenges & Solutions

### Challenge 1: Git LFS Pointers
- **Problem:** Model files were placeholders (135 bytes)
- **Solution:** Used `git lfs pull` to download actual models

### Challenge 2: CUDA Memory Issues
- **Problem:** Out of memory errors on Jetson
- **Solution:** Reduced batch size, aggressive memory clearing

### Challenge 3: Audio Length Mismatch
- **Problem:** Variable audio lengths caused tensor errors
- **Solution:** Implemented padding/cropping to fixed length

### Challenge 4: Package Compatibility
- **Problem:** numpy 2.x incompatible with scipy
- **Solution:** Downgraded to numpy 1.24.4

---

## 📈 Current Status

### ✅ Ready
- Environment configured
- Datasets downloaded and organized
- Models verified and loaded
- Training script optimized

### 🔄 In Progress
- Model training execution
- Performance evaluation

### 📋 Next Steps
1. Run training on GTZAN
2. Evaluate accuracy
3. Test on FMA dataset
4. Apply to Music_TBC folder
5. Deploy classification system

---

## 💡 Innovation Points

1. **Transfer Learning**
   - Leverage pre-trained OpenJMLA
   - Reduce training time and data requirements

2. **Multi-Dataset Training**
   - Combine GTZAN + FMA for robustness
   - 26,000 total training samples

3. **Embedded Optimization**
   - Optimized for Jetson platform
   - Real-time GPU memory management

4. **Scalable Architecture**
   - Easy to add new genres
   - Extensible to multi-label classification

---

## 🎓 Lessons Learned

1. **Model Management**
   - Always verify actual file sizes vs Git LFS pointers
   - Keep multiple checkpoints for flexibility

2. **Memory Constraints**
   - Embedded systems require aggressive optimization
   - Monitor GPU memory continuously

3. **Data Consistency**
   - Audio preprocessing critical for batch training
   - Standardize all inputs to same dimensions

4. **Package Dependencies**
   - Version compatibility matters
   - Test environment before large downloads

---

## 📊 Expected Outcomes

### Performance Targets
- **Accuracy:** >70% on GTZAN (baseline)
- **Accuracy:** >75% on FMA (with more data)
- **Speed:** <1 second per track (inference)

### Deliverables
- Trained genre classifier model
- Classification script for new music
- Organized music library
- Complete documentation

---

## 🔮 Future Enhancements

### Short Term
- Add validation metrics (precision, recall, F1)
- Implement confusion matrix visualization
- Create inference script for Music_TBC

### Long Term
- Multi-label classification (MagnaTagATune)
- Real-time classification API
- Web interface for music organization
- Integration with music players

---

## 📚 Resources

### Documentation
- `music_project_info.md` - Complete project guide
- `PROJECT_HISTORY.md` - Development timeline
- Code comments in all scripts

### Datasets
- GTZAN: `/media/mijesu_970/SSD_Data/DataSets/GTZAN/`
- FMA: `/media/mijesu_970/SSD_Data/DataSets/FMA/`

### Models
- OpenJMLA: `/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/`

### Scripts
- Training: `Classifed_JMLA_GTZAN.py`
- Model check: `check_model.py`
- Utilities: Various helper scripts

---

## 🙏 Acknowledgments

- **OpenJMLA Team** - Pre-trained model
- **GTZAN Dataset** - Genre classification benchmark
- **FMA** - Free Music Archive dataset
- **PyTorch Team** - Deep learning framework
- **librosa** - Audio processing library

---

## 📞 Project Information

**Project Name:** Music Genre Classification with AI

**Location:** `/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass/`

**Status:** Development Phase

**Last Updated:** November 23, 2025

---

# Thank You!

## Questions?

**Project Repository:** `/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass/`

**Documentation:** See `music_project_info.md` and `PROJECT_HISTORY.md`
