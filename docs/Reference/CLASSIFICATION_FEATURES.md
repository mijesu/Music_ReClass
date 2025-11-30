# Music Classification Features

## Overview
This document explains the features used for music genre classification in the Music_ReClass project.

## 1. Audio Features (from raw audio files)

### Mel-Spectrograms
- Visual representation of frequency content over time
- Used by GTZAN and JMLA models
- Typical size: 128x128 or 224x224 pixels
- Shows time on x-axis, frequency on y-axis, intensity as color

### MFCCs (Mel-Frequency Cepstral Coefficients)
- Captures timbral texture of audio
- Represents the shape of the vocal tract
- Typically 13-40 coefficients
- Good for distinguishing instrument types

### Chroma Features
- Pitch class distribution (12 semitones: C, C#, D, etc.)
- Captures harmonic and melodic characteristics
- Genre-specific chord progressions

### Spectral Features
- **Spectral Centroid** - "Center of mass" of spectrum (brightness)
- **Spectral Rolloff** - Frequency below which 85% of energy is contained
- **Spectral Contrast** - Difference between peaks and valleys
- **Spectral Bandwidth** - Width of frequency range

## 2. Pre-computed Features (MSD-style)

### Timbre (12 dimensions)
- Tone color and texture
- Distinguishes instruments playing same note
- Extracted from audio segments

### Pitch (12 dimensions)
- Musical note distribution
- Chroma-based representation
- Key and scale information

### Rhythm Features
- **Tempo** - Beats per minute (BPM)
- **Beat Strength** - Regularity of rhythm
- **Danceability** - Suitability for dancing

### Dynamic Features
- **Loudness** - Overall volume in dB
- **Energy** - Intensity and activity level

### Structural Features
- **Duration** - Track length in seconds
- **Key** - Musical key (0-11)
- **Mode** - Major (1) or Minor (0)

## 3. High-level Features

### Danceability
- How suitable the track is for dancing
- Based on tempo, rhythm stability, beat strength

### Energy
- Perceptual measure of intensity and activity
- Fast, loud, noisy tracks have high energy

### Valence
- Musical positivity/mood
- High valence = happy, cheerful
- Low valence = sad, angry

### Acousticness
- Confidence measure of acoustic vs electronic
- 1.0 = fully acoustic, 0.0 = fully electronic

## Current Models in Music_ReClass

| Model | Features Used | Input Type | Training Time | Accuracy |
|-------|--------------|------------|---------------|----------|
| **JMLA** (OpenJMLA) | Mel-spectrograms | Audio → 128x128 spectrograms | Pre-trained | Clustering |
| **GTZAN Basic** (train_gtzan_rtx.py) | Mel-spectrograms | Audio → spectrograms | 15-20 min | 70-85% |
| **GTZAN Enhanced** (train_gtzan_enhanced.py) | Mel-spectrograms | Audio → spectrograms | 4 hours | 80-90% |
| **MSD Features** (train_msd_features.py) | 31 pre-computed features | HDF5 files | ~1 hour | 65-75% |
| **Combined Multi-Modal** (train_combined_4hr.py) | Spectrograms + MSD features | Dual input | 4 hours | 75-85% |
| **Progressive FMA** (train_fma_progressive.py) | Mel-spectrograms | Audio → spectrograms | 12h + 2h | 75-85% |

## Why Spectrograms Work Best

1. **Visual Pattern Recognition**
   - CNNs can learn patterns like humans see in sheet music
   - Genre-specific visual signatures

2. **Time-Frequency Relationships**
   - Captures both temporal and spectral information
   - Shows how sound evolves over time

3. **Genre-Specific Patterns**
   - Metal: Dense high frequencies, distortion patterns
   - Blues: Specific chord progressions, bent notes
   - Classical: Complex harmonic structures
   - Hip-hop: Strong bass, repetitive beats
   - Electronic: Synthetic textures, precise rhythms

4. **Transfer Learning**
   - Pre-trained image models (ResNet, VGG) work on spectrograms
   - Leverages computer vision advances

## Feature Extraction Process

### For Audio Files (JMLA/GTZAN approach):
```
Audio File (.wav/.mp3)
    ↓
Load audio (librosa)
    ↓
Extract mel-spectrogram
    ↓
Resize to 128x128 or 224x224
    ↓
Normalize
    ↓
Feed to CNN
```

### For MSD Pre-computed Features:
```
HDF5 File (.h5)
    ↓
Read features (h5py)
    ↓
Extract: timbre (12) + pitch (12) + rhythm (7)
    ↓
Normalize
    ↓
Feed to MLP
```

### For Combined Multi-Modal:
```
Audio File + HDF5 File
    ↓
Branch 1: Spectrogram → CNN
Branch 2: Features → MLP
    ↓
Concatenate outputs
    ↓
Final classification layer
```

## JMLA Classification Results

Classification of 25 Chinese music files from Musics_TBC folder:
- 6 files → Metal
- 4 files → Hip-Hop
- 3 files → Blues
- 3 files → Country
- 3 files → Rock
- Remaining → Other genres

Used JMLA model with K-Means clustering on extracted features.

## Datasets and Their Features

### GTZAN
- 1,000 tracks, 10 genres
- Raw audio files
- Features: Extracted on-the-fly (spectrograms)

### FMA Medium
- 25,000 tracks, 16 genres
- Raw audio files + metadata CSV
- Features: Extracted on-the-fly + pre-computed metadata

### MSD (Million Song Dataset)
- 1 million tracks (subset: 10,000)
- NO audio files - only pre-computed features
- Features: All pre-computed in HDF5 format
- Requires genre labels from Tagtraum or Last.fm

## Recommendations

**For highest accuracy (85-90%):**
- Use mel-spectrograms with deep CNN (ResNet)
- Train on large dataset (FMA Medium)
- Apply data augmentation
- Use ensemble of multiple models

**For fastest training (15-20 min):**
- Use mel-spectrograms with simple CNN
- Train on GTZAN (1,000 tracks)
- Batch size 32, mixed precision

**For minimal compute:**
- Use MSD pre-computed features
- Simple MLP (3 layers)
- No GPU required (but slower)

**For best of both worlds:**
- Multi-modal approach combining spectrograms + features
- Leverages visual patterns + numerical features
- 4-hour training on RTX 4060 Ti

## Next Steps

1. Download Tagtraum genre labels for MSD training
2. Or focus on FMA dataset with built-in labels
3. Or continue with GTZAN for quick iterations
4. Consider ensemble approach combining multiple models
