# FMA Genre Classification Kaggle Notebook Analysis

**Date:** November 25, 2025  
**Time:** 07:43 (GMT+8)  
**Source:** `Reference/genre-classification-with-fma-data.ipynb`

---

## Overview

**Approach:** Traditional ML (XGBoost) with pre-computed features  
**Dataset:** FMA Medium (25,000 tracks, 16 genres)  
**Features:** 518 pre-computed audio features  
**Method:** Feature selection + PCA + XGBoost

---

## Data Processing Pipeline

### 1. Feature Loading
**Input:** `features.csv` (951 MB)
- 106,574 tracks × 518 features
- Features include: MFCC, chroma, spectral, tonnetz, zcr, rmse

**Processing:**
```python
def rename_fma_features(features):
    # Rename columns to: feature_number_statistic
    # Example: mfcc_01_mean, chroma_cqt_05_std
```

### 2. Label Extraction
**Input:** `tracks.csv` + `genres.csv`

**Steps:**
1. Extract `genre_top` from tracks.csv
2. 56,976 tracks missing genre_top
3. Fill missing using genre hierarchy from genres.csv
4. Final: 104,343 tracks with labels, 2,231 dropped (no genre)

**Genre Distribution (before filtering):**
- Electronic: ~15,000
- Experimental: ~12,000
- Rock: ~10,000
- Hip-Hop: ~8,000
- Folk: ~6,000
- Instrumental: ~5,000
- Pop: ~4,000
- International: ~3,000
- Others: <3,000 each

### 3. Data Filtering
**Removed:**
- "International" genre (too diverse)
- Genres with <1,000 samples (too small)

**Final Dataset:**
- 10 genres
- ~90,000+ tracks
- Balanced distribution

---

## Feature Engineering

### Manual Feature Selection
**Rationale:** Based on MIR (Music Information Retrieval) experience

**Removed Features:**
- Chromagram variants (STFT, CENS) - kept only CQT
- Spectral bandwidth & rolloff - highly correlated with centroid

**Selected Features:**
- MFCC (Mel-Frequency Cepstral Coefficients)
- Chroma CQT
- Spectral centroid
- Tonnetz (tonal centroid features)
- Zero crossing rate (ZCR)
- RMS energy

### Normalization
```python
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
```

### Dimensionality Reduction
**PCA (Principal Component Analysis):**
- Input: 518 features
- Output: 60 components
- Explained variance: 80%
- Reduces computational cost
- Removes noise and redundancy

**Visualization:**
- Cumulative explained variance plot
- 60 components capture 80% variance
- 90% variance requires ~100 components

---

## Model Training

### XGBoost Classifier
```python
xgb = XGBClassifier(n_estimators=50)
xgb.fit(X_train, y_train)
```

**Parameters:**
- n_estimators: 50 trees
- Default hyperparameters

**Train/Test Split:**
- Training: 70%
- Testing: 30%
- Shuffle: True
- Random state: 123

---

## Results

### Performance Metrics
- **Accuracy:** ~55-60%
- **F1 Score (macro):** ~0.50

### Confusion Matrix Analysis

**High Confusion Pairs:**
1. Rock ↔ Blues (similar instrumentation)
2. Electronic ↔ Pop (electronic elements in pop)
3. Electronic ↔ Hip-Hop (beats and production)

**Observations:**
1. Model biased toward common labels (Electronic, Experimental, Rock)
2. Genre similarity causes confusion
3. "Experimental" and "Instrumental" are not musically distinct genres

---

## Key Insights

### Advantages
✅ **Fast training:** Minutes vs hours for deep learning  
✅ **Interpretable:** Feature importance analysis possible  
✅ **No GPU required:** Runs on CPU  
✅ **Small model size:** <10 MB vs 100+ MB for neural networks  
✅ **Good baseline:** 55-60% accuracy without audio processing

### Limitations
❌ **Lower accuracy:** 55-60% vs 70-90% for deep learning  
❌ **Fixed features:** Cannot learn new representations  
❌ **Manual selection:** Requires domain expertise  
❌ **Genre confusion:** Similar genres hard to distinguish

### Comparison with Deep Learning

| Approach | Accuracy | Training Time | Model Size | GPU |
|----------|----------|---------------|------------|-----|
| XGBoost + PCA | 55-60% | 5-10 min | <10 MB | No |
| CNN (Basic) | 70-80% | 45 min | ~50 MB | Yes |
| Transfer Learning | 80-90% | 4 hours | ~50 MB | Yes |

---

## Methodology Summary

```
FMA features.csv (518 features)
         ↓
Manual Feature Selection
         ↓
StandardScaler Normalization
         ↓
PCA (518 → 60 components)
         ↓
XGBoost Classifier (50 trees)
         ↓
Predictions (10 genres)
```

---

## Recommendations

### For Quick Baseline
- Use this XGBoost approach
- Fast results for initial testing
- Good for feature importance analysis

### For Production
- Use deep learning (CNN or transfer learning)
- Higher accuracy (70-90%)
- Better generalization

### For Ensemble
- Combine XGBoost + CNN predictions
- Voting or averaging
- Potential 5-10% accuracy boost

---

## Related Files

**Notebook:** `Reference/genre-classification-with-fma-data.ipynb`  
**Dataset:** FMA Medium (25,000 tracks)  
**Features:** `DataSets/FMA/Misc/fma_metadata/features.csv`  
**Converted:** `AI_models/FMA/FMA.npy` (211 MB)

---

**Status:** ✅ Analyzed - Traditional ML baseline approach documented
