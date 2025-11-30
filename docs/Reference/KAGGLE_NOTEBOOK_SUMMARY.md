# Summary: Genre Classification with FMA Data

**Source:** https://www.kaggle.com/code/jojothepizza/genre-classification-with-fma-data

---

## Overview

This notebook demonstrates music genre classification using the FMA (Free Music Archive) dataset with traditional machine learning approaches using pre-computed audio features.

---

## Approach

### 1. Data Preprocessing

**Features Used:**
- Pre-computed audio features from FMA metadata (`features.csv`)
- Features include: MFCCs, spectral features, chroma features, tempo, etc.
- Total: ~500+ audio features per track

**Labels:**
- Uses `genre_top` (top-level genre) from FMA metadata
- Original dataset: 106,574 tracks
- After cleaning: ~104,000 tracks with valid genres

**Data Cleaning:**
- Removed tracks without `genre_top` labels
- Filled missing `genre_top` using parent genre from `genres.csv`
- Removed 'International' genre (not musically characteristic)
- Removed genres with <1,000 songs (small sample size)
- Final dataset: Focused on major genres only

---

## Feature Engineering

### Manual Feature Selection

**Removed highly correlated features:**
- Kept only CQT chromagram (removed STFT and CENS versions)
- Removed spectral bandwidth and rolloff (kept centroids only)
- Rationale: Reduce redundancy and computational cost

**Feature Analysis:**
- Created correlation heatmaps to identify redundant features
- Focused on "mean" statistics of audio features
- Reduced feature set from 500+ to more manageable subset

### Dimensionality Reduction

**PCA (Principal Component Analysis):**
- Applied StandardScaler for normalization
- Used PCA to reduce dimensions
- Selected 60 components (explains ~80% variance)
- Transformed features: 500+ → 60 PCA components

---

## Model Training

### Algorithm: XGBoost Classifier

**Configuration:**
- Model: XGBClassifier
- n_estimators: 50
- Features: 60 PCA components
- Train/Test split: 70/30

**Data Split:**
- Training set: 70% of data
- Test set: 30% of data
- Random state: 123 (for reproducibility)
- Shuffle: True

---

## Key Differences from Our Approach

| Aspect | Kaggle Notebook | Our Approach |
|--------|----------------|--------------|
| **Features** | Pre-computed FMA features | Mel-spectrograms (raw audio) |
| **Model** | XGBoost (traditional ML) | CNN / Transfer Learning (Deep Learning) |
| **Feature Extraction** | Manual selection + PCA | OpenJMLA / CNN automatic learning |
| **Dataset** | FMA Medium/Large | GTZAN + FMA Medium |
| **Preprocessing** | StandardScaler + PCA | Audio → Mel-spectrogram |
| **Approach** | Feature engineering heavy | End-to-end learning |

---

## Advantages of Kaggle Approach

✅ **Fast training** - XGBoost is quick with tabular data
✅ **Interpretable** - Can analyze feature importance
✅ **Low memory** - No need to load audio files
✅ **Pre-computed features** - Uses FMA's provided features
✅ **Traditional ML** - Works well with limited compute

---

## Advantages of Our Approach

✅ **End-to-end learning** - Learns features automatically
✅ **Transfer learning** - Leverages OpenJMLA pre-training
✅ **Raw audio** - Not limited to pre-computed features
✅ **Deep learning** - Can capture complex patterns
✅ **Scalable** - Can work with any audio dataset

---

## What We Can Learn

### 1. Feature Selection Strategy
- Remove highly correlated features
- Use correlation heatmaps for analysis
- Focus on meaningful feature subsets

### 2. Data Cleaning
- Remove ambiguous genres ('International')
- Filter out genres with insufficient samples
- Use parent genres for missing labels

### 3. Dimensionality Reduction
- PCA can reduce features while preserving variance
- 60 components for ~80% variance is a good target
- Normalize features before PCA

### 4. Baseline Comparison
- XGBoost provides a strong baseline
- Can compare our deep learning results against this

---

## Potential Improvements to Our Project

### 1. Hybrid Approach
```python
# Combine deep learning features + traditional features
openjmla_features = extract_features(audio)  # 768-dim
fma_features = load_fma_features(track_id)  # Pre-computed
combined = concatenate([openjmla_features, fma_features])
classifier(combined)
```

### 2. Use FMA Pre-computed Features
- FMA provides pre-computed audio features
- Can use as additional input or for comparison
- Located in: `fma_metadata/features.csv`

### 3. Genre Filtering
- Apply same filtering: remove small genres
- Focus on top 8 genres with >1000 samples
- Remove ambiguous categories

### 4. Baseline Comparison
- Train XGBoost on FMA features as baseline
- Compare against our CNN/OpenJMLA approach
- Measure improvement from deep learning

---

## Implementation Ideas

### Quick Baseline with FMA Features

```python
# Load FMA pre-computed features
features = pd.read_csv('fma_metadata/features.csv')
tracks = pd.read_csv('fma_metadata/tracks.csv')

# Extract genre labels
labels = tracks['genre_top']

# Train XGBoost
from xgboost import XGBClassifier
model = XGBClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Compare with our deep learning model
```

### Feature Importance Analysis

```python
# After training XGBoost
importance = model.feature_importances_
# Identify which audio features matter most
# Use insights to improve our CNN architecture
```

---

## Conclusion

The Kaggle notebook demonstrates a **traditional ML approach** using pre-computed features and XGBoost. It's fast, interpretable, and effective.

Our approach uses **deep learning with raw audio**, which is more flexible and can learn features automatically through OpenJMLA transfer learning.

**Best strategy:** Use both approaches
1. XGBoost baseline for quick results and feature analysis
2. Deep learning for maximum performance
3. Compare and combine insights

---

*Summary created: 2025-11-23*
*Notebook analyzed: genre-classification-with-fma-data.ipynb*
