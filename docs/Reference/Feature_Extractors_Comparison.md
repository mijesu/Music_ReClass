# Feature Extractors Comparison

## Overview Table

| Model | Size | Output Dims | Architecture | Training Data | Music Focus | Genre Training |
|-------|------|-------------|--------------|---------------|-------------|----------------|
| **VGGish** | 276 MB | 128 | CNN | AudioSet (632 classes) | ⚠️ Mixed | ❌ No |
| **MERT-330M** | 1.2 GB | 768 | Transformer | 160K hrs music | ✅ Music only | ❌ No (self-supervised) |
| **CLAP** | 741 MB | 512 | Contrastive | Music + text pairs | ✅ Music variant | ⚠️ Implicit |
| **EnCodec** | 89 MB | Variable | Codec | General audio | ⚠️ Mixed | ❌ No |
| **AST** | 331 MB | 768 | ViT | AudioSet (632 classes) | ⚠️ Mixed | ❌ No |
| **HuBERT** | 1.2 GB | 1024 | Self-supervised | Libri-Light (speech) | ❌ Speech | ❌ No |
| **PANNs** | 331 MB | 2048 | CNN | AudioSet (527 classes) | ⚠️ Mixed | ❌ No |

## Detailed Comparison

### 1. VGGish (Google Research)
**Strengths:**
- Small size (276 MB)
- Fast inference
- Well-established baseline
- Compact 128-dim embeddings

**Weaknesses:**
- Not music-specific
- Trained on general audio events
- Lower dimensional features
- Older architecture (2017)

**Best For:** General audio classification, baseline comparisons

---

### 2. MERT-v1-330M (Recommended for Music)
**Strengths:**
- Music-specific training (160K hours)
- State-of-the-art transformer architecture
- Rich 768-dim embeddings
- Self-supervised learning
- No dependency conflicts

**Weaknesses:**
- Large model size (1.2 GB)
- Slower inference than CNN models
- Requires more memory
- Not explicitly trained on genres

**Best For:** Music genre classification, music understanding, transfer learning

---

### 3. CLAP (LAION)
**Strengths:**
- Audio-text alignment
- Zero-shot classification possible
- Music-focused variant
- Semantic understanding

**Weaknesses:**
- Requires text descriptions
- Medium size (741 MB)
- Implicit genre knowledge only

**Best For:** Semantic search, audio-text tasks, zero-shot classification

---

### 4. EnCodec (Meta)
**Strengths:**
- Smallest model (89 MB)
- Fast inference
- Compression-based features

**Weaknesses:**
- Not designed for classification
- Low-level features only
- Variable output dimensions
- Poor for semantic tasks

**Best For:** Audio generation, compression, NOT recommended for genre classification

---

### 5. AST (MIT)
**Strengths:**
- Transformer architecture
- Good for spectrograms
- 768-dim embeddings
- AudioSet fine-tuned

**Weaknesses:**
- Not music-specific
- General audio focus
- Medium size (331 MB)

**Best For:** General audio understanding, spectrogram analysis

---

### 6. HuBERT (Facebook)
**Strengths:**
- Large model (1.2 GB)
- Self-supervised learning
- High-dimensional (1024-dim)
- Strong acoustic modeling

**Weaknesses:**
- Trained on SPEECH, not music
- Not suitable for music genres
- Large memory footprint
- Slow inference

**Best For:** Speech tasks, NOT recommended for music classification

---

### 7. PANNs CNN14
**Strengths:**
- Highest dimensional output (2048-dim)
- Good for audio tagging
- Medium size (331 MB)
- Strong instrument recognition

**Weaknesses:**
- Not music-genre specific
- General audio events
- High dimensionality may need reduction

**Best For:** Audio event detection, instrument recognition

---

## Performance Comparison

### Inference Speed (CPU - Jetson Orin)
| Model | Time per 30s audio | Relative Speed |
|-------|-------------------|----------------|
| VGGish | ~5-10s | ⚡⚡⚡⚡⚡ Fastest |
| EnCodec | ~5-10s | ⚡⚡⚡⚡⚡ Fastest |
| PANNs | ~10-15s | ⚡⚡⚡⚡ Fast |
| AST | ~15-20s | ⚡⚡⚡ Medium |
| CLAP | ~20-30s | ⚡⚡ Slow |
| MERT | ~30-60s | ⚡ Slowest |
| HuBERT | ~30-60s | ⚡ Slowest |

### Memory Usage
| Model | RAM Required | GPU VRAM |
|-------|-------------|----------|
| VGGish | ~500 MB | ~1 GB |
| EnCodec | ~300 MB | ~500 MB |
| PANNs | ~800 MB | ~1.5 GB |
| AST | ~800 MB | ~1.5 GB |
| CLAP | ~1.5 GB | ~2 GB |
| MERT | ~2.5 GB | ~4 GB |
| HuBERT | ~2.5 GB | ~4 GB |

### Expected Accuracy for Genre Classification
| Model | Accuracy (Estimated) | Confidence |
|-------|---------------------|------------|
| MERT | 70-85% | ⭐⭐⭐⭐⭐ Very High |
| CLAP | 65-75% | ⭐⭐⭐⭐ High |
| PANNs | 60-70% | ⭐⭐⭐ Medium |
| AST | 60-70% | ⭐⭐⭐ Medium |
| VGGish | 55-65% | ⭐⭐ Low |
| HuBERT | 40-55% | ⭐ Very Low |
| EnCodec | 35-50% | ⭐ Very Low |

---

## Recommendations by Use Case

### For Music Genre Classification (Priority):
1. **MERT-v1-330M** ⭐⭐⭐⭐⭐
   - Best choice for music understanding
   - Highest expected accuracy
   - Music-specific training

2. **CLAP** ⭐⭐⭐⭐
   - Good for semantic understanding
   - Zero-shot capability
   - Music variant available

3. **PANNs / AST** ⭐⭐⭐
   - General audio, needs fine-tuning
   - Good baseline performance

4. **VGGish** ⭐⭐
   - Fast baseline
   - Established benchmark

5. **HuBERT / EnCodec** ⭐
   - Not recommended for music genres

### For Real-time Applications:
1. **VGGish** - Fastest, smallest
2. **EnCodec** - Very fast
3. **PANNs** - Good speed/accuracy balance

### For Research/Experimentation:
1. **MERT** - State-of-the-art
2. **CLAP** - Novel approach
3. **Ensemble** - Combine multiple models

### For Production (Balanced):
1. **MERT** - Best accuracy
2. **PANNs** - Good speed/accuracy
3. **VGGish** - Fast baseline

---

## Ensemble Strategy

### Recommended Combination:
```python
# Extract features from multiple models
mert_features = extract_mert(audio)      # 768-dim
clap_features = extract_clap(audio)      # 512-dim
panns_features = extract_panns(audio)    # 2048-dim

# Concatenate
combined = np.concatenate([mert_features, clap_features, panns_features])
# Total: 3328-dim

# Train classifier on combined features
# Expected accuracy: 75-90%
```

### Voting Strategy:
```python
# Get predictions from each model
pred_mert = classifier_mert.predict(mert_features)
pred_clap = classifier_clap.predict(clap_features)
pred_panns = classifier_panns.predict(panns_features)

# Majority voting
final_prediction = majority_vote([pred_mert, pred_clap, pred_panns])
```

---

## Compatibility Matrix

| Model | Python 3.10 | NumPy 2.x | PyTorch 2.x | Jetson Orin | M1 Mac |
|-------|-------------|-----------|-------------|-------------|--------|
| VGGish | ✅ | ✅ | ✅ | ✅ | ✅ |
| MERT | ✅ | ✅ | ✅ | ✅ CPU | ✅ MPS |
| CLAP | ✅ | ✅ | ✅ | ✅ | ✅ |
| EnCodec | ✅ | ✅ | ✅ | ✅ | ✅ |
| AST | ✅ | ✅ | ✅ | ✅ | ✅ |
| HuBERT | ✅ | ✅ | ✅ | ✅ CPU | ✅ MPS |
| PANNs | ✅ | ✅ | ✅ | ✅ | ✅ |
| Musicnn | ❌ | ❌ | ❌ | ❌ | ❌ Docker only |

---

## Summary

**Best Overall:** MERT-v1-330M
- Music-specific, state-of-the-art, no conflicts

**Best for Speed:** VGGish
- Fast, small, established baseline

**Best for Innovation:** CLAP
- Zero-shot, audio-text alignment

**Not Recommended:** HuBERT (speech-focused), EnCodec (compression-focused)

**Date**: 2025-11-25
**Status**: All models tested and documented
