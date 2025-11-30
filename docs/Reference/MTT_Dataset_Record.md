# MagnaTagATune (MTT) Dataset Processing Record

**Date:** November 25, 2025  
**Time:** 05:30 - 07:41 (GMT+8)

---

## Dataset Information

### MagnaTagATune Dataset
- **Total clips:** 25,863 audio clips (29 seconds each)
- **Tags:** 188 multi-label tags
- **Size:** ~50 GB (audio files)
- **Location:** `/media/mijesu_970/SSD_Data/DataSets/MTT/`

### Tag Categories
- **Genres (29):** rock, pop, jazz, classical, blues, country, folk, reggae, metal, punk, disco, techno, trance, hip hop, rap, funk, electronic, electronica, ambient, new age, opera, world, hard rock, soft rock, heavy metal, industrial, dance, house, jungle
- **Instruments:** guitar, piano, drums, violin, trumpet, flute, synthesizer, etc.
- **Vocals:** male voice, female voice, singing, opera, choir, etc.
- **Moods:** happy, sad, calm, dark, mellow, upbeat, etc.
- **Characteristics:** fast, slow, loud, quiet, acoustic, electric, etc.

---

## Processing Steps

### 1. Feature Extraction from Echonest XML
**Script:** `utils/convert_mtt_features.py`

**Input:** 25,837 XML files in `/DataSets/MTT/Data/Echonest_xml/`

**Features Extracted (36 dimensions):**
- Duration, loudness, tempo, tempo confidence
- Time signature, key, mode (with confidence scores)
- Timbre mean/std (12 dimensions each)
- Pitch mean/std (12 dimensions each)
- Loudness statistics

**Output:**
- `MTT.npy` - 3.55 MB (NumPy array)
- `MTT.pth` - 5.33 MB (PyTorch tensor with metadata)

### 2. Annotation Integration
**Script:** `utils/combine_mtt_annotations.py`

**Input:** 
- Features: `MTT.npy` / `MTT.pth`
- Annotations: `annotations_final.csv` (25,863 clips × 188 tags)

**Output:**
- `MTT_combined.pth` - 26 MB (PyTorch format)
- `MTT_combined.npz` - 2.92 MB (NumPy compressed format)

**Contents:**
- features: (25837, 36)
- tags: (25863, 188)
- tag_names: 188 tag labels
- clip_ids: audio clip identifiers

---

## File Locations

### Features
```
/media/mijesu_970/SSD_Data/AI_models/MTT/
├── MTT.npy                  # 3.55 MB - Features only
├── MTT.pth                  # 5.33 MB - Features with metadata
├── MTT_combined.npz         # 2.92 MB - Features + tags (compressed)
└── MTT_combined.pth         # 26 MB - Features + tags + full metadata
```

### Scripts
```
/media/mijesu_970/SSD_Data/Python/Music_Reclass/utils/
├── convert_mtt_features.py      # Extract features from XML
├── combine_mtt_annotations.py   # Combine with annotations (PTH)
└── combine_mtt_npy.py          # Combine with annotations (NPZ)
```

---

## Alternative Feature Extractor: musicnn

**Reference:** `Reference/musicnn_example.ipynb`

**Model:** MTT_musicnn (pre-trained on MagnaTagATune)

**Usage:**
```python
from musicnn.extractor import extractor
taggram, tags, features = extractor(file_name, model='MTT_musicnn', extract_features=True)
```

**Features Available:**
- `timbral` (408) - Timbral characteristics
- `temporal` (153) - Temporal patterns
- `cnn1`, `cnn2`, `cnn3` - Mid-level CNN features
- `mean_pool`, `max_pool` - Pooled features
- `penultimate` - Final layer features
- `taggram` - Tag predictions over time

**Architecture:**
1. Front-end: Musically motivated CNN
2. Mid-end: Dense layers with residual connections
3. Back-end: Temporal pooling for variable-length inputs

---

## Dataset Comparison

| Dataset | Clips | Genres | Type | Features | Size |
|---------|-------|--------|------|----------|------|
| GTZAN | 1,000 | 10 | Single-label | Mel-spec | ~1.2 GB |
| FMA Medium | 25,000 | 16 | Single-label | 518 pre-computed | ~22 GB |
| MSD | 10,000 | 13 | Single-label | Echonest | ~2.6 GB |
| MTT | 25,863 | 29 genres + 159 other tags | Multi-label | 36 Echonest | ~50 GB |

---

## Next Steps

1. Train multi-label classifier using MTT_combined.pth/npz
2. Compare Echonest features vs musicnn features
3. Evaluate on MTT test set
4. Apply to Music_TBC classification

---

**Status:** ✅ Complete - Dataset ready for training
