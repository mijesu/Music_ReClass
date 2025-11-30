# Feature Extractors Record

## Downloaded Models (7 Total)

### 1. VGGish
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/VGGish/`
- **Size**: 276 MB
- **Output**: 128-dimensional embeddings
- **Type**: CNN-based, general audio tagging
- **Source**: Google Research, trained on AudioSet
- **Usage**: `from torchvggish import vggish; model = vggish()`
- **Training Dataset**: AudioSet (632 audio event classes, not genre-specific)
- **Genre Coverage**: General audio events including music, speech, environmental sounds
  - Music-related classes: Musical instrument, Music, Singing, etc.
  - Not trained specifically for music genre classification

### 2. MERT-v1-330M
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/MERT/MERT-v1-330M/`
- **Size**: 1.2 GB (pytorch_model.pth)
- **Output**: 768-dimensional embeddings
- **Type**: Transformer-based, music-specific
- **Source**: Hugging Face `m-a-p/MERT-v1-330M`
- **Usage**: `AutoModel.from_pretrained("m-a-p/MERT-v1-330M")`
- **Note**: Largest available MERT model (95M version deleted to save space)
- **Training Dataset**: Self-supervised on large-scale music audio (160K hours)
- **Genre Coverage**: Not genre-labeled, but trained on diverse music
  - Pre-training: Masked acoustic modeling on unlabeled music
  - Covers multiple genres implicitly through diverse training data
  - Designed for transfer learning to any music understanding task

### 3. CLAP (larger_clap_music)
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/CLAP/larger_clap_music/`
- **Size**: 741 MB
- **Output**: 512-dimensional embeddings
- **Type**: Contrastive audio-text model
- **Source**: LAION `laion/larger_clap_music`
- **Usage**: Audio-text alignment, semantic search
- **Specialty**: Music-focused variant
- **Training Dataset**: Music-specific subset from LAION-Audio-630K
- **Genre Coverage**: Text-based genre understanding through captions
  - Trained on music audio with text descriptions
  - Can understand genre through text prompts (e.g., "rock music", "classical piano")
  - Implicit genre knowledge from audio-text pairs

### 4. EnCodec (24khz)
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/EnCodec/encodec_24khz/`
- **Size**: 89 MB
- **Output**: Codebook embeddings (variable dimensions)
- **Type**: Neural audio codec
- **Source**: Meta `facebook/encodec_24khz`
- **Usage**: Compression-based features, reconstruction
- **Note**: Less suitable for classification, more for generation
- **Training Dataset**: General audio (speech, music, environmental sounds)
- **Genre Coverage**: Not genre-aware
  - Trained for audio compression/reconstruction
  - Learns low-level acoustic features, not semantic genre information
  - Embeddings represent audio structure, not musical style

### 5. AST (Audio Spectrogram Transformer)
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/AST/ast-finetuned-audioset/`
- **Size**: 331 MB
- **Output**: 768-dimensional embeddings
- **Type**: Vision Transformer adapted for audio spectrograms
- **Source**: MIT `MIT/ast-finetuned-audioset-10-10-0.4593`
- **Usage**: Transformer-based audio understanding
- **Training**: Fine-tuned on AudioSet
- **Training Dataset**: AudioSet (632 audio event classes)
- **Genre Coverage**: Similar to VGGish, general audio events
  - Music-related classes include instruments and music types
  - Not specifically trained for genre classification
  - Better at identifying instruments and playing techniques than genres

### 6. HuBERT (Large)
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/HuBERT/hubert-large-ll60k/`
- **Size**: 1.2 GB
- **Output**: 1024-dimensional embeddings
- **Type**: Self-supervised audio representation
- **Source**: Facebook `facebook/hubert-large-ll60k`
- **Usage**: Similar to Wav2Vec2, hidden-unit BERT
- **Training**: 60k hours of Libri-Light dataset
- **Training Dataset**: Libri-Light (60K hours of English audiobooks/speech)
- **Genre Coverage**: Not applicable - trained on speech, not music
  - Primarily for speech understanding tasks
  - May capture some acoustic patterns useful for music
  - Not recommended as primary feature extractor for music genre classification
  - Better suited for speech-related audio tasks

### 7. PANNs CNN14
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/PANNs/cnn14/`
- **Size**: 331 MB
- **Output**: 2048-dimensional embeddings
- **Type**: Pre-trained Audio Neural Networks (CNN)
- **Source**: MIT (same as AST model)
- **Usage**: Large-scale audio tagging
- **Note**: Highest dimensional output among all models
- **Training Dataset**: AudioSet (527 audio event classes)
- **Genre Coverage**: General audio events, some music-related
  - Music classes: Musical instrument, Music genre (limited), Singing
  - Instrument classes: Piano, Guitar, Drum, Violin, etc.
  - Not primarily designed for genre classification
  - Strong at instrument recognition and audio event detection

---

## Available But Not Downloaded

### BEATs (Microsoft)
- **Size**: ~400 MB
- **Output**: 768-dimensional embeddings
- **Type**: Iterative self-supervised learning
- **Source**: Microsoft Research
- **Reason Not Downloaded**: Insufficient disk space (626 MB free)
- **GitHub**: https://github.com/microsoft/unilm/tree/master/beats

### AudioMAE
- **Size**: ~300-600 MB
- **Output**: Variable dimensions
- **Type**: Masked Autoencoder for audio (ViT-based)
- **Source**: Research implementation
- **Reason Not Downloaded**: Insufficient disk space
- **Specialty**: Good for music understanding

### Wav2Vec2 (Large)
- **Size**: ~1.2 GB
- **Output**: 1024-dimensional embeddings
- **Type**: Self-supervised speech/audio representation
- **Source**: Facebook `facebook/wav2vec2-large`
- **Reason Not Downloaded**: Insufficient disk space
- **Note**: Similar to HuBERT (already downloaded)

---

## Library-Based Extractors (No Model Download Required)

### Musicnn ⚠️ DEPENDENCY CONFLICTS
- **Installation**: `pip install musicnn` (NOT RECOMMENDED)
- **Size**: ~20 MB
- **Output**: Timbral and temporal features
- **Type**: CNN trained on MagnaTagATune
- **Usage**: Pre-trained weights included in package
- **Note**: Example notebook already available in project
- **Training Dataset**: MagnaTagATune (MTT) - 25,863 audio clips
- **Genre Coverage**: 29 music genres (multi-label)
  - **Genres**: rock, pop, alternative, indie, electronic, folk, heavy metal, punk, country, classic rock, 
    alternative rock, jazz, beautiful, dance, soul, electronica, blues, female vocalists, chillout, 
    experimental, hip hop, instrumental, psychedelic, reggae, ambient, hard rock, metal, world, oldies
  - Trained specifically for music tagging and genre recognition
  - Best suited for music genre classification among all models
- **⚠️ CONFLICTS**: 
  - Requires TensorFlow >=1.14 + numpy<1.17 (incompatible with Python 3.10+)
  - Cannot install in current environment due to NumPy version conflicts
  - Last updated: 2025-11-19 (still uses old dependencies)
  - PyTorch forks exist but also have old dependencies
  - **Workaround**: Use Docker container or Python 3.7 virtual environment
  - **Alternative**: Use MERT-v1-330M for music-specific features instead

### Essentia
- **Installation**: `pip install essentia`
- **Size**: Library only
- **Output**: 100+ audio features (MFCCs, spectral, rhythm, etc.)
- **Type**: Traditional audio analysis
- **Usage**: Feature extraction on-the-fly, no pre-trained model
- **Specialty**: Comprehensive music information retrieval

### librosa
- **Installation**: `pip install librosa`
- **Size**: Library only
- **Output**: Various features (chroma, spectral contrast, tonnetz, MFCCs)
- **Type**: Traditional audio analysis
- **Usage**: Python library for music/audio analysis
- **Note**: Most commonly used for audio preprocessing

### openSMILE
- **Installation**: `pip install opensmile`
- **Size**: Library only
- **Output**: Speech and music features
- **Type**: Traditional feature extraction
- **Usage**: Configurable feature sets
- **Specialty**: Speech emotion recognition, music analysis

---

## Model Comparison Summary

| Model | Size | Output Dims | Type | Best For |
|-------|------|-------------|------|----------|
| VGGish | 276 MB | 128 | CNN | General audio |
| MERT-330M | 1.2 GB | 768 | Transformer | Music understanding |
| CLAP | 741 MB | 512 | Contrastive | Audio-text alignment |
| EnCodec | 89 MB | Variable | Codec | Compression/generation |
| AST | 331 MB | 768 | ViT | Audio spectrograms |
| HuBERT | 1.2 GB | 1024 | Self-supervised | Raw audio features |
| PANNs | 331 MB | 2048 | CNN | Audio tagging |
| Musicnn ⚠️ | ~20 MB | Variable | CNN | Music tagging (CONFLICTS) |

---

## Storage Information

- **Total Model Storage**: ~4 GB
- **Location**: `/media/mijesu_970/SSD_Data/AI_models/`
- **Disk Usage**: 53 GB / 57 GB (99% full)
- **Available Space**: 626 MB
- **Deleted**: MERT-v1-95M (1.3 GB), Fairseq .pt files (3.8 GB)

---

## Next Steps

1. **Install library-based extractors**: Musicnn, Essentia, librosa (no disk space needed)
2. **Create comparison script**: Extract features from same audio samples using all models
3. **Evaluate performance**: Compare feature quality for genre classification
4. **Free space if needed**: To download BEATs, AudioMAE, or Wav2Vec2

---

## Notes

- All `.bin` files renamed to `.pth` for consistency
- Cache directories cleaned to save space
- MERT-v1-330M is the largest available MERT model (no 1B version exists)
- PANNs CNN14 has highest dimensional output (2048-dim)
- EnCodec less suitable for classification tasks (designed for compression)
- HuBERT and Wav2Vec2 are similar architectures (HuBERT already downloaded)

---

## Training Dataset Genre Coverage Summary

| Model | Training Dataset | Genre-Specific | Music Focus | Best For Genre Classification |
|-------|------------------|----------------|-------------|-------------------------------|
| **Musicnn ⚠️** | MagnaTagATune (29 genres) | ✅ Yes | ✅ Music only | ⭐⭐⭐⭐⭐ Excellent (CONFLICTS) |
| **MERT** | 160K hrs music (unlabeled) | ❌ No | ✅ Music only | ⭐⭐⭐⭐ Very Good |
| **CLAP** | Music + text pairs | ⚠️ Implicit | ✅ Music variant | ⭐⭐⭐ Good |
| **VGGish** | AudioSet (632 classes) | ❌ No | ⚠️ Mixed audio | ⭐⭐ Fair |
| **AST** | AudioSet (632 classes) | ❌ No | ⚠️ Mixed audio | ⭐⭐ Fair |
| **PANNs** | AudioSet (527 classes) | ❌ No | ⚠️ Mixed audio | ⭐⭐ Fair |
| **HuBERT** | Libri-Light (speech) | ❌ No | ❌ Speech only | ⭐ Poor |
| **EnCodec** | General audio | ❌ No | ⚠️ Mixed audio | ⭐ Poor |

### Recommendations by Use Case:

**For Music Genre Classification (Priority Order):**
1. **MERT** - Music-specific transformer, best for transfer learning (RECOMMENDED)
2. **CLAP** - Music variant with semantic understanding
3. **Musicnn** - ⚠️ Explicitly trained on music genres (DEPENDENCY CONFLICTS - use Docker/Python 3.7)
4. **VGGish/AST/PANNs** - General audio, may need fine-tuning
5. **HuBERT/EnCodec** - Not recommended for music genres

**For Feature Extraction Comparison:**
- Test all working models to compare performance
- Combine features from multiple models (ensemble approach)
- Use MERT as primary for music-specific features (replaces Musicnn due to conflicts)
- Use CLAP for semantic/text-based understanding

**Date Created**: 2025-11-25
**Last Updated**: 2025-11-25
