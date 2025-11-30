# Project References

## External Resources

### Kaggle Notebooks

1. **Genre Classification with FMA Data**
   - URL: https://www.kaggle.com/code/jojothepizza/genre-classification-with-fma-data
   - Topic: Music genre classification using FMA dataset
   - Added: 2025-11-23

## Datasets

1. **GTZAN Dataset**
   - Location: `/media/mijesu_970/SSD_Data/DataSets/GTZAN/`
   - Size: 1,000 tracks, 10 genres
   - Status: Downloaded

2. **FMA (Free Music Archive)**
   - Location: `/media/mijesu_970/SSD_Data/DataSets/FMA/`
   - Size: 25,000 tracks (Medium), 16 genres
   - Status: Downloaded
   - Official: https://github.com/mdeff/fma

3. **MagnaTagATune**
   - Size: 25,863 clips, 188 tags
   - Status: Planned

4. **Million Song Dataset (MSD)**
   - Size: 1M songs metadata
   - Status: Planned

## Models

1. **OpenJMLA**
   - Location: `/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/`
   - Type: Vision Transformer for audio
   - Parameters: 86M
   - Status: Downloaded

## Alternative Pre-trained Audio Models

### 1. OpenJMLA ✓ (Current)
- **Type:** General audio feature extractor
- **Architecture:** Vision Transformer (ViT)
- **Parameters:** 86M
- **Model Size:** ~330MB (epoch_20) + 1.3GB (main model)
- **Use case:** Audio representation learning
- **Status:** Downloaded and configured

### 2. Musicnn
- **Type:** Music tagging model
- **Architecture:** CNN for music
- **Parameters:** ~2M
- **Model Size:** ~8MB
- **Use case:** Music feature extraction and tag prediction
- **Source:** Music Technology Group (MTG)
- **Repository:** https://github.com/jordipons/musicnn

### 3. VGGish
- **Type:** Google's audio embedding model
- **Architecture:** VGG-based CNN
- **Parameters:** ~70M
- **Model Size:** ~280MB
- **Use case:** General audio embeddings (128-dim)
- **Source:** Google Research
- **Repository:** https://github.com/tensorflow/models/tree/master/research/audioset/vggish

### 4. CLMR (Contrastive Learning of Musical Representations)
- **Type:** Self-supervised music representation
- **Architecture:** Contrastive learning framework
- **Parameters:** ~50M
- **Model Size:** ~200MB
- **Use case:** Music feature learning without labels
- **Source:** Research paper implementation
- **Repository:** https://github.com/Spijkervet/CLMR

### 5. Jukebox
- **Type:** OpenAI's music generation model
- **Architecture:** Large-scale Transformer
- **Parameters:** 1.2B (small), 5B (large)
- **Model Size:** ~5GB (small), ~20GB (large)
- **Use case:** Music generation and understanding
- **Source:** OpenAI
- **Repository:** https://github.com/openai/jukebox
- **Note:** Very large model, requires significant compute

## Documentation

- Project Info: `music_project_info.md`
- Project History: `PROJECT_HISTORY.md`
- Presentation: `PROJECT_PRESENTATION.md`

---

*Last Updated: 2025-11-23*
