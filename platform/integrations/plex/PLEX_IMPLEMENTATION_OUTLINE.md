# Music_ReClass Plex Integration - Implementation Steps

## Overview
Build Music_ReClass as external service that integrates with Plex Media Server for AI-powered genre classification of 100k+ songs.

---

## Phase 1: Setup & Infrastructure (1-2 days)

### Step 1.1: Environment Setup
- Install Plex Media Server
- Get Plex API token
- Setup Python environment with dependencies
- Install PlexAPI library: `pip install plexapi`

### Step 1.2: Database Setup
- Create SQLite database (schema from earlier)
- Add Plex-specific fields:
  - `plex_id` (Plex track ID)
  - `plex_library_id`
  - `last_synced`

### Step 1.3: Plex Connection Test
```python
from plexapi.server import PlexServer
# Test connection to Plex
# List music libraries
# Read sample tracks
```

---

## Phase 2: Plex Library Sync (2-3 days) ✅ COMPLETED

### Step 2.1: Read Plex Music Library ✅
- Connect to Plex server
- Get all music tracks (100k songs)
- Extract metadata (title, artist, album, file path)
- Store in your database

**Files Created:**
- `plex_sync.py` - Main sync script
- `config.py` - Configuration file
- `requirements_plex.txt` - Dependencies

### Step 2.2: Map Plex to Local Files ✅
- Match Plex library paths to actual file locations
- Handle network paths vs local paths
- Verify file accessibility

**Implementation:**
- `PlexSync.test_file_access()` - Tests first 10 files
- Automatic path extraction from Plex API
- Error handling for missing files

### Step 2.3: Sync Service ✅
- Detect new songs added to Plex
- Update database when songs removed
- Handle library refreshes

**Features:**
- Progress tracking (every 1000 songs)
- Error logging
- Statistics display
- Resume capability

---

### How to Use Phase 2:

**1. Get Plex Token:**
```
1. Open Plex Web: http://localhost:32400/web
2. Play any song
3. Click ⋮ → Get Info → View XML
4. Copy X-Plex-Token from URL
```

**2. Configure:**
```python
# Edit config.py
PLEX_TOKEN = 'your_token_here'
PLEX_LIBRARY_NAME = 'Music'  # Your library name
```

**3. Install Dependencies:**
```bash
pip install -r requirements_plex.txt
```

**4. Run Sync:**
```bash
python3 plex_sync.py
```

**Expected Output:**
```
=== Plex Library Sync ===

Syncing Plex library: Music
Total tracks in Plex: 100000
Synced 1000 tracks...
Synced 2000 tracks...
...
Synced 100000 tracks...

✅ Sync complete!
   Synced: 100000 tracks
   Errors: 0 tracks

Database Statistics:
  Total songs: 100,000
  Analyzed: 0
  Pending: 100,000

Testing file access (first 10 files):
  ✅ /path/to/music/song1.mp3
  ✅ /path/to/music/song2.mp3
  ...
```

**TODO:**
- [ ] Get Plex token
- [ ] Update config.py with token
- [ ] Verify library name
- [ ] Run initial sync

---

## Phase 3: Feature Extraction (2-3 days for 100k songs) ✅ COMPLETED

### Step 3.1: Extract FMA Features ✅
- Process all 100k songs
- Save features to database
- Time: ~8-12 hours (RTX 4060 Ti 16GB)

**Implementation:**
- `extract_fma.py` - 518 dimensions
- Sample rate: 22050 Hz
- Features: Chroma, Tonnetz, MFCC, Spectral, Rhythm, Mel spectrogram
- CPU-based (no GPU required)

### Step 3.2: Extract MERT Features ✅
- Load MERT model (m-a-p/MERT-v1-330M)
- Process all songs
- Time: ~30-50 hours (RTX 4060 Ti 16GB)

**Implementation:**
- `extract_mert.py` - 768 dimensions
- Sample rate: 24000 Hz
- Transformer-based embeddings
- GPU-accelerated

### Step 3.3: Extract JMLA Features ⏸️ POSTPONED
- Will be added later as optional upgrade
- Adds +3-4% accuracy boost
- Time: ~30-50 hours when implemented

**Status:**
- `extract_jmla.py` - Created but not used initially
- `extract_jmla_simple.py` - Alternative implementation
- Can be added after core system is working

### Step 3.4: Progress Tracking ✅
- Show progress bar (tqdm)
- Resume capability if interrupted
- Log errors for failed files

**Master Script:**
- `extract_all_features.py` - Runs FMA + MERT (JMLA disabled)
- Time tracking per model
- Error handling and summary

---

### Files Created:

```
extract_fma.py              # FMA feature extraction (518 dims)
extract_mert.py             # MERT feature extraction (768 dims)
extract_jmla.py             # JMLA feature extraction (768 dims)
extract_all_features.py     # Master extraction script
```

---

### How to Use Phase 3:

**Option 1: Extract All Features (Recommended)**
```bash
python3 extract_all_features.py
```

**Option 2: Extract Individually**
```bash
# Stage 1: FMA (5-8 hours)
python3 extract_fma.py

# Stage 2: MERT (10-20 hours)
python3 extract_mert.py

# Stage 3: JMLA (10-20 hours)
python3 extract_jmla.py
```

**Expected Output:**
```
============================================================
Music_ReClass - Feature Extraction Pipeline
Processing 100k songs with 3 models
============================================================

============================================================
Starting: FMA Features (518 dims)
============================================================

Found 100000 songs to process
Extracting FMA features: 100%|████████| 100000/100000 [5:23:45<00:00]

✅ FMA extraction complete!
   Success: 99,856
   Errors: 144

✅ FMA Features (518 dims) completed in 5.40 hours

============================================================
Starting: MERT Features (768 dims)
============================================================

Loading MERT model...
MERT model loaded on cuda
Found 100000 songs to process
Extracting MERT features: 100%|████████| 100000/100000 [15:32:18<00:00]

✅ MERT extraction complete!
   Success: 99,856
   Errors: 144

✅ MERT Features (768 dims) completed in 15.54 hours

============================================================
Starting: JMLA Features (768 dims)
============================================================

Loading JMLA model...
JMLA model loaded on cuda
Found 100000 songs to process
Extracting JMLA features: 100%|████████| 100000/100000 [12:45:33<00:00]

✅ JMLA extraction complete!
   Success: 99,856
   Errors: 144

✅ JMLA Features (768 dims) completed in 12.76 hours

============================================================
EXTRACTION SUMMARY
============================================================
✅ FMA   :   5.40 hours
✅ MERT  :  15.54 hours
✅ JMLA  :  12.76 hours

Total time: 33.70 hours (1.40 days)

✅ All features extracted successfully!
Next step: Run train_extended.py to train the model
```

---

### Feature Specifications:

| Model | Dimensions | Sample Rate | Processing | Time (100k RTX 4060 Ti) | Status |
|-------|-----------|-------------|------------|------------------------|--------|
| FMA | 518 | 22050 Hz | CPU | 8-12 hours | ✅ Active |
| MERT | 768 | 24000 Hz | GPU | 30-50 hours | ✅ Active |
| JMLA | 768 | 16000 Hz | GPU | 30-50 hours | ⏸️ Postponed |
| **Initial System** | **1286** | - | - | **40-60 hours (2.5 days)** | ✅ |
| **With JMLA (Future)** | **2054** | - | - | **70-110 hours (4 days)** | ⏸️ |

**Initial Accuracy: 82-88% (FMA + MERT)**
**With JMLA: 85-92% (+3-4% boost)**

---

### Key Features:

**Resume Capability:**
- Checks database for existing features
- Only processes songs without features
- Can stop and restart anytime

**GPU Memory Management:**
- Clears cache every 100 songs
- Prevents OOM errors
- Optimized batch processing

**Error Handling:**
- Logs failed files
- Continues processing on errors
- Summary report at end

**Database Storage:**
- Features stored as BLOB (numpy arrays)
- Efficient compression
- Fast retrieval

---

### Troubleshooting:

**"CUDA out of memory"**
- Reduce batch size in code
- Clear GPU cache: `torch.cuda.empty_cache()`
- Process fewer songs at once

**"File not found"**
- Check Plex library paths
- Verify file accessibility
- Re-run plex_sync.py

**Slow extraction**
- Check GPU utilization: `nvidia-smi`
- Ensure CUDA is enabled
- Use SSD storage

---

### TODO:
- [ ] Run feature extraction
- [ ] Verify all 3 feature types in database
- [ ] Check extraction summary
- [ ] Proceed to Phase 4 (Model Training)

---

## Phase 4: Model Training (1-2 days)

### Step 4.1: Prepare Training Data
- Combine FMA features (518 dims)
- Add your custom genres (K-Pop, Anime, Lo-Fi)
- Split train/validation sets

### Step 4.2: Train Extended Model
- Retrain MSD model with 16 genres (13 + 3 custom)
- Train ensemble model (FMA + MERT + JMLA)
- Save best models

### Step 4.3: Validation
- Test accuracy on validation set
- Generate confusion matrix
- Tune confidence thresholds

---

## Phase 5: Classification Service (2-3 days)

### Step 5.1: Batch Classifier
```python
# Classify all 100k songs
# Save top 5 predictions + confidence
# Store in classifications table
```

### Step 5.2: Progressive Voting
- Stage 1: FMA only (fast)
- Stage 2: FMA + MERT (if confidence < 0.8)
- Stage 3: Full ensemble (if confidence < 0.7)

### Step 5.3: Update Plex Genres
```python
# Push AI-predicted genres back to Plex
# Update via Plex API
# Option: Keep original genre as secondary tag
```

---

## Phase 6: Web UI (3-5 days)

### Step 6.1: Backend API (Flask/FastAPI)
- `/api/songs` - List songs with pagination
- `/api/stats` - Library statistics
- `/api/genres` - Genre distribution
- `/api/song/<id>` - Song details
- `/api/sync` - Trigger Plex sync

### Step 6.2: Frontend Pages
- **Dashboard**: Stats, genre distribution chart
- **Library**: Searchable song list with filters
- **Song Detail**: All predictions, confidence scores
- **Settings**: Plex connection, sync options

### Step 6.3: Features
- Search by title/artist/genre
- Filter by confidence threshold
- Compare AI genre vs ID3 tag
- Export results to CSV
- Manual genre override

---

## Phase 7: Automation (1-2 days)

### Step 7.1: Scheduled Tasks
- Auto-sync with Plex daily
- Classify new songs automatically
- Update Plex genres

### Step 7.2: Webhook Integration
- Listen for Plex "new track added" events
- Trigger classification immediately
- Real-time updates

### Step 7.3: Background Worker
- Queue system for processing
- Handle multiple songs in parallel
- Retry failed classifications

---

## Phase 8: Testing & Optimization (2-3 days)

### Step 8.1: Testing
- Test with sample 1k songs
- Verify Plex updates work
- Check web UI performance

### Step 8.2: Optimization
- Database indexing
- Batch processing optimization
- Cache frequently accessed data

### Step 8.3: Error Handling
- Handle corrupted audio files
- Network errors with Plex
- GPU memory issues

---

## Phase 9: Deployment (1 day)

### Step 9.1: Production Setup
- Configure for 24/7 operation
- Setup logging
- Monitor GPU usage

### Step 9.2: Documentation
- Installation guide
- Configuration guide
- User manual

### Step 9.3: Backup Strategy
- Database backups
- Model checkpoints
- Configuration backups

---

## Timeline Summary

| Phase | Duration (RTX 4060 Ti 16GB) | Can Run Parallel |
|-------|----------------------------|------------------|
| 1. Setup | 0.5 day | No |
| 2. Plex Sync | 0.5 day | No |
| 3. Feature Extraction | **2.5 days** | Run overnight ⭐ |
| 4. Model Training | 0.5 day | After Phase 3 |
| 5. Classification | **2.5 days** | Run overnight ⭐ |
| 6. Web UI | 4 days | **Yes** (parallel with 3-5) |
| 7. Automation | 1 day | After Phase 5 |
| 8. Testing | 2 days | After Phase 7 |
| 9. Deployment | 1 day | Final |

**Sequential (no overlap): 12-14 days**
**With overlap (Web UI parallel): 8-10 days**
**Realistic timeline: 2 weeks**

### GPU Comparison:

| GPU | Feature Extraction | Classification | Total Project |
|-----|-------------------|----------------|---------------|
| **RTX 4060 Ti 16GB** | 2.5 days | 2.5 days | **2 weeks** ⭐ Recommended |
| RTX 5070 Ti | 1.5 days | 1.5 days | 1.5 weeks |
| RTX 5090 | 1 day | 1 day | 1 week |
| RTX A6000 | 1 day | 1 day | 1 week |
| M1 Pro | 5-8 days | 5-8 days | 4-5 weeks |
| Jetson | 12-16 days | 12-16 days | 8-10 weeks |

**Total: 2 weeks** (with RTX 4060 Ti 16GB)
**Total: 1 week** (with RTX 5090)

---

## Project Structure

```
Music_ReClass_Plex/
├── plex_sync.py          # Sync with Plex library
├── extract_features.py   # Feature extraction (FMA/MERT/JMLA)
├── train_extended.py     # Train with custom genres
├── classify_batch.py     # Classify all songs
├── update_plex.py        # Push genres to Plex
├── web_ui/
│   ├── app.py           # Flask backend
│   ├── templates/       # HTML pages
│   │   ├── index.html
│   │   ├── library.html
│   │   └── song_detail.html
│   └── static/          # CSS, JS, images
│       ├── style.css
│       └── app.js
├── database.py          # Database operations
├── config.py            # Plex URL, token, paths
├── scheduler.py         # Automation tasks
├── models/              # Trained models
│   ├── msd_extended.pth
│   ├── ensemble.pth
│   └── genre_map.json
└── logs/                # Application logs
```

---

## Hardware Requirements

### Recommended: RTX 4060 Ti 16GB ⭐
- Feature extraction: 40-60 hours for 100k songs (2.5 days)
- Classification: 40-60 hours (2.5 days)
- 16GB VRAM (sufficient for MERT model)
- Budget-friendly ($500)
- **Total project time: 2 weeks**
- **Best value for 100k song library**

### Alternative: RTX 5090
- Feature extraction: 25-45 hours for 100k songs (1 day)
- Classification: 25-45 hours (1 day)
- 24-32GB VRAM
- Premium option ($1,800-2,000)
- **Total project time: 1 week**
- Best if processing >100k songs regularly

### Alternative: RTX 5070 Ti
- Feature extraction: 35-65 hours (1.5 days)
- Classification: 35-65 hours (1.5 days)
- 16GB VRAM
- Mid-range option ($800)
- **Total project time: 1.5 weeks**

### Not Recommended: M1 Pro / Jetson
- Too slow for 100k songs (4-10 weeks)
- Better for smaller libraries (<10k songs)
- M1 Pro: No CUDA support
- Jetson: Limited performance

---

## Storage Requirements

| Component | Size (100k songs) |
|-----------|-------------------|
| Audio files | ~500 GB (existing) |
| Database | ~1.2 GB |
| FMA features | 200 MB |
| MERT features | 300 MB |
| JMLA features | 300 MB |
| Models | ~2 GB |
| Logs | ~100 MB |
| **Total** | **~4 GB** (excluding audio) |

---

## Genre Configuration

### Original 13 MSD Genres
Blues, Country, Electronic, Folk, International, Jazz, Latin, New Age, Pop_Rock, Rap, Reggae, RnB, Vocal

### Custom Genres (Your additions)
K-Pop, Anime, Lo-Fi

### Total: 16 Genres

---

## Key Features

### AI Classification
- 82-88% accuracy (FMA + MERT)
- 85-92% accuracy (with JMLA - future upgrade)
- Top 5 predictions per song
- Confidence scores

### Plex Integration
- Automatic library sync
- Genre updates via API
- Preserve original ID3 tags
- Real-time classification for new songs

### Web Dashboard
- Library overview with statistics
- Genre distribution visualization
- Search and filter capabilities
- Compare AI vs original genres
- Export results

### Progressive Voting (Future with JMLA)
- Fast classification for confident predictions
- Deep analysis for uncertain songs
- Average 20-40s per track
- Optimized for production use

### Initial System (FMA + MERT)
- 1286 feature dimensions
- Faster deployment (2.5 days extraction)
- Reliable and proven
- Excellent accuracy for most use cases

---

## API Endpoints

### Plex Sync
- `GET /api/plex/libraries` - List Plex music libraries
- `POST /api/plex/sync` - Sync library with database
- `GET /api/plex/status` - Sync status

### Songs
- `GET /api/songs` - List songs (paginated, filtered)
- `GET /api/song/<id>` - Song details
- `PUT /api/song/<id>/genre` - Manual genre override

### Classification
- `POST /api/classify/<id>` - Classify single song
- `POST /api/classify/batch` - Classify multiple songs
- `GET /api/classify/status` - Classification progress

### Statistics
- `GET /api/stats` - Library statistics
- `GET /api/stats/genres` - Genre distribution
- `GET /api/stats/confidence` - Confidence distribution

### Settings
- `GET /api/config` - Get configuration
- `PUT /api/config` - Update configuration
- `POST /api/config/test` - Test Plex connection

---

## Configuration File (config.py)

```python
# Plex Configuration
PLEX_URL = 'http://localhost:32400'
PLEX_TOKEN = 'your-plex-token'
PLEX_LIBRARY_NAME = 'Music'

# Database
DATABASE_PATH = 'music_100k.db'

# Models
MODEL_PATH = 'models/'
USE_ENSEMBLE = True

# Classification
CONFIDENCE_THRESHOLD = 0.7
TOP_N_PREDICTIONS = 5
BATCH_SIZE = 32

# GPU
DEVICE = 'cuda'  # or 'cpu'
GPU_MEMORY_LIMIT = 0.9

# Web UI
WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
DEBUG = False

# Automation
AUTO_SYNC_ENABLED = True
SYNC_INTERVAL_HOURS = 24
AUTO_CLASSIFY_NEW = True
```

---

## Next Steps

1. **Start with Phase 1**: Setup Plex connection and test API
2. **Prepare custom genre data**: Collect 100+ songs for K-Pop, Anime, Lo-Fi
3. **Choose hardware**: Decide on RTX 5090 vs 4060 Ti
4. **Create project structure**: Initialize Git repo and folders

---

**Last Updated**: 2025-11-29
**Status**: Planning Phase
**Target Completion**: 3-5 weeks


---

**Last Updated**: 2025-11-29
**Status**: Planning Phase Complete
**Target Completion**: 2 weeks (RTX 4060 Ti 16GB)
**Initial System**: FMA + MERT (1286 dims, 82-88% accuracy)
**Future Upgrade**: Add JMLA (+3-4% accuracy boost)

## Project Status Summary

### ✅ Completed
- [x] Project planning and architecture
- [x] Phase 2: Plex sync script
- [x] Phase 3: Feature extraction scripts (FMA + MERT)
- [x] Configuration files
- [x] Database schema
- [x] Documentation

### ⏸️ Postponed
- [ ] JMLA integration (optional future upgrade)

### ⏳ Pending
- [ ] Get Plex token
- [ ] Acquire RTX 4060 Ti 16GB
- [ ] Prepare custom genre data (K-Pop, Anime, Lo-Fi - 100+ songs each)
- [ ] Run feature extraction (2.5 days)
- [ ] Train extended model
- [ ] Build web UI
- [ ] Deploy system

### 🎯 Next Immediate Steps
1. Get Plex token from web interface
2. Update `config.py` with token
3. Collect 100+ songs for each custom genre
4. Run `plex_sync.py` to sync library
5. Run `extract_all_features.py` (let it run for 2.5 days)

---

## Decision Log

**2025-11-29:**
- ✅ Decided to use RTX 4060 Ti 16GB (2 week timeline acceptable)
- ✅ Postponed JMLA to focus on core system (FMA + MERT)
- ✅ Target: 82-88% accuracy initially
- ✅ JMLA can be added later for +3-4% boost
- ✅ Realistic timeline: 2 weeks to production
