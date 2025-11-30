"""Configuration for Music_ReClass Plex Integration"""

# ===== PLEX CONFIGURATION =====
PLEX_URL = 'http://localhost:32400'
PLEX_TOKEN = 'YOUR_PLEX_TOKEN_HERE'  # TODO: Get from Plex Web
PLEX_LIBRARY_NAME = 'Music'

# ===== DATABASE =====
DATABASE_PATH = 'music_100k.db'

# ===== MODELS =====
MODEL_PATH = 'models/'
USE_ENSEMBLE = True

# Model files
FMA_MODEL = MODEL_PATH + 'msd_model.pth'
MERT_MODEL = MODEL_PATH + 'mert_model.pth'
JMLA_MODEL = MODEL_PATH + 'jmla_model.pth'
ENSEMBLE_MODEL = MODEL_PATH + 'ensemble_model.pth'

# ===== GENRES =====
GENRES = [
    # Original 13 MSD genres
    'Blues', 'Country', 'Electronic', 'Folk', 'International',
    'Jazz', 'Latin', 'New Age', 'Pop_Rock', 'Rap', 'Reggae', 'RnB', 'Vocal',
    # Custom genres
    'K-Pop', 'Anime', 'Lo-Fi'
]

NUM_GENRES = len(GENRES)

# ===== CLASSIFICATION =====
CONFIDENCE_THRESHOLD = 0.7
TOP_N_PREDICTIONS = 5
BATCH_SIZE = 32

# Progressive voting thresholds
STAGE_1_THRESHOLD = 0.85  # Use FMA only if confidence > 0.85
STAGE_2_THRESHOLD = 0.75  # Use FMA+MERT if confidence > 0.75
# Otherwise use full ensemble

# ===== GPU =====
DEVICE = 'cuda'  # 'cuda' or 'cpu'
GPU_MEMORY_LIMIT = 0.9

# ===== FEATURE EXTRACTION =====
SAMPLE_RATE = 22050
DURATION = 30  # seconds

# Feature dimensions
FMA_DIMS = 518
MERT_DIMS = 768
JMLA_DIMS = 768
COMBINED_DIMS = FMA_DIMS + MERT_DIMS + JMLA_DIMS  # 2054

# ===== WEB UI =====
WEB_HOST = '0.0.0.0'
WEB_PORT = 5000
DEBUG = False

# ===== AUTOMATION =====
AUTO_SYNC_ENABLED = True
SYNC_INTERVAL_HOURS = 24
AUTO_CLASSIFY_NEW = True

# ===== LOGGING =====
LOG_PATH = 'logs/'
LOG_LEVEL = 'INFO'  # DEBUG, INFO, WARNING, ERROR

# ===== PATHS =====
TEMP_PATH = 'temp/'
EXPORT_PATH = 'exports/'
