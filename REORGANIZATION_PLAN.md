# Deployment Reorganization Plan

## New Structure

```
Music_ReClass/
├── src/                          # Source code
│   ├── extractors/              # Feature extraction
│   ├── models/                  # Model architectures
│   ├── training/                # Training scripts
│   └── inference/               # Classification/prediction
│
├── config/                       # Configuration files
│   ├── model_configs/           # Model parameters (from json/)
│   └── deployment_configs/      # Deployment settings
│
├── scripts/                      # Utility scripts
│   ├── setup/                   # Setup scripts (startup.sh, etc.)
│   └── maintenance/             # Maintenance scripts
│
├── models/                       # Trained model weights
│   └── .gitkeep
│
├── data/                         # Data directory
│   ├── features/                # Extracted features
│   └── .gitkeep
│
├── tests/                        # Test files
│   └── .gitkeep
│
├── docs/                         # Documentation (keep current)
│   ├── deployment/              # Deployment guides
│   ├── api/                     # API documentation
│   └── ...
│
├── platform/                     # Platform-specific code
│   ├── rtx/                     # RTX-specific
│   ├── jetson/                  # Jetson-specific
│   └── m1/                      # M1-specific
│
├── logs/                         # Keep as is
├── requirements.txt             # Main dependencies
├── setup.py                     # Package setup
├── README.md                    # Main readme
└── .env.example                 # Environment template
```

## Migration Steps

1. Create new directory structure
2. Move extractors/ → src/extractors/
3. Move training/ → src/training/
4. Move classification/ → src/inference/
5. Move json/ → config/model_configs/
6. Move RTX/, MBP/ → platform/
7. Move startup.sh, shutdown.sh → scripts/setup/
8. Move features/ → data/features/
9. Create requirements.txt from platform-specific ones
10. Create setup.py for package installation
11. Update imports in all files
12. Create deployment documentation

## Files to Archive

- KeyFile/ (business plan - move to docs/archive/)
- Plex/ (separate project - move to platform/integrations/)
- DAILY_*.md (move to logs/)

## New Files to Create

- setup.py
- requirements.txt
- .env.example
- src/__init__.py
- API endpoint (if needed)
- Docker configuration (optional)
