# Music Reclassification Project - Workflow Flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│                         START PROJECT                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: QUICK BASELINE                       │
│                         (5 minutes)                              │
├─────────────────────────────────────────────────────────────────┤
│  Script: training/quick_baseline.py                             │
│  • XGBoost on FMA pre-computed features                         │
│  • No GPU required                                              │
│  • Expected: 50-55% accuracy                                    │
│  • Output: baseline_results.json                                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PHASE 2: DATA ANALYSIS                          │
│                        (10 minutes)                              │
├─────────────────────────────────────────────────────────────────┤
│  Script: analysis/analyze_data.py                               │
│  • Genre distribution visualization                             │
│  • Mel-spectrogram comparison                                   │
│  • Class imbalance check                                        │
│  • Output: plots + recommendations                              │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    ┌───────┴───────┐
                    │  GPU Check    │
                    └───┬───────┬───┘
                        │       │
                   GPU  │       │  No GPU
                        │       │
                        ▼       ▼
        ┌───────────────────────────────────┐
        │  Script: utils/gpu_monitor.py     │
        │  • Check available memory         │
        │  • Get batch size recommendation  │
        └───────────────┬───────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 3: PRODUCTION TRAINING                        │
│                      (45 minutes)                                │
├─────────────────────────────────────────────────────────────────┤
│  Script: training/train_gtzan_v2.py ⭐ RECOMMENDED              │
│  • OpenJMLA transfer learning (86M frozen params)               │
│  • Data augmentation enabled                                    │
│  • Validation loop with metrics                                 │
│  • Expected: 70-80% accuracy                                    │
│  • Output: best_model.pth + confusion_matrix.png                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATE RESULTS                              │
├─────────────────────────────────────────────────────────────────┤
│  • Review confusion matrix                                      │
│  • Check classification report                                  │
│  • Identify problem genres                                      │
│  • Compare with baseline                                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────┴───────┐
                    │  Satisfied?   │
                    └───┬───────┬───┘
                        │       │
                   Yes  │       │  No
                        │       │
                        │       ▼
                        │   ┌─────────────────────────────────────┐
                        │   │  OPTIMIZATION LOOP                  │
                        │   ├─────────────────────────────────────┤
                        │   │  • Adjust hyperparameters           │
                        │   │  • Try different augmentation       │
                        │   │  • Use FMA dataset (more data)      │
                        │   │  • Ensemble models                  │
                        │   └──────────────┬──────────────────────┘
                        │                  │
                        │                  │
                        │   ┌──────────────┘
                        │   │
                        ▼   ▼
┌─────────────────────────────────────────────────────────────────┐
│              PHASE 4: PRODUCTION DEPLOYMENT                      │
├─────────────────────────────────────────────────────────────────┤
│  • Load best_model.pth                                          │
│  • Classify Music_TBC folder                                    │
│  • Organize files by predicted genre                            │
│  • Generate classification report                               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                         COMPLETE                                 │
│  Music files classified and organized by genre                  │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
                      ALTERNATIVE PATHS
═══════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────────┐
│                    COMPARISON PATH                               │
│                      (Optional)                                  │
├─────────────────────────────────────────────────────────────────┤
│  Script: training/compare_models.py                             │
│  • Compare XGBoost vs CNN vs OpenJMLA                           │
│  • Generate comparison report                                   │
│  • Output: model_comparison.json                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE PATH                                 │
│                  (Advanced - 2+ hours)                           │
├─────────────────────────────────────────────────────────────────┤
│  1. Train XGBoost (fast, interpretable)                         │
│  2. Train CNN (medium accuracy)                                 │
│  3. Train OpenJMLA V2 (high accuracy)                           │
│  4. Combine predictions (voting/averaging)                      │
│  • Expected: 80-90% accuracy                                    │
└─────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════
                      DECISION POINTS
═══════════════════════════════════════════════════════════════════

Need quick results (5 min)?
  → Use quick_baseline.py

Need interpretability?
  → Use train_xgboost_fma.py (feature importance)

Need best accuracy?
  → Use train_gtzan_v2.py ⭐

Limited GPU memory?
  → Run gpu_monitor.py first
  → Adjust batch size in config

Small dataset (GTZAN)?
  → Enable data augmentation
  → Use transfer learning (OpenJMLA)

Large dataset (FMA)?
  → Train from scratch possible
  → Consider distributed training


═══════════════════════════════════════════════════════════════════
                      TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════

Out of Memory (OOM)?
  ├─ Run: utils/gpu_monitor.py
  ├─ Reduce batch size (2 → 1)
  ├─ Enable gradient checkpointing
  └─ Clear cache more frequently

Low accuracy (<60%)?
  ├─ Check data quality (analyze_data.py)
  ├─ Enable data augmentation
  ├─ Try transfer learning
  └─ Use larger dataset (FMA)

Training too slow?
  ├─ Reduce epochs
  ├─ Enable early stopping
  ├─ Use smaller model
  └─ Try XGBoost baseline

Overfitting?
  ├─ Add dropout layers
  ├─ Enable data augmentation
  ├─ Reduce model complexity
  └─ Use more training data


═══════════════════════════════════════════════════════════════════
                      FILE LOCATIONS
═══════════════════════════════════════════════════════════════════

Scripts:     /media/mijesu_970/SSD_Data/Python/Music_Reclass/
Docs:        /media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass/
Datasets:    /media/mijesu_970/SSD_Data/DataSets/
Models:      /media/mijesu_970/SSD_Data/AI_models/OpenJMLA/
Target:      /media/mijesu_970/SSD_Data/Music_TBC/
```
