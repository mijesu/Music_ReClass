# Daily Startup Checklist - Working with Kiro

Quick reference for starting your Music_ReClass development session with Kiro CLI.

---

## 🚀 Daily Startup Routine

### 1. Start Kiro CLI
```bash
cd ~/Music_ReClass
kiro-cli chat
```

### 2. Trigger Daily Routine
Simply type in kiro-cli:
```
startup
```
or
```
morning
```

Kiro will automatically:
- ✅ Show project status (git status, recent commits)
- ✅ Display TODO list from last session
- ✅ Check GPU status
- ✅ Show disk space usage
- ✅ List recent logs
- ✅ Display yesterday's session summary
- ✅ Auto sync from git (pull latest)
- ✅ Create today's session file
- ✅ Show quick reference commands

---

## 📋 Common Daily Tasks

### Training Tasks
- [ ] Check training logs: `tail -f logs/training.log`
- [ ] Monitor GPU usage: `watch -n 1 nvidia-smi`
- [ ] Resume training if interrupted
- [ ] Backup trained models to external storage

### Feature Extraction Tasks
- [ ] Check extracted features: `ls -lh features/`
- [ ] Verify feature dimensions match expected
- [ ] Run extraction for new audio files
- [ ] Update database with new features

### Classification Tasks
- [ ] Test models on sample files
- [ ] Review classification results
- [ ] Update genre labels if needed
- [ ] Generate classification reports

### Documentation Tasks
- [ ] Update PROJECT_HISTORY.md with progress
- [ ] Document any issues encountered
- [ ] Update TODO list
- [ ] Commit changes to git

---

## 🔧 Quick Commands Reference

### Project Navigation
```bash
cd ~/Music_ReClass                    # Main project
cd ~/Music_ReClass/extractors         # Feature extraction
cd ~/Music_ReClass/training           # Training scripts
cd ~/Music_ReClass/classification     # Classification scripts
cd ~/Music_ReClass/docs               # Documentation
```

### Git Operations
```bash
git status                            # Check changes
git add .                             # Stage all changes
git commit -m "Description"           # Commit changes
git push                              # Push to remote
./sync_and_push.sh                    # Automated sync
```

### Feature Extraction
```bash
# Standalone extraction (testing)
python3 extractors/extract_fma_features.py /path/to/audio/

# Database extraction (production)
python3 extractors/extract_fma.py
python3 extractors/extract_mert.py
python3 extractors/extract_jmla.py

# Extract all features
python3 extractors/extract_all_features.py
```

### Training
```bash
# Quick baseline (2 min)
python3 training/train_msd.py

# Production training (45 min)
python3 training/train_gtzan_v2.py

# Best accuracy (4 hrs)
python3 training/train_gtzan_enhanced.py

# Progressive ensemble (8-12 hrs)
python3 training/train_fma_progressive.py
```

### Classification
```bash
# Classify music files
python3 classification/classify_music_tbc.py --input /path/to/music

# Ensemble classification
python3 classification/Reclass_FMJ_EV.py
```

### Analysis
```bash
# Dataset analysis
python3 analysis/analyze_data.py

# Model inspection
python3 analysis/check_model.py

# GPU monitoring
python3 utils/gpu_monitor.py
```

---

## 💬 Effective Kiro Prompts

### Getting Started
- "Show me the project structure"
- "What scripts are available in extractors folder?"
- "Explain the progressive voting strategy"

### Development
- "Create a script to extract features from [folder]"
- "Help me debug this error: [error message]"
- "Optimize this training script for better performance"

### Analysis
- "Compare the accuracy of FMA vs MERT features"
- "Show me the confusion matrix for the last training"
- "What genres are most difficult to classify?"

### Documentation
- "Update the README with the new feature"
- "Document this function: [code]"
- "Create a guide for [specific task]"

### Troubleshooting
- "Why is GPU memory running out?"
- "How to fix this import error?"
- "Optimize batch size for my GPU"

---

## 📊 Daily Progress Tracking

### Morning Check
- [ ] Review yesterday's training results
- [ ] Check for any errors in logs
- [ ] Plan today's tasks with Kiro

### During Work
- [ ] Document decisions and changes
- [ ] Test code before committing
- [ ] Ask Kiro for optimization suggestions

### End of Day
- [ ] Commit all changes to git
- [ ] Update PROJECT_HISTORY.md
- [ ] Note any issues for next session
- [ ] Backup important files

---

## 🎯 Weekly Goals Template

### Week of [Date]

**Training Goals:**
- [ ] Train model on [dataset]
- [ ] Achieve [target]% accuracy
- [ ] Optimize [specific aspect]

**Feature Extraction:**
- [ ] Extract features for [N] songs
- [ ] Verify feature quality
- [ ] Update database

**Classification:**
- [ ] Classify [folder/dataset]
- [ ] Review and correct errors
- [ ] Generate reports

**Documentation:**
- [ ] Update [specific docs]
- [ ] Create guide for [task]
- [ ] Review and consolidate

---

## 🔍 Troubleshooting Quick Reference

### GPU Issues
```bash
# Check GPU status
nvidia-smi

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

### Python Environment
```bash
# Check Python version
python3 --version

# Check installed packages
pip list | grep -E "torch|librosa|numpy"

# Reinstall package
pip install --upgrade [package]
```

### Git Issues
```bash
# Discard local changes
git checkout -- [file]

# Reset to last commit
git reset --hard HEAD

# Pull latest changes
git pull origin main
```

---

## 📁 Important File Locations

### Models
- Trained models: `AI_models/trained_models/`
- MSD model: `AI_models/MSD/msd_model.pth`
- GTZAN model: `AI_models/ZTGAN/GTZAN.pth`

### Features
- FMA features: `features/FMA_features.npy`
- MERT features: `features/MERT_features.npy`
- JMLA features: `features/JMLA_features.npy`

### Logs
- Training logs: `logs/training.log`
- FMA logs: `logs/fma_base.log`
- Chat histories: `logs/chat_history_*.json`

### Documentation
- Main docs: `docs/`
- Guides: `docs/guides/`
- Technical: `docs/technical/`
- Model notes: `docs/MODEL_NOTES.md`

---

## 🎓 Learning Resources

### Project Documentation
- [SUMMARY.md](docs/SUMMARY.md) - Project overview
- [MODEL_NOTES.md](docs/MODEL_NOTES.md) - Model details
- [FEATURES_AND_CONCEPTS.md](docs/FEATURES_AND_CONCEPTS.md) - ML concepts
- [IMPLEMENTATION_GUIDES.md](docs/guides/IMPLEMENTATION_GUIDES.md) - How-to guides

### External Resources
- [FMA Dataset](https://github.com/mdeff/fma)
- [MERT Model](https://huggingface.co/m-a-p/MERT-v1-330M)
- [OpenJMLA](https://huggingface.co/UniMus/OpenJMLA)
- [librosa Documentation](https://librosa.org/doc/latest/)

---

## 💡 Tips for Working with Kiro

### Be Specific
❌ "Fix this code"
✅ "This training script has a GPU memory error on line 45. How can I optimize batch size?"

### Provide Context
❌ "Create a script"
✅ "Create a script to extract FMA features from /path/to/audio and save to features/ folder"

### Ask for Explanations
- "Explain how progressive voting works"
- "Why is MERT better than FMA for this task?"
- "What's the difference between these two approaches?"

### Request Reviews
- "Review this code for optimization opportunities"
- "Check if this documentation is clear"
- "Suggest improvements for this training script"

### Iterate
- Start with simple requests
- Build on previous responses
- Refine based on results

---

## 📝 Session Template

Copy this template for each session:

```markdown
## Session: [Date]

### Goals
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

### Completed
- [x] Task 1
- [x] Task 2

### Issues Encountered
- Issue 1: [description] → Solution: [solution]
- Issue 2: [description] → Solution: [solution]

### Next Session
- [ ] Continue with [task]
- [ ] Fix [issue]
- [ ] Implement [feature]

### Notes
- Important decision: [description]
- Performance: [metrics]
- Ideas: [future improvements]
```

---

**Quick Start Command:**
```bash
cd ~/Music_ReClass && kiro-cli chat
```

**First Prompt:**
```
Show me today's tasks and recent project updates
```

---

**Last Updated**: November 30, 2025  
**Version**: 1.0
