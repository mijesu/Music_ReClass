# Daily Shutdown Checklist

Quick reference for ending your Music_ReClass development session properly.

---

## 🔚 Daily Shutdown Routine

### 1. In Kiro CLI
Simply type:
```
shutdown
```
or
```
goodnight
```

Kiro will automatically:
- ✅ Sync documentation from SSD
- ✅ Check for uncommitted changes (prompt to commit)
- ✅ Stop running Python processes
- ✅ Backup models, logs, features
- ✅ Create session notes template
- ✅ Clean temporary files
- ✅ Show session summary
- ✅ Display final checklist

### 2. Manual Checklist (if needed)

#### Code & Git
- [ ] All code changes tested
- [ ] All changes committed with clear messages
- [ ] Pushed to remote repository
- [ ] No merge conflicts

#### Training & Processes
- [ ] Training processes completed or safely stopped
- [ ] Training logs saved
- [ ] Model checkpoints backed up
- [ ] GPU memory cleared

#### Documentation
- [ ] Session notes documented
- [ ] PROJECT_HISTORY.md updated (if significant progress)
- [ ] TODO.md updated with next tasks
- [ ] Any issues documented

#### Files & Cleanup
- [ ] Important models backed up
- [ ] Feature files verified
- [ ] Logs archived
- [ ] Temporary files cleaned

---

## 📝 Session Documentation Template

Copy this to your session notes:

```markdown
# Session: [Date] [Time]

## Duration
Start: [HH:MM]
End: [HH:MM]

## Goals
- [ ] Goal 1
- [ ] Goal 2
- [ ] Goal 3

## Completed
- [x] Task 1 - Description
- [x] Task 2 - Description

## In Progress
- [ ] Task 3 - 50% complete
- [ ] Task 4 - Blocked by [reason]

## Issues Encountered
1. **Issue**: [Description]
   - **Solution**: [How it was resolved]
   - **Time Lost**: [Estimate]

2. **Issue**: [Description]
   - **Status**: Unresolved
   - **Next Steps**: [What to try]

## Performance Metrics
- Training Accuracy: [X]%
- Training Time: [X] minutes
- GPU Usage: [X]%
- Files Processed: [X]

## Code Changes
- Modified: [file1.py, file2.py]
- Added: [new_script.py]
- Deleted: [old_file.py]
- Commits: [N] commits

## Next Session TODO
- [ ] High Priority: [Task]
- [ ] Medium Priority: [Task]
- [ ] Low Priority: [Task]

## Notes & Learnings
- [Important observation]
- [Optimization discovered]
- [Idea for future]

## Questions for Kiro
- [ ] How to optimize [specific task]?
- [ ] Best approach for [problem]?
```

---

## 💾 Backup Strategy

### Automatic Backups (via shutdown.sh)
- **Location**: `~/Music_ReClass_Backups/YYYYMMDD/`
- **Includes**: Models, logs, features (if <100MB)
- **Frequency**: Every shutdown

### Manual Backups (Weekly)
```bash
# Full project backup
tar -czf ~/Backups/Music_ReClass_$(date +%Y%m%d).tar.gz ~/Music_ReClass

# Backup to external drive
rsync -av --progress ~/Music_ReClass /media/external/backups/
```

### What to Backup
- ✅ Trained models (*.pth)
- ✅ Training logs
- ✅ Session notes
- ✅ Configuration files
- ✅ Custom scripts
- ❌ Large datasets (keep original sources)
- ❌ Temporary files
- ❌ Cache files

---

## 🔍 Pre-Shutdown Checks

### Training Status
```bash
# Check running processes
ps aux | grep python3

# Check GPU usage
nvidia-smi

# Check last training log
tail -20 logs/training.log
```

### Git Status
```bash
# Check uncommitted changes
git status

# Check unpushed commits
git log origin/main..HEAD

# Check branch
git branch
```

### File Integrity
```bash
# Check model files
ls -lh models/*.pth

# Check feature files
ls -lh features/*.npy

# Check logs
ls -lh logs/*.log
```

---

## 🚨 Emergency Shutdown

If you need to shutdown quickly:

```bash
# Save current work
git add . && git commit -m "WIP: Emergency save $(date)"

# Stop all Python processes
pkill -f python3

# Quick backup
cp -r models ~/emergency_backup_$(date +%Y%m%d_%H%M%S)

# Push to remote
git push
```

---

## 📊 Weekly Review Checklist

At the end of each week:

### Progress Review
- [ ] Review all session notes
- [ ] Update PROJECT_HISTORY.md with weekly summary
- [ ] Calculate total training time
- [ ] Review accuracy improvements

### Code Quality
- [ ] Review and refactor messy code
- [ ] Update documentation
- [ ] Remove unused scripts
- [ ] Consolidate duplicate code

### Performance Analysis
- [ ] Compare model performances
- [ ] Identify bottlenecks
- [ ] Plan optimizations

### Planning
- [ ] Set goals for next week
- [ ] Prioritize tasks
- [ ] Identify blockers
- [ ] Schedule training runs

---

## 🎯 Monthly Maintenance

Once a month:

### Cleanup
- [ ] Archive old logs (>30 days)
- [ ] Remove old model checkpoints
- [ ] Clean up backup directory
- [ ] Update dependencies

### Documentation
- [ ] Review and update README.md
- [ ] Update MODEL_NOTES.md
- [ ] Consolidate scattered notes
- [ ] Update guides

### Optimization
- [ ] Profile code performance
- [ ] Optimize slow scripts
- [ ] Update training strategies
- [ ] Review best practices

---

## 💡 Best Practices

### Commit Messages
✅ Good:
- "Add MERT feature extraction with database support"
- "Fix GPU memory leak in training loop"
- "Update README with progressive voting strategy"

❌ Bad:
- "Update"
- "Fix stuff"
- "WIP"

### Session Notes
- Write notes immediately after completing tasks
- Include specific metrics and numbers
- Document both successes and failures
- Note time spent on each task

### Backup Habits
- Backup after every significant milestone
- Keep at least 3 versions of important models
- Test restore process periodically
- Store backups in multiple locations

---

## 🔗 Quick Commands

### Git Operations
```bash
# Commit all changes
git add . && git commit -m "Description"

# Push to remote
git push

# Create backup branch
git branch backup_$(date +%Y%m%d)

# View commit history
git log --oneline -10
```

### Process Management
```bash
# List Python processes
ps aux | grep python3

# Stop specific process
kill [PID]

# Stop all Python processes
pkill -f python3

# Check GPU processes
nvidia-smi
```

### File Management
```bash
# Find large files
find . -type f -size +100M

# Clean Python cache
find . -type d -name "__pycache__" -exec rm -rf {} +

# Archive old logs
tar -czf logs_archive_$(date +%Y%m%d).tar.gz logs/*.log
```

---

## 📞 Support & Resources

### Before Closing
- Check [DAILY_STARTUP.md](DAILY_STARTUP.md) for tomorrow's startup
- Review [TODO.md](TODO.md) for pending tasks
- Update [PROJECT_HISTORY.md](docs/PROJECT_HISTORY.md) if needed

### Documentation
- [SUMMARY.md](docs/SUMMARY.md) - Project overview
- [MODEL_NOTES.md](docs/MODEL_NOTES.md) - Model details
- [IMPLEMENTATION_GUIDES.md](docs/guides/IMPLEMENTATION_GUIDES.md) - How-to guides

---

## 🎓 Shutdown Script Usage

### Basic Usage
```bash
./shutdown.sh
```

### What It Does
1. ✅ Checks for uncommitted changes (prompts to commit)
2. ✅ Finds running Python processes (prompts to stop)
3. ✅ Backs up models, logs, and features
4. ✅ Creates session notes template
5. ✅ Cleans temporary files
6. ✅ Shows session summary

### Customization
Edit `shutdown.sh` to:
- Change backup location
- Add custom cleanup tasks
- Modify checklist items
- Add notification hooks

---

**Quick Shutdown Command:**
```bash
cd ~/Music_ReClass && ./shutdown.sh
```

**Emergency Save:**
```bash
git add . && git commit -m "WIP: $(date)" && git push
```

---

**Last Updated**: November 30, 2025  
**Version**: 1.0
