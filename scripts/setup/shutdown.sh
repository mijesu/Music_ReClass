#!/bin/bash
# Music_ReClass Daily Shutdown Script
# Run this at the end of each work session

clear
echo "=========================================="
echo "  Music_ReClass - Daily Shutdown"
echo "=========================================="
echo ""

# Navigate to project root (2 levels up from scripts/setup/)
cd "$(dirname "$0")/../.."
PROJECT_DIR=$(pwd)
echo "📁 Project Directory: $PROJECT_DIR"
echo ""

# Sync documentation from SSD
echo "📥 Syncing documentation from SSD..."
SRC="/media/mijesu_970/SSD_Data/Kiro_Projects/Music_Reclass"
DEST="$PROJECT_DIR/docs"
if [ -d "$SRC" ]; then
    rsync -av --delete "$SRC"/*.md "$DEST/" 2>/dev/null
    rsync -av --delete "$SRC/Reference/" "$DEST/Reference/" 2>/dev/null
    echo "  ✅ Documentation synced"
else
    echo "  ⚠️  SSD source not found, skipping sync"
fi
echo ""

# Check for uncommitted changes
echo "📊 Checking for uncommitted changes..."
if [[ -n $(git status -s) ]]; then
    echo "  ⚠️  You have uncommitted changes:"
    git status -s | sed 's/^/    /'
    echo ""
    read -p "  Do you want to commit these changes? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        read -p "  Enter commit message: " commit_msg
        git add .
        git commit -m "$commit_msg"
        echo "  ✅ Changes committed"
        echo ""
        read -p "  Push to remote? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            git push
            echo "  ✅ Pushed to remote"
        fi
    fi
else
    echo "  ✅ No uncommitted changes"
fi
echo ""

# Check for running training processes
echo "🔍 Checking for running Python processes..."
PYTHON_PROCS=$(ps aux | grep -E "python3.*(train|extract|classify)" | grep -v grep)
if [[ -n "$PYTHON_PROCS" ]]; then
    echo "  ⚠️  Found running processes:"
    echo "$PYTHON_PROCS" | awk '{printf "    PID %s: %s\n", $2, $11}' 
    echo ""
    read -p "  Do you want to stop these processes? (y/n): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ps aux | grep -E "python3.*(train|extract|classify)" | grep -v grep | awk '{print $2}' | xargs kill
        echo "  ✅ Processes stopped"
    fi
else
    echo "  ✅ No running processes"
fi
echo ""

# Backup important files
echo "💾 Backing up important files..."
BACKUP_DIR="$HOME/Music_ReClass_Backups/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup trained models
if [ -d "models" ] && [ "$(ls -A models/*.pth 2>/dev/null)" ]; then
    cp models/*.pth "$BACKUP_DIR/" 2>/dev/null
    echo "  ✅ Backed up models to $BACKUP_DIR"
fi

# Backup logs
if [ -d "logs" ] && [ "$(ls -A logs/*.log 2>/dev/null)" ]; then
    cp logs/*.log "$BACKUP_DIR/" 2>/dev/null
    echo "  ✅ Backed up logs to $BACKUP_DIR"
fi

# Backup features (if small enough)
FEATURES_SIZE=$(du -sm features 2>/dev/null | cut -f1)
if [ -n "$FEATURES_SIZE" ] && [ "$FEATURES_SIZE" -lt 100 ]; then
    cp -r features "$BACKUP_DIR/" 2>/dev/null
    echo "  ✅ Backed up features to $BACKUP_DIR"
fi
echo ""

# Session summary
echo "📝 Session Summary:"
TODAY=$(date +%Y-%m-%d)
echo "  Date: $TODAY"
echo "  Commits today: $(git log --since="$TODAY 00:00" --oneline | wc -l)"
echo "  Files modified: $(git diff --name-only HEAD~1 2>/dev/null | wc -l)"
echo ""

# Check GPU status before shutdown
echo "🖥️  Final GPU Status:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s | Memory: %s/%s MB | Temp: %s°C\n", $1, $2, $3, $4, $5}'
else
    echo "  ⚠️  nvidia-smi not available"
fi
echo ""

# Create session notes file
SESSION_FILE="logs/session_$(date +%Y%m%d_%H%M%S).md"
cat > "$SESSION_FILE" << EOF
# Session Notes - $(date +"%Y-%m-%d %H:%M:%S")

## What I Worked On
- 

## Completed Tasks
- [ ] 

## Issues Encountered
- 

## Next Session TODO
- [ ] 

## Notes
- 

EOF
echo "📋 Session notes template created: $SESSION_FILE"
echo "   Edit this file to document your session"
echo ""

# Cleanup temporary files
echo "🧹 Cleaning up temporary files..."
find . -type f -name "*.pyc" -delete 2>/dev/null
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name ".DS_Store" -delete 2>/dev/null
find . -type f -name "*.tmp" -delete 2>/dev/null
echo "  ✅ Cleanup complete"
echo ""

# Final checklist
echo "=========================================="
echo "✅ Daily Shutdown Checklist:"
echo ""
echo "  [ ] All changes committed and pushed"
echo "  [ ] Training processes stopped"
echo "  [ ] Important files backed up"
echo "  [ ] Session notes documented"
echo "  [ ] Temporary files cleaned"
echo ""
echo "📖 Remember to:"
echo "  • Update PROJECT_HISTORY.md if significant progress"
echo "  • Document any issues in session notes"
echo "  • Plan tomorrow's tasks"
echo ""
echo "=========================================="
echo "👋 Session closed. See you next time!"
echo "=========================================="
