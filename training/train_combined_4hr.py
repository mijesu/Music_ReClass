import torch
import torch.nn as nn
import librosa
import numpy as np
import h5py
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Paths - UPDATE THESE
FMA_PATH = "/path/to/FMA/Data/fma_medium"
MSD_PATH = "/media/mijesu_970/SSD_Data/AI_models/MSD/Data"
OUTPUT_DIR = "./models"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

GENRES = ['blues', 'classical', 'country', 'disco', 'electronic', 'folk', 
          'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class MSDFeatureExtractor:
    """Extract features from MSD HDF5 files"""
    @staticmethod
    def extract(h5_file):
        try:
            with h5py.File(h5_file, 'r') as f:
                features = []
                
                # Analysis features
                if 'analysis/songs' in f:
                    songs = f['analysis/songs']
                    features.extend([
                        songs['danceability'][0],
                        songs['energy'][0],
                        songs['loudness'][0],
                        songs['tempo'][0],
                        songs['duration'][0]
                    ])
                
                # Timbre (12 coefficients mean)
                if 'analysis/segments_timbre' in f:
                    timbre = f['analysis/segments_timbre'][:]
                    features.extend(np.mean(timbre, axis=0)[:12])
                
                # Pitch (12 chroma mean)
                if 'analysis/segments_pitches' in f:
                    pitch = f['analysis/segments_pitches'][:]
                    features.extend(np.mean(pitch, axis=0)[:12])
                
                return np.array(features, dtype=np.float32)
        except:
            return np.zeros(29, dtype=np.float32)  # 5 + 12 + 12

class CombinedDataset(Dataset):
    def __init__(self, fma_path, msd_path, genres, sr=22050, duration=30, augment=False):
        self.fma_path = Path(fma_path)
        self.msd_path = Path(msd_path)
        self.genres = genres
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        self.augment = augment
        self.files = []
        
        print("Loading FMA dataset...")
        # Load FMA audio files
        for idx, genre in enumerate(genres):
            genre_path = self.fma_path / genre
            if genre_path.exists():
                for audio_file in list(genre_path.glob('*.mp3'))[:2000]:  # Limit for 4hr
                    self.files.append((str(audio_file), idx, 'fma'))
        
        print(f"Loaded {len(self.files)} FMA files")
        
        # Add MSD files (pre-computed features)
        print("Loading MSD features...")
        msd_files = list(Path(msd_path).rglob('*.h5'))[:5000]  # Limit for 4hr
        for h5_file in msd_files:
            # Assign random genre for demo (in production, use actual labels)
            idx = np.random.randint(0, len(genres))
            self.files.append((str(h5_file), idx, 'msd'))
        
        print(f"Total dataset: {len(self.files)} files")
    
    def augment_audio(self, audio):
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        if np.random.random() < 0.5:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        if np.random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        return audio
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_path, label, source = self.files[idx]
        
        if source == 'fma':
            # Process audio
            audio, _ = librosa.load(file_path, sr=self.sr, duration=self.duration)
            
            if self.augment:
                audio = self.augment_audio(audio)
            
            if len(audio) < self.target_length:
                audio = np.pad(audio, (0, self.target_length - len(audio)))
            else:
                audio = audio[:self.target_length]
            
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            audio_features = torch.FloatTensor(mel_spec_db).unsqueeze(0)
            
            # Dummy MSD features for FMA files
            msd_features = torch.zeros(29)
        else:
            # MSD file - use pre-computed features
            msd_features = torch.FloatTensor(MSDFeatureExtractor.extract(file_path))
            
            # Dummy audio features for MSD files
            audio_features = torch.zeros(1, 128, 1292)  # Typical mel-spec shape
        
        return audio_features, msd_features, label

class MultiModalClassifier(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        
        # Audio branch (CNN)
        self.audio_branch = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # MSD features branch (MLP)
        self.msd_branch = nn.Sequential(
            nn.Linear(29, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, audio, msd_features):
        audio_out = self.audio_branch(audio)
        msd_out = self.msd_branch(msd_features)
        
        # Concatenate features
        combined = torch.cat([audio_out, msd_out], dim=1)
        output = self.fusion(combined)
        return output

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for audio, msd_feat, labels in loader:
        audio = audio.to(device)
        msd_feat = msd_feat.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(audio, msd_feat)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for audio, msd_feat, labels in loader:
            audio = audio.to(device)
            msd_feat = msd_feat.to(device)
            labels = labels.to(device)
            
            outputs = model(audio, msd_feat)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), 100. * correct / total, all_preds, all_labels

def plot_confusion_matrix(y_true, y_pred, genres, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.title('Multi-Modal Classifier - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("COMBINED MULTI-MODAL TRAINING - 4 HOUR TARGET")
    print("="*70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Dataset
    full_dataset = CombinedDataset(FMA_PATH, MSD_PATH, GENRES, augment=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.augment = True
    
    # Optimized for 4 hours
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=20, num_workers=4, pin_memory=True)
    
    print(f"\nDataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Batch size: 20, Workers: 4")
    
    # Model
    model = MultiModalClassifier(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    scaler = torch.cuda.amp.GradScaler()
    
    print(f"\nModel: Multi-Modal (Audio CNN + MSD MLP)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    print("\n" + "="*70)
    print("TRAINING START - Target: 4 hours")
    print("="*70)
    
    # Early stopping
    import sys
    sys.path.append('..')
    from utils.early_stopping import EarlyStopping
    
    early_stopping = EarlyStopping(patience=10, mode='max', save_path=f'{OUTPUT_DIR}/combined_best.pth')
    
    epochs = 80
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s, Total: {elapsed/3600:.2f}h)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Early stopping check
        if early_stopping(epoch, val_acc, model):
            break
        
        # Stop if approaching 4 hours
        if elapsed > 3.8 * 3600:
            print(f"\n⏰ Reaching 4-hour limit, stopping training")
            break
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/combined_best.pth'))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print(f"\nTraining Time: {total_time/3600:.2f} hours")
    print(f"Best Validation Accuracy: {val_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=GENRES, digits=3))
    
    plot_confusion_matrix(val_labels, val_preds, GENRES, f'{OUTPUT_DIR}/confusion_matrix_combined.png')
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print(f"  Model: {OUTPUT_DIR}/combined_best.pth")
    print(f"  Confusion matrix: {OUTPUT_DIR}/confusion_matrix_combined.png")
    print(f"  Expected accuracy: 70-80% (with proper MSD labels)")
    print("="*70)

if __name__ == '__main__':
    main()
