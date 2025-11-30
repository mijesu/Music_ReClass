import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Paths
GTZAN_PATH = "/path/to/GTZAN/Data"  # UPDATE THIS
OUTPUT_DIR = "./models"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class AudioDataset(Dataset):
    def __init__(self, data_path, genres, sr=22050, duration=30, augment=False):
        self.data_path = Path(data_path)
        self.genres = genres
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        self.augment = augment
        self.files = []
        
        print("Loading dataset...")
        for idx, genre in enumerate(genres):
            genre_path = self.data_path / genre
            if genre_path.exists():
                for audio_file in genre_path.glob('*.wav'):
                    try:
                        librosa.load(str(audio_file), sr=sr, duration=1)
                        self.files.append((str(audio_file), idx))
                    except:
                        print(f"Skipping corrupted: {audio_file.name}")
        print(f"Loaded {len(self.files)} files")
    
    def augment_audio(self, audio):
        # More aggressive augmentation
        if np.random.random() < 0.6:
            rate = np.random.uniform(0.85, 1.15)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        if np.random.random() < 0.6:
            n_steps = np.random.randint(-3, 4)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        if np.random.random() < 0.4:
            noise = np.random.randn(len(audio)) * np.random.uniform(0.003, 0.008)
            audio = audio + noise
        # Random volume
        if np.random.random() < 0.3:
            audio = audio * np.random.uniform(0.8, 1.2)
        return audio
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path, label = self.files[idx]
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        if self.augment:
            audio = self.augment_audio(audio)
        
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec_db).unsqueeze(0), label

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class EnhancedGTZANClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Deep ResNet-style architecture
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 3)
        self.layer4 = self._make_layer(256, 512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Multi-layer classifier with attention
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def _make_layer(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
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
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
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
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.title('Confusion Matrix - Enhanced GTZAN Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Acc')
    ax2.plot(history['val_acc'], label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*70)
    print("ENHANCED GTZAN CLASSIFIER - 4 HOUR TRAINING")
    print("="*70)
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Dataset
    full_dataset = AudioDataset(GTZAN_PATH, GENRES, augment=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.augment = True
    
    # Optimized for 4-hour training
    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=24, num_workers=4, pin_memory=True)
    
    print(f"\nDataset: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Batch size: 24, Workers: 4")
    
    # Enhanced model
    model = EnhancedGTZANClassifier(num_classes=len(GENRES)).to(device)
    criterion = FocalLoss(alpha=1, gamma=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=150, steps_per_epoch=len(train_loader)
    )
    scaler = torch.cuda.amp.GradScaler()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: ResNet-style with {total_params:,} parameters")
    print(f"Trainable: {trainable_params:,}")
    
    # Training
    print("\n" + "="*70)
    print("TRAINING START - Target: 4 hours")
    print("="*70)
    
    # Early stopping
    import sys
    from src.utils.early_stopping import EarlyStopping
    
    early_stopping = EarlyStopping(patience=20, mode='max', save_path=f'{OUTPUT_DIR}/GTZAN_enhanced.pth')
    
    epochs = 150
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
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
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': early_stopping.best_score,
                'history': history
            }, f'{OUTPUT_DIR}/checkpoint_epoch{epoch+1}.pth')
    
    total_time = time.time() - start_time
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/GTZAN_enhanced.pth'))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print(f"\nTraining Time: {total_time/3600:.2f} hours")
    print(f"Best Validation Accuracy: {val_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=GENRES, digits=3))
    
    plot_confusion_matrix(val_labels, val_preds, GENRES, f'{OUTPUT_DIR}/confusion_matrix_enhanced.png')
    plot_training_history(history, f'{OUTPUT_DIR}/training_history.png')
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print(f"  Model: {OUTPUT_DIR}/GTZAN_enhanced.pth")
    print(f"  Confusion matrix: {OUTPUT_DIR}/confusion_matrix_enhanced.png")
    print(f"  Training history: {OUTPUT_DIR}/training_history.png")
    print("="*70)

if __name__ == '__main__':
    main()
