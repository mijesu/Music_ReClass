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
MODEL_PATH = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
GTZAN_PATH = "/media/mijesu_970/SSD_Data/DataSets/GTZAN/Data"
CHECKPOINT_PATH = "checkpoint_v2.pth"

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
        
        for idx, genre in enumerate(genres):
            genre_path = self.data_path / genre
            if genre_path.exists():
                for audio_file in genre_path.glob('*.wav'):
                    # Test if file can be loaded
                    try:
                        librosa.load(str(audio_file), sr=sr, duration=1)
                        self.files.append((str(audio_file), idx))
                    except:
                        print(f"Skipping corrupted file: {audio_file.name}")
                        continue
    
    def augment_audio(self, audio):
        """Apply data augmentation"""
        # Time stretching
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Pitch shifting
        if np.random.random() < 0.5:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        
        # Add noise
        if np.random.random() < 0.3:
            noise = np.random.randn(len(audio)) * 0.005
            audio = audio + noise
        
        return audio
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path, label = self.files[idx]
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Augmentation
        if self.augment:
            audio = self.augment_audio(audio)
        
        # Pad or crop
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        # Extract features
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return torch.FloatTensor(mel_spec_db).unsqueeze(0), label

class OpenJMLAClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Simple CNN for genre classification
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        # Trainable classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
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

def plot_confusion_matrix(y_true, y_pred, genres, save_path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=genres, yticklabels=genres)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'val_acc': val_acc
    }, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer, scheduler):
    if Path(CHECKPOINT_PATH).exists():
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}, Val Acc: {checkpoint['val_acc']:.2f}%")
        return checkpoint['epoch'] + 1
    return 0

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Dataset
    print("\nLoading GTZAN dataset...")
    full_dataset = AudioDataset(GTZAN_PATH, GENRES, augment=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Enable augmentation for training
    train_dataset.dataset.augment = True
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    print("\nInitializing OpenJMLA model...")
    model = OpenJMLAClassifier(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Load checkpoint if exists
    start_epoch = load_checkpoint(model, optimizer, scheduler)
    
    # Training
    print("\n" + "="*60)
    print("TRAINING START")
    print("="*60)
    
    epochs = 20
    best_val_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch+1}/{epochs} ({epoch_time:.1f}s)")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_v2.pth')
            print(f"  ✓ Best model saved (Acc: {best_val_acc:.2f}%)")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{patience}")
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, val_acc)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    model.load_state_dict(torch.load('best_model_v2.pth'))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    print(f"\nBest Validation Accuracy: {val_acc:.2f}%")
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=GENRES))
    
    # Plot confusion matrix
    plot_confusion_matrix(val_labels, val_preds, GENRES)
    
    print("\n✓ Training complete!")
    print(f"  Best model: best_model_v2.pth")
    print(f"  Confusion matrix: confusion_matrix.png")

if __name__ == '__main__':
    main()
