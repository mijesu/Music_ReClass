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

# Paths - UPDATE THESE FOR YOUR RTX PC
GTZAN_PATH = "/path/to/GTZAN/Data"  # Update this path
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

class GTZANClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Deeper CNN for RTX GPU
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Mixed precision training
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
    plt.title('Confusion Matrix - GTZAN Genre Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize logger
    import sys
    sys.path.append('..')
    from utils.training_logger import TrainingLogger
    
    logger = TrainingLogger(log_dir='./logs', experiment_name='gtzan_rtx')
    
    logger.log_message(f"Device: {device}")
    if torch.cuda.is_available():
        logger.log_message(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.log_message(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Dataset
    full_dataset = AudioDataset(GTZAN_PATH, GENRES, augment=False)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    train_dataset.dataset.augment = True
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4, pin_memory=True)
    
    logger.log_message(f"\nTrain: {len(train_dataset)}, Val: {len(val_dataset)}")
    logger.log_message(f"Batch size: 32, Workers: 4")
    
    # Model
    model = GTZANClassifier(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = torch.cuda.amp.GradScaler()
    
    # Log configuration
    logger.log_config(
        model='GTZANClassifier',
        dataset='GTZAN',
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        batch_size=32,
        num_workers=4,
        optimizer='AdamW',
        learning_rate=0.001,
        weight_decay=0.01,
        scheduler='CosineAnnealingWarmRestarts',
        epochs=50,
        patience=10,
        mixed_precision=True,
        model_parameters=sum(p.numel() for p in model.parameters())
    )
    
    logger.log_message(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    from utils.early_stopping import EarlyStopping
    
    logger.log_message("\n" + "="*70)
    logger.log_message("TRAINING START")
    logger.log_message("="*70)
    
    early_stopping = EarlyStopping(patience=10, mode='max', save_path=f'{OUTPUT_DIR}/GTZAN_best.pth')
    
    epochs = 50
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch metrics
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
        
        # Early stopping check
        if early_stopping(epoch, val_acc, model):
            logger.log_message("Early stopping triggered")
            break
    
    total_time = time.time() - start_time
    
    # Final evaluation
    logger.log_message("\n" + "="*70)
    logger.log_message("FINAL EVALUATION")
    logger.log_message("="*70)
    
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/GTZAN_best.pth'))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    report = classification_report(val_labels, val_preds, target_names=GENRES, digits=3)
    logger.log_final_results(val_acc, total_time, report)
    
    plot_confusion_matrix(val_labels, val_preds, GENRES, f'{OUTPUT_DIR}/confusion_matrix.png')
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print(f"  Model: {OUTPUT_DIR}/GTZAN_best.pth")
    print(f"  Confusion matrix: {OUTPUT_DIR}/confusion_matrix.png")
    print("="*70)

if __name__ == '__main__':
    main()
