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
import argparse

# Paths - UPDATE THESE
FMA_PATH = "/media/mijesu_970/SSD_Data/DataSets/GTZAN/Data"  # Using GTZAN for test
GTZAN_PATH = "/media/mijesu_970/SSD_Data/DataSets/GTZAN/Data"
OUTPUT_DIR = "./models"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

FMA_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 
              'reggae', 'rock']  # Using GTZAN for test
GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 
                'reggae', 'rock']

class AudioDataset(Dataset):
    def __init__(self, data_path, genres, sr=22050, duration=30, augment=False, limit=None):
        self.data_path = Path(data_path)
        self.genres = genres
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        self.augment = augment
        self.files = []
        
        print(f"Loading dataset from {data_path}...")
        for idx, genre in enumerate(genres):
            genre_path = self.data_path / genre
            if not genre_path.exists():
                print(f"Warning: {genre} folder not found")
                continue
            
            files = list(genre_path.glob('*.mp3')) + list(genre_path.glob('*.wav'))
            if limit:
                files = files[:limit]
            
            for audio_file in files:
                try:
                    librosa.load(str(audio_file), sr=sr, duration=1)
                    self.files.append((str(audio_file), idx))
                except:
                    continue
        
        print(f"Loaded {len(self.files)} files")
    
    def augment_audio(self, audio):
        if np.random.random() < 0.6:
            rate = np.random.uniform(0.85, 1.15)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        if np.random.random() < 0.6:
            n_steps = np.random.randint(-3, 4)
            audio = librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
        if np.random.random() < 0.4:
            noise = np.random.randn(len(audio)) * np.random.uniform(0.003, 0.008)
            audio = audio + noise
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

class ProgressiveClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        # Feature extractor (shared)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 3)
        self.layer4 = self._make_layer(256, 512, 3)
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Classifier head (replaceable)
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
    
    def replace_classifier(self, num_classes):
        """Replace classifier head for fine-tuning"""
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
    plt.title('Progressive Training - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Progressive FMA Training')
    parser.add_argument('--mode', type=str, default='base', choices=['base', 'finetune'],
                       help='Training mode: base (FMA) or finetune (GTZAN)')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Path to pretrained model for fine-tuning')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize logger
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils.training_logger import TrainingLogger
    from utils.early_stopping import EarlyStopping
    
    exp_name = f'fma_{args.mode}'
    logger = TrainingLogger(log_dir='./logs', experiment_name=exp_name)
    
    logger.log_message("="*70)
    logger.log_message(f"PROGRESSIVE FMA TRAINING - {args.mode.upper()} MODE")
    logger.log_message("="*70)
    logger.log_message(f"Device: {device}")
    if torch.cuda.is_available():
        logger.log_message(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Dataset
    if args.mode == 'base':
        dataset = AudioDataset(FMA_PATH, FMA_GENRES, augment=False)
        num_classes = len(FMA_GENRES)
        model_name = 'FMA_base.pth'
    else:  # finetune
        dataset = AudioDataset(GTZAN_PATH, GTZAN_GENRES, augment=False)
        num_classes = len(GTZAN_GENRES)
        model_name = 'FMA_finetuned_GTZAN.pth'
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataset.dataset.augment = True
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            num_workers=4, pin_memory=True)
    
    logger.log_message(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    model = ProgressiveClassifier(num_classes=num_classes).to(device)
    
    # Load pretrained model for fine-tuning
    if args.mode == 'finetune' and args.load_model:
        logger.log_message(f"Loading pretrained model: {args.load_model}")
        checkpoint = torch.load(args.load_model)
        
        # Load feature extractor weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if 'classifier' not in k}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        
        # Replace classifier for new number of classes
        model.replace_classifier(num_classes)
        model = model.to(device)
        
        # Freeze feature extractor for first few epochs
        for param in model.conv1.parameters():
            param.requires_grad = False
        for param in model.layer1.parameters():
            param.requires_grad = False
        for param in model.layer2.parameters():
            param.requires_grad = False
        
        logger.log_message("Feature extractor frozen (layers 1-2)")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                   lr=0.001 if args.mode == 'base' else 0.0001, 
                                   weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01 if args.mode == 'base' else 0.001,
        epochs=args.epochs, steps_per_epoch=len(train_loader)
    )
    scaler = torch.cuda.amp.GradScaler()
    
    # Log config
    logger.log_config(
        mode=args.mode,
        model='ProgressiveClassifier',
        dataset='FMA' if args.mode == 'base' else 'GTZAN',
        num_classes=num_classes,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=0.001 if args.mode == 'base' else 0.0001,
        pretrained=args.load_model if args.mode == 'finetune' else None,
        model_parameters=sum(p.numel() for p in model.parameters()),
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    
    # Training
    logger.log_message("\nTraining started...")
    early_stopping = EarlyStopping(patience=15, mode='max', 
                                   save_path=f'{OUTPUT_DIR}/{model_name}')
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Unfreeze layers after 10 epochs in finetune mode
        if args.mode == 'finetune' and epoch == 10:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
            logger.log_message("All layers unfrozen")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.log_message(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s, Total: {elapsed/3600:.2f}h)")
        logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)
        
        if early_stopping(epoch, val_acc, model):
            logger.log_message("Early stopping triggered")
            break
    
    total_time = time.time() - start_time
    
    # Final evaluation
    logger.log_message("\n" + "="*70)
    logger.log_message("FINAL EVALUATION")
    logger.log_message("="*70)
    
    model.load_state_dict(torch.load(f'{OUTPUT_DIR}/{model_name}'))
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    genres = FMA_GENRES if args.mode == 'base' else GTZAN_GENRES
    report = classification_report(val_labels, val_preds, target_names=genres, digits=3)
    logger.log_final_results(val_acc, total_time, report)
    
    plot_confusion_matrix(val_labels, val_preds, genres, 
                         f'{OUTPUT_DIR}/confusion_matrix_{args.mode}.png')
    
    logger.log_message(f"\n✓ Model saved: {OUTPUT_DIR}/{model_name}")
    logger.log_message(f"✓ Next step: python train_fma_progressive.py --mode finetune --load-model {OUTPUT_DIR}/{model_name}")

if __name__ == '__main__':
    main()
