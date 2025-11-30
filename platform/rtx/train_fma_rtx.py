#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm

class FMADataset(Dataset):
    def __init__(self, audio_dir, track_ids, labels):
        self.audio_dir = audio_dir
        self.track_ids = track_ids
        self.labels = labels
    
    def __len__(self):
        return len(self.track_ids)
    
    def __getitem__(self, idx):
        track_id = self.track_ids[idx]
        # FMA directory structure: fma_medium/000/000002.mp3
        subdir = str(track_id).zfill(6)[:3]
        audio_path = os.path.join(self.audio_dir, subdir, f"{str(track_id).zfill(6)}.mp3")
        
        try:
            y, sr = librosa.load(audio_path, sr=22050, duration=30)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Resize to 128x128
            if mel_spec_db.shape[1] < 128:
                mel_spec_db = np.pad(mel_spec_db, ((0,0), (0, 128-mel_spec_db.shape[1])))
            else:
                mel_spec_db = mel_spec_db[:, :128]
            
            mel_spec_db = (mel_spec_db + 80) / 80  # Normalize
            mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        except:
            mel_spec_tensor = torch.zeros(1, 128, 128)
        
        return mel_spec_tensor, self.labels[idx]

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def load_fma_metadata():
    tracks = pd.read_csv('/media/mijesu_970/SSD_Data/DataSets/FMA/Misc/fma_metadata/tracks.csv',
                         index_col=0, header=[0, 1])
    
    # Get medium subset with valid genres
    medium = tracks['set', 'subset'] == 'medium'
    valid = tracks['track', 'genre_top'].notna() & medium
    
    track_ids = tracks[valid].index.tolist()
    labels_str = tracks.loc[valid, ('track', 'genre_top')].values
    
    genres = sorted(set(labels_str))
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    labels = [genre_to_idx[g] for g in labels_str]
    
    print(f"Loaded {len(track_ids)} tracks")
    print(f"Genres ({len(genres)}): {genres}")
    
    return track_ids, labels, genres

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    audio_dir = '/media/mijesu_970/SSD_Data/DataSets/FMA/Data/fma_medium'
    track_ids, labels, genres = load_fma_metadata()
    
    # Split data
    split = int(0.8 * len(track_ids))
    train_ids, val_ids = track_ids[:split], track_ids[split:]
    train_labels, val_labels = labels[:split], labels[split:]
    
    train_dataset = FMADataset(audio_dir, train_ids, train_labels)
    val_dataset = FMADataset(audio_dir, val_ids, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=4)
    
    model = SimpleCNN(num_classes=len(genres)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    print(f"\nTraining | Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    print(f"Batch size: 32 | Epochs: 30\n")
    
    best_acc = 0
    for epoch in range(30):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for specs, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
            specs, labels_batch = specs.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(specs)
                    loss = criterion(outputs, labels_batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(specs)
                loss = criterion(outputs, labels_batch)
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            correct += predicted.eq(labels_batch).sum().item()
        
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for specs, labels_batch in val_loader:
                specs, labels_batch = specs.to(device), labels_batch.to(device)
                outputs = model(specs)
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'genres': genres,
                'epoch': epoch,
                'accuracy': val_acc
            }, '/media/mijesu_970/SSD_Data/AI_models/fma_rtx_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete! Best Val Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    train()
