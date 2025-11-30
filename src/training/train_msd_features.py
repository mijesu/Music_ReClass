#!/usr/bin/env python3
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# MSD feature dataset
class MSDDataset(Dataset):
    def __init__(self, file_paths, labels):
        self.file_paths = file_paths
        self.labels = labels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        with h5py.File(self.file_paths[idx], 'r') as f:
            features = []
            if 'analysis/songs' in f:
                song = f['analysis/songs'][0]
                features.extend([
                    song['danceability'], song['energy'], song['loudness'],
                    song['tempo'], song['duration'], song['key'], song['mode']
                ])
            if 'analysis/segments_timbre' in f:
                timbre = np.array(f['analysis/segments_timbre'])
                features.extend(timbre.mean(axis=0)[:12])
            if 'analysis/segments_pitches' in f:
                pitches = np.array(f['analysis/segments_pitches'])
                features.extend(pitches.mean(axis=0)[:12])
            
            features = np.array(features, dtype=np.float32)
            features = np.nan_to_num(features, 0)
        
        return torch.FloatTensor(features), self.labels[idx]

# Simple MLP model
class MSDModel(nn.Module):
    def __init__(self, input_size=31, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def load_tagtraum_labels(tagtraum_file):
    """Load Tagtraum genre annotations"""
    track_to_genre = {}
    with open(tagtraum_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) == 2:
                track_id, genre = parts
                track_to_genre[track_id] = genre
    return track_to_genre

def load_msd_data(msd_dir, tagtraum_file):
    """Load MSD files with Tagtraum genre labels"""
    track_to_genre = load_tagtraum_labels(tagtraum_file)
    
    # Get unique genres and create mapping
    genres = sorted(set(track_to_genre.values()))
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    
    file_paths, labels = [], []
    
    # Walk through MSD directory structure
    for root, dirs, files in os.walk(msd_dir):
        for h5_file in files:
            if not h5_file.endswith('.h5'):
                continue
            
            # Extract track ID from filename (format: TRXXXXX.h5)
            track_id = h5_file.replace('.h5', '')
            
            if track_id in track_to_genre:
                genre = track_to_genre[track_id]
                file_paths.append(os.path.join(root, h5_file))
                labels.append(genre_to_idx[genre])
    
    print(f"Found {len(file_paths)} tracks with genre labels")
    print(f"Genres: {genres}")
    
    return file_paths, labels, genres

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    msd_dir = '/media/mijesu_970/SSD_Data/DataSets/millionsongsubset_full/data'
    tagtraum_file = '/media/mijesu_970/SSD_Data/DataSets/msd_tagtraum_cd1.cls'
    
    print("Loading MSD data with Tagtraum labels...")
    file_paths, labels, genres = load_msd_data(msd_dir, tagtraum_file)
    
    X_train, X_val, y_train, y_val = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42
    )
    
    train_dataset = MSDDataset(X_train, y_train)
    val_dataset = MSDDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    
    model = MSDModel(num_classes=len(genres)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {device} with {len(train_dataset)} samples")
    
    for epoch in range(50):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for features, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features, labels_batch = features.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
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
            for features, labels_batch in val_loader:
                features, labels_batch = features.to(device), labels_batch.to(device)
                outputs = model(features)
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()
        
        val_acc = 100. * val_correct / val_total
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    torch.save(model.state_dict(), 'msd_model.pth')
    print("Model saved to msd_model.pth")

if __name__ == '__main__':
    train()
