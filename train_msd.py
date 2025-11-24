#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

class FMAFeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class MSDModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)

def load_fma_features():
    """Load FMA pre-computed features"""
    print("Loading FMA metadata...")
    tracks = pd.read_csv('/media/mijesu_970/SSD_Data/DataSets/FMA/Misc/fma_metadata/tracks.csv', 
                         index_col=0, header=[0, 1])
    features = pd.read_csv('/media/mijesu_970/SSD_Data/DataSets/FMA/Misc/fma_metadata/features.csv',
                          index_col=0, header=[0, 1, 2])
    
    # Get medium subset tracks
    medium = tracks['set', 'subset'] == 'medium'
    
    # Get genre labels (top-level genre)
    genre_col = ('track', 'genre_top')
    valid = tracks[genre_col].notna() & medium
    
    track_ids = tracks[valid].index
    labels_str = tracks.loc[track_ids, genre_col].values
    
    # Map genres to indices
    genres = sorted(set(labels_str))
    genre_to_idx = {g: i for i, g in enumerate(genres)}
    labels = [genre_to_idx[g] for g in labels_str]
    
    # Use all available features (chroma, mfcc, spectral, etc.)
    X = features.loc[track_ids].values
    
    # Remove NaN
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    labels = np.array(labels)[valid_mask]
    
    print(f"Loaded {len(X)} tracks with {X.shape[1]} features")
    print(f"Genres ({len(genres)}): {genres}")
    
    return X, labels, genres

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Loading FMA pre-computed features...")
    X, y, genres = load_fma_features()
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_dataset = FMAFeatureDataset(X_train, y_train)
    val_dataset = FMAFeatureDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    model = MSDModel(input_size=X.shape[1], num_classes=len(genres)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)
    
    print(f"\nTraining on {device} | Train: {len(train_dataset)} | Val: {len(val_dataset)}\n")
    
    best_acc = 0
    for epoch in range(50):
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for features, labels_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/50"):
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
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'genres': genres,
                'epoch': epoch,
                'accuracy': val_acc
            }, '/media/mijesu_970/SSD_Data/AI_models/msd_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete! Best Val Acc: {best_acc:.2f}%")

if __name__ == '__main__':
    train()
