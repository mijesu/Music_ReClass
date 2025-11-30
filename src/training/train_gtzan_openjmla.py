import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import os
import gc
from pathlib import Path

# Paths
GTZAN_PATH = "/media/mijesu_970/SSD_Data/datasets/GTZAN/Data/genres_original"
MODEL_PATH = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_gpu_memory():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'reserved': torch.cuda.memory_reserved() / 1024**2,
            'free': (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**2
        }
    return None

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class GTZANDataset(Dataset):
    def __init__(self, data_path, sr=22050, duration=30):
        self.files = []
        self.labels = []
        self.genres = sorted(os.listdir(data_path))
        self.sr = sr
        self.duration = duration
        
        for idx, genre in enumerate(self.genres):
            genre_path = os.path.join(data_path, genre)
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):
                    self.files.append(os.path.join(genre_path, file))
                    self.labels.append(idx)
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio, _ = librosa.load(self.files[idx], sr=self.sr, duration=self.duration)
        if len(audio) < self.sr * self.duration:
            audio = np.pad(audio, (0, self.sr * self.duration - len(audio)))
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return torch.FloatTensor(mel_db).unsqueeze(0), self.labels[idx]

class OpenJMLAClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        self.encoder = checkpoint['model']
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.classifier(features)

def train():
    # Check GPU and suggest batch size
    mem = get_gpu_memory()
    if mem:
        print(f"GPU Memory - Free: {mem['free']:.0f}MB")
        batch_size = max(2, min(8, int(mem['free'] / 200)))  # Auto-adjust batch size
        print(f"Using batch size: {batch_size}")
    else:
        batch_size = 4
        print("No GPU detected, using CPU with batch size 4")
    
    dataset = GTZANDataset(GTZAN_PATH)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    model = OpenJMLAClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.001)
    
    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)}")
    print(f"Device: {DEVICE}\n")
    
    for epoch in range(10):
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if batch_idx % 10 == 0:
                mem_info = get_gpu_memory()
                mem_str = f"GPU: {mem_info['allocated']:.0f}MB" if mem_info else ""
                print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f} {mem_str}")
            
            # Clear memory every 20 batches
            if batch_idx % 20 == 0:
                clear_memory()
        
        model.eval()
        val_loss, correct = 0, 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                val_loss += criterion(output, target).item()
                correct += (output.argmax(1) == target).sum().item()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}, "
              f"Val Acc: {100*correct/len(val_ds):.2f}%\n")
        clear_memory()
    
    torch.save(model.state_dict(), "gtzan_openjmla_model.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    train()
