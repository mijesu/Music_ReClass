Reimport torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import gc

# Paths
MODEL_PATH = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
GTZAN_PATH = "/media/mijesu_970/SSD_Data/DataSets/GTZAN/Data"
FMA_PATH = "/media/mijesu_970/SSD_Data/DataSets/FMA/Data/fma_medium"

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def clear_memory():
    """Clear memory and run garbage collection"""
    gc.collect()
    torch.cuda.empty_cache()

def show_gpu_memory():
    """Display GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"GPU Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB")

class AudioDataset(Dataset):
    def __init__(self, data_path, genres, sr=22050, duration=30):
        self.data_path = Path(data_path)
        self.genres = genres
        self.sr = sr
        self.duration = duration
        self.target_length = sr * duration
        self.files = []
        
        for idx, genre in enumerate(genres):
            genre_path = self.data_path / genre
            if genre_path.exists():
                for audio_file in genre_path.glob('*.wav'):
                    self.files.append((str(audio_file), idx))
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        audio_path, label = self.files[idx]
        audio, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        
        # Pad or crop to fixed length
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        return mel_spec_tensor, label

class GenreClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
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
        
        # Clear memory after each batch
        del inputs, labels, outputs, loss
        clear_memory()
    
    return total_loss / len(train_loader), 100. * correct / total

def main():
    # Clear memory before starting
    clear_memory()
    
    device = torch.device('cuda')
    print(f"Using device: {device}")
    show_gpu_memory()
    
    # Load dataset
    print("\nLoading GTZAN dataset...")
    dataset = AudioDataset(GTZAN_PATH, GENRES)
    print(f"Found {len(dataset)} audio files")
    show_gpu_memory()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=0)
    
    # Model
    model = GenreClassifier(num_classes=len(GENRES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    print("\nModel loaded:")
    show_gpu_memory()
    
    # Training
    print("\nStarting training...")
    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        show_gpu_memory()
        clear_memory()
    
    # Save model
    torch.save(model.state_dict(), 'genre_classifier.pth')
    print("\nModel saved as 'genre_classifier.pth'")

if __name__ == '__main__':
    main()
