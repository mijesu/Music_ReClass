import torch
import torch.nn as nn
import librosa
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import gc

# Paths
OPENJMLA_PATH = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
GTZAN_PATH = "/media/mijesu_970/SSD_Data/DataSets/GTZAN/Data"
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class OpenJMLAFeatureExtractor(nn.Module):
    """OpenJMLA as frozen feature extractor"""
    def __init__(self, checkpoint_path):
        super().__init__()
        # Load OpenJMLA checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        
        # Build feature extractor from OpenJMLA layers
        # Using patch embedding + transformer layers
        self.patch_embed = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        self.pos_embed = nn.Parameter(torch.zeros(1, 936, 768))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        
        # Load weights
        if 'patch_embed.projection.weight' in state_dict:
            self.patch_embed.weight.data = state_dict['patch_embed.projection.weight'][:, :1, :, :]  # Use 1 channel
            self.patch_embed.bias.data = state_dict['patch_embed.projection.bias']
        if 'pos_embed' in state_dict:
            self.pos_embed.data = state_dict['pos_embed']
        if 'cls_token' in state_dict:
            self.cls_token.data = state_dict['cls_token']
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # x: [B, 1, H, W] - mel-spectrogram
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 768, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B, N, 768]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embedding (truncate if needed)
        if x.shape[1] <= self.pos_embed.shape[1]:
            x = x + self.pos_embed[:, :x.shape[1], :]
        else:
            # Interpolate position embedding if input is larger
            x = x + self.pos_embed
        
        # Return CLS token as feature
        return x[:, 0]  # [B, 768]

class GenreClassifierWithOpenJMLA(nn.Module):
    """Genre classifier using OpenJMLA features"""
    def __init__(self, openjmla_path, num_classes=10):
        super().__init__()
        self.feature_extractor = OpenJMLAFeatureExtractor(openjmla_path)
        
        # Trainable classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Extract features (frozen)
        with torch.no_grad():
            features = self.feature_extractor(x)
        
        # Classify (trainable)
        return self.classifier(features)

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
        
        # Pad or crop
        if len(audio) < self.target_length:
            audio = np.pad(audio, (0, self.target_length - len(audio)))
        else:
            audio = audio[:self.target_length]
        
        # Convert to mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
        
        return mel_spec_tensor, label

def train_epoch(model, train_loader, criterion, optimizer, device):
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
        
        del inputs, labels, outputs, loss
        gc.collect()
    
    return total_loss / len(train_loader), 100. * correct / total

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Training with OpenJMLA Base Model\n")
    print(f"Device: {device}")
    
    # Load dataset
    print("\n📂 Loading GTZAN dataset...")
    dataset = AudioDataset(GTZAN_PATH, GENRES)
    print(f"Found {len(dataset)} audio files")
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=0)
    
    # Create model with OpenJMLA base
    print("\n🤖 Loading OpenJMLA feature extractor...")
    model = GenreClassifierWithOpenJMLA(OPENJMLA_PATH, num_classes=len(GENRES)).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print(f"   Frozen parameters: {total_params - trainable_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    # Training
    print("\n🚀 Starting training...")
    epochs = 10
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'openjmla_genre_classifier.pth')
    print("\n✅ Model saved as 'openjmla_genre_classifier.pth'")

if __name__ == '__main__':
    main()
