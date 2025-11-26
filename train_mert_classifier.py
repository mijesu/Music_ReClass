#!/usr/bin/env python3
"""
Train MERT Classifier on FMA Dataset
Extracts MERT embeddings and trains a simple MLP classifier

Expected Accuracy: 78-85%
Training Time: 2-4 hours
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Genre mapping (16 FMA genres)
GENRES = ['Blues', 'Classical', 'Country', 'Easy Listening', 'Electronic', 
          'Experimental', 'Folk', 'Hip-Hop', 'Instrumental', 'International',
          'Jazz', 'Old-Time/Historic', 'Pop', 'Rock', 'Soul-RnB', 'Spoken']

class SimpleClassifier(nn.Module):
    """Simple MLP classifier"""
    def __init__(self, input_dim=768, num_classes=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

class MERTFeatureExtractor:
    """Extract MERT embeddings from audio files"""
    def __init__(self, device='cuda'):
        self.device = device
        print("Loading MERT model...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True)
        mert_path = "/media/mijesu_970/SSD_Data/AI_models/MERT"
        self.model = AutoModel.from_pretrained(
            mert_path, trust_remote_code=True).to(device)
        self.model.eval()
        print("MERT model loaded!")
    
    def extract(self, audio_path):
        """Extract 768-dim MERT embedding"""
        audio, sr = librosa.load(audio_path, sr=24000, duration=30)
        inputs = self.processor(audio, sampling_rate=24000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.cpu().numpy()[0]

def extract_mert_features(fma_audio_dir, fma_metadata, output_path, device='cuda'):
    """Extract MERT features for all FMA tracks"""
    extractor = MERTFeatureExtractor(device)
    
    # Load FMA metadata
    metadata = np.load(fma_metadata, allow_pickle=True).item()
    
    features = {}
    print(f"\nExtracting MERT features from {len(metadata)} tracks...")
    
    for track_id, info in tqdm(metadata.items()):
        audio_path = Path(fma_audio_dir) / info['path']
        if audio_path.exists():
            try:
                features[track_id] = {
                    'embedding': extractor.extract(str(audio_path)),
                    'genre': info['genre'],
                    'genre_id': GENRES.index(info['genre'])
                }
            except Exception as e:
                print(f"Error processing {track_id}: {e}")
    
    # Save features
    np.save(output_path, features)
    print(f"\nSaved {len(features)} MERT features to {output_path}")
    return features

def train_classifier(features_path, output_model_path, device='cuda', epochs=20, batch_size=32):
    """Train classifier on MERT features"""
    
    # Load features
    print(f"Loading MERT features from {features_path}...")
    features = np.load(features_path, allow_pickle=True).item()
    
    # Prepare data
    X = np.array([f['embedding'] for f in features.values()])
    y = np.array([f['genre_id'] for f in features.values()])
    
    # Split train/val
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(device)
    y_train = torch.LongTensor(y_train).to(device)
    X_val = torch.FloatTensor(X_val).to(device)
    y_val = torch.LongTensor(y_val).to(device)
    
    # Create model
    model = SimpleClassifier(768, 16).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\nTraining on {len(X_train)} samples, validating on {len(X_val)} samples")
    print(f"Device: {device}")
    
    best_acc = 0
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * train_correct / len(X_train)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_correct = (val_predicted == y_val).sum().item()
            val_acc = 100 * val_correct / len(X_val)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss/len(X_train)*batch_size:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_model_path)
            print(f"  → Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {output_model_path}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MERT Classifier')
    parser.add_argument('--mode', choices=['extract', 'train', 'both'], default='both',
                       help='Extract features, train classifier, or both')
    parser.add_argument('--fma-audio', 
                       default='/media/mijesu_970/SSD_Data/DataSets/FMA/Data/fma_medium',
                       help='FMA audio directory')
    parser.add_argument('--fma-metadata',
                       default='/media/mijesu_970/SSD_Data/AI_models/FMA/FMA.npy',
                       help='FMA metadata file')
    parser.add_argument('--features-output',
                       default='/media/mijesu_970/SSD_Data/AI_models/FMA/MERT_features.npy',
                       help='Output path for MERT features')
    parser.add_argument('--model-output',
                       default='/media/mijesu_970/SSD_Data/AI_models/trained_models/mert_classifier.pth',
                       help='Output path for trained model')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    
    args = parser.parse_args()
    
    if args.mode in ['extract', 'both']:
        print("="*60)
        print("STEP 1: Extracting MERT Features")
        print("="*60)
        extract_mert_features(
            args.fma_audio,
            args.fma_metadata,
            args.features_output,
            args.device
        )
    
    if args.mode in ['train', 'both']:
        print("\n" + "="*60)
        print("STEP 2: Training MERT Classifier")
        print("="*60)
        train_classifier(
            args.features_output,
            args.model_output,
            args.device,
            args.epochs,
            args.batch_size
        )

if __name__ == '__main__':
    main()
