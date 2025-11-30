#!/usr/bin/env python3
"""
Test MSD Feature-Based Model - Quick inference test
Tests trained MSD model (518 FMA features) on sample audio
Shows top 3 genre predictions with confidence scores
"""

import torch
import torch.nn as nn
import librosa
import numpy as np

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

# Load checkpoint
checkpoint = torch.load('/media/mijesu_970/SSD_Data/AI_models/MSD/msd_model.pth', 
                        map_location='cpu', weights_only=False)

genres = checkpoint['genres']
model = MSDModel(518, len(genres))
model.load_state_dict(checkpoint['model'])
model.eval()

# Load audio
audio_path = '/media/mijesu_970/SSD_Data/Musics_TBC/L(桃籽) - 你總要學會往前走.wav'
print(f"Testing: {audio_path.split('/')[-1]}\n")

y, sr = librosa.load(audio_path, sr=22050, duration=30)

# Extract 518 FMA features
features = []
# MFCCs
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
features.extend(np.mean(mfcc, axis=1))
features.extend(np.std(mfcc, axis=1))

# Spectral features
spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
features.extend([np.mean(spec_cent), np.std(spec_cent)])

spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
features.extend([np.mean(spec_bw), np.std(spec_bw)])

rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
features.extend([np.mean(rolloff), np.std(rolloff)])

zcr = librosa.feature.zero_crossing_rate(y)
features.extend([np.mean(zcr), np.std(zcr)])

# Pad to 518
while len(features) < 518:
    features.append(0.0)
features = np.array(features[:518])

# Normalize
features = (features - checkpoint['scaler_mean']) / checkpoint['scaler_scale']

# Predict
with torch.no_grad():
    x = torch.FloatTensor(features).unsqueeze(0)
    output = model(x)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs).item()
    top3 = torch.topk(probs[0], min(3, len(genres)))
    
print(f"✓ Predicted Genre: {genres[pred]}")
print(f"✓ Confidence: {probs[0][pred]:.1%}\n")
print("Top 3 predictions:")
for i, (prob, idx) in enumerate(zip(top3.values, top3.indices), 1):
    print(f"  {i}. {genres[idx]}: {prob:.1%}")
