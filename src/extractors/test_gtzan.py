#!/usr/bin/env python3
"""
Test GTZAN CNN Model - Quick inference test
Tests trained GTZAN model on a sample audio file
Shows top 3 genre predictions with confidence scores
"""

import torch
import torch.nn as nn
import librosa
import numpy as np

class GTZANClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),      # 0
            nn.BatchNorm2d(16),                   # 1
            nn.ReLU(),                            # 2
            nn.MaxPool2d(2),                      # 3
            nn.Dropout2d(0.2),                    # 4
            nn.Conv2d(16, 32, 3, padding=1),     # 5
            nn.BatchNorm2d(32),                   # 6
            nn.ReLU(),                            # 7
            nn.MaxPool2d(2),                      # 8
            nn.Dropout2d(0.2),                    # 9
            nn.Conv2d(32, 64, 3, padding=1),     # 10
            nn.BatchNorm2d(64),                   # 11
            nn.ReLU(),                            # 12
            nn.MaxPool2d(2),                      # 13
            nn.Dropout2d(0.3),                    # 14
            nn.Conv2d(64, 128, 3, padding=1),    # 15
            nn.BatchNorm2d(128),                  # 16
            nn.ReLU(),                            # 17
            nn.AdaptiveAvgPool2d(1)               # 18
        )
        self.classifier = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

checkpoint = torch.load('/media/mijesu_970/SSD_Data/AI_models/ZTGAN/GTZAN.pth', 
                        map_location='cpu', weights_only=False)

genres = checkpoint['label_encoder'].classes_
model = GTZANClassifier(len(genres))
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

audio_path = '/media/mijesu_970/SSD_Data/Musics_TBC/L(桃籽) - 你總要學會往前走.wav'
print(f"Testing: {audio_path.split('/')[-1]}\n")

audio, sr = librosa.load(audio_path, sr=22050, duration=30)
target_length = 22050 * 30

if len(audio) < target_length:
    audio = np.pad(audio, (0, target_length - len(audio)))
else:
    audio = audio[:target_length]

mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

with torch.no_grad():
    x = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
    output = model(x)
    probs = torch.softmax(output, dim=1)
    pred = torch.argmax(probs).item()
    top3 = torch.topk(probs[0], 3)
    
print(f"✓ Predicted Genre: {genres[pred]}")
print(f"✓ Confidence: {probs[0][pred]:.1%}\n")
print("Top 3 predictions:")
for i, (prob, idx) in enumerate(zip(top3.values, top3.indices), 1):
    print(f"  {i}. {genres[idx]}: {prob:.1%}")
