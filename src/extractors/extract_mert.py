#!/usr/bin/env python3
"""
Extract MERT features (768 dims) - DATABASE VERSION
For production use with SQLite database integration
Use extract_mert_features.py for standalone/testing
"""

import torch
import torchaudio
import numpy as np
import sqlite3
from pathlib import Path
from tqdm import tqdm
import io
from transformers import AutoModel
from src.utils.config import DATABASE_PATH, MERT_DIMS, DEVICE

class MERTExtractor:
    def __init__(self):
        print("Loading MERT model...")
        self.model = AutoModel.from_pretrained("m-a-p/MERT-v1-330M", trust_remote_code=True)
        self.model.to(DEVICE)
        self.model.eval()
        self.sr = 24000  # MERT requires 24kHz
        print(f"MERT model loaded on {DEVICE}")
    
    def extract(self, audio_path):
        """Extract MERT features"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample to 24kHz
            if sr != self.sr:
                resampler = torchaudio.transforms.Resample(sr, self.sr)
                waveform = resampler(waveform)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Truncate to 30 seconds
            max_length = self.sr * 30
            if waveform.shape[1] > max_length:
                waveform = waveform[:, :max_length]
            
            # Extract features
            with torch.no_grad():
                waveform = waveform.to(DEVICE)
                outputs = self.model(waveform, output_hidden_states=True)
                # Use last hidden state, average over time
                features = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"Error: {e}")
            return None

def save_features(db, song_id, features):
    """Save features to database"""
    buffer = io.BytesIO()
    np.save(buffer, features)
    feature_blob = buffer.getvalue()
    
    cursor = db.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO features 
        (song_id, feature_type, feature_vector, dimensions)
        VALUES (?, ?, ?, ?)
    ''', (song_id, 'mert', feature_blob, MERT_DIMS))
    db.commit()

def extract_all():
    """Extract MERT features for all songs"""
    db = sqlite3.connect(DATABASE_PATH)
    extractor = MERTExtractor()
    
    # Get songs without MERT features
    cursor = db.cursor()
    cursor.execute('''
        SELECT s.song_id, s.file_path 
        FROM songs s
        LEFT JOIN features f ON s.song_id = f.song_id AND f.feature_type = 'mert'
        WHERE f.feature_id IS NULL
    ''')
    songs = cursor.fetchall()
    
    print(f"Found {len(songs)} songs to process")
    
    success = 0
    errors = 0
    
    for song_id, file_path in tqdm(songs, desc="Extracting MERT features"):
        if not Path(file_path).exists():
            errors += 1
            continue
        
        features = extractor.extract(file_path)
        if features is not None:
            save_features(db, song_id, features)
            success += 1
        else:
            errors += 1
        
        # Clear GPU cache periodically
        if success % 100 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n✅ MERT extraction complete!")
    print(f"   Success: {success:,}")
    print(f"   Errors: {errors:,}")
    
    db.close()

if __name__ == '__main__':
    extract_all()
