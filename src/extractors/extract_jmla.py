#!/usr/bin/env python3
"""
Extract JMLA features (768 dims) - DATABASE VERSION
For production use with SQLite database integration
Use extract_jmla_features.py for standalone/testing
"""

import torch
import librosa
import numpy as np
import sqlite3
from pathlib import Path
from tqdm import tqdm
import io
from transformers import AutoModel
from src.utils.config import DATABASE_PATH, JMLA_DIMS, DEVICE

class JMLAExtractor:
    def __init__(self):
        print("Loading JMLA model...")
        self.model = AutoModel.from_pretrained('UniMus/OpenJMLA', trust_remote_code=True)
        self.model.to(DEVICE)
        self.model.eval()
        self.sr = 16000  # JMLA requires 16kHz
        print(f"JMLA model loaded on {DEVICE}")
    
    def wav_to_mel(self, audio_path):
        """Convert audio to log-mel spectrogram"""
        y, sr = librosa.load(audio_path, sr=self.sr, duration=30, mono=True)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=160, n_mels=128
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        return log_mel
    
    def normalize(self, lms, min_val=-4.5, max_val=4.5):
        """Normalize spectrogram"""
        return np.clip((lms - min_val) / (max_val - min_val), 0, 1)
    
    def pad_or_crop(self, lms, target_len=2992):
        """Pad or crop to target length"""
        if lms.shape[1] < target_len:
            pad_width = target_len - lms.shape[1]
            lms = np.pad(lms, ((0, 0), (0, pad_width)), mode='constant')
        else:
            lms = lms[:, :target_len]
        return lms
    
    def extract(self, audio_path):
        """Extract JMLA features"""
        try:
            # Get mel spectrogram
            lms = self.wav_to_mel(audio_path)
            lms = self.normalize(lms)
            lms = self.pad_or_crop(lms)
            
            # Prepare input dictionary (matching OpenJMLA format)
            lms_tensor = torch.from_numpy(lms).unsqueeze(0).float().to(DEVICE)
            
            input_dict = {
                'filenames': [Path(audio_path).name],
                'ans_crds': [0],
                'audio_crds': [0],
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]).to(DEVICE),
                'input_ids': torch.tensor([[1, 694, 5777, 683, 13]]).to(DEVICE),
                'spectrogram': lms_tensor
            }
            
            # Extract features from encoder
            with torch.no_grad():
                # Access the audio encoder directly
                if hasattr(self.model, 'audio_encoder'):
                    encoder_output = self.model.audio_encoder(lms_tensor)
                elif hasattr(self.model, 'encoder'):
                    encoder_output = self.model.encoder(lms_tensor)
                else:
                    # Fallback: use forward pass and extract hidden states
                    outputs = self.model(input_dict, output_hidden_states=True)
                    if hasattr(outputs, 'encoder_hidden_states'):
                        encoder_output = outputs.encoder_hidden_states[-1]
                    else:
                        # Last resort: use the model's internal representation
                        encoder_output = self.model.forward(input_dict)
                
                # Average pool to get fixed-size features
                if len(encoder_output.shape) == 3:  # [batch, seq, dim]
                    features = encoder_output.mean(dim=1).squeeze()
                else:
                    features = encoder_output.squeeze()
            
            # Ensure 768 dims
            if features.shape[0] != JMLA_DIMS:
                if features.shape[0] > JMLA_DIMS:
                    features = features[:JMLA_DIMS]
                else:
                    features = torch.nn.functional.pad(features, (0, JMLA_DIMS - features.shape[0]))
            
            return features.cpu().numpy()
            
        except Exception as e:
            print(f"Error extracting JMLA features: {e}")
            import traceback
            traceback.print_exc()
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
    ''', (song_id, 'jmla', feature_blob, JMLA_DIMS))
    db.commit()

def extract_all():
    """Extract JMLA features for all songs"""
    db = sqlite3.connect(DATABASE_PATH)
    extractor = JMLAExtractor()
    
    # Get songs without JMLA features
    cursor = db.cursor()
    cursor.execute('''
        SELECT s.song_id, s.file_path 
        FROM songs s
        LEFT JOIN features f ON s.song_id = f.song_id AND f.feature_type = 'jmla'
        WHERE f.feature_id IS NULL
    ''')
    songs = cursor.fetchall()
    
    print(f"Found {len(songs)} songs to process")
    
    success = 0
    errors = 0
    
    for song_id, file_path in tqdm(songs, desc="Extracting JMLA features"):
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
    
    print(f"\n✅ JMLA extraction complete!")
    print(f"   Success: {success:,}")
    print(f"   Errors: {errors:,}")
    
    db.close()

if __name__ == '__main__':
    extract_all()
