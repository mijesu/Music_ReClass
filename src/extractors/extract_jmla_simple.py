#!/usr/bin/env python3
"""
Simplified JMLA feature extraction - TEXT-BASED VERSION
Converts JMLA text output to feature vectors
Alternative approach using text generation instead of embeddings
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

class JMLASimpleExtractor:
    def __init__(self):
        print("Loading JMLA model...")
        self.model = AutoModel.from_pretrained('UniMus/OpenJMLA', trust_remote_code=True)
        self.model.to(DEVICE)
        self.model.eval()
        self.sr = 16000
        
        # Genre keywords for text parsing
        self.genre_keywords = {
            'Blues': ['blues', 'soul', 'r&b'],
            'Classical': ['classical', 'orchestra', 'symphony', 'piano', 'violin'],
            'Country': ['country', 'folk', 'acoustic'],
            'Electronic': ['electronic', 'edm', 'techno', 'house', 'synth'],
            'Folk': ['folk', 'acoustic', 'traditional'],
            'Hip-Hop': ['hip-hop', 'rap', 'hip hop'],
            'Jazz': ['jazz', 'swing', 'bebop'],
            'Metal': ['metal', 'rock', 'heavy'],
            'Pop': ['pop', 'contemporary'],
            'Rock': ['rock', 'guitar'],
            'K-Pop': ['k-pop', 'korean', 'kpop'],
            'Anime': ['anime', 'japanese', 'j-pop'],
            'Lo-Fi': ['lo-fi', 'lofi', 'chill']
        }
        print(f"JMLA model loaded on {DEVICE}")
    
    def wav_to_mel(self, audio_path):
        """Convert audio to log-mel spectrogram"""
        y, sr = librosa.load(audio_path, sr=self.sr, duration=30, mono=True)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=1024, hop_length=160, n_mels=128
        )
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel
    
    def normalize(self, lms):
        """Normalize spectrogram"""
        return np.clip((lms + 4.5) / 9.0, 0, 1)
    
    def pad_or_crop(self, lms, target_len=2992):
        """Pad or crop to target length"""
        if lms.shape[1] < target_len:
            lms = np.pad(lms, ((0, 0), (0, target_len - lms.shape[1])))
        else:
            lms = lms[:, :target_len]
        return lms
    
    def text_to_features(self, text):
        """Convert JMLA text output to feature vector"""
        text = text.lower()
        
        # Create feature vector based on genre keywords
        features = np.zeros(JMLA_DIMS)
        
        # First 16 dims: genre probabilities
        for i, (genre, keywords) in enumerate(self.genre_keywords.items()):
            if i < 16:
                score = sum(1 for kw in keywords if kw in text)
                features[i] = score
        
        # Normalize genre scores
        if features[:16].sum() > 0:
            features[:16] = features[:16] / features[:16].sum()
        
        # Remaining dims: text embedding (simple hash-based)
        words = text.split()
        for i, word in enumerate(words[:100]):
            idx = 16 + (hash(word) % (JMLA_DIMS - 16))
            features[idx] += 0.1
        
        return features
    
    def extract(self, audio_path):
        """Extract JMLA features via text generation"""
        try:
            # Get mel spectrogram
            lms = self.wav_to_mel(audio_path)
            lms = self.normalize(lms)
            lms = self.pad_or_crop(lms)
            
            # Prepare input
            lms_tensor = torch.from_numpy(lms).unsqueeze(0).float().to(DEVICE)
            
            input_dict = {
                'filenames': [Path(audio_path).name],
                'ans_crds': [0],
                'audio_crds': [0],
                'attention_mask': torch.tensor([[1, 1, 1, 1, 1]]).to(DEVICE),
                'input_ids': torch.tensor([[1, 694, 5777, 683, 13]]).to(DEVICE),
                'spectrogram': lms_tensor
            }
            
            # Generate text description
            with torch.no_grad():
                gen_ids = self.model.forward_test(input_dict)
                gen_text = self.model.neck.tokenizer.batch_decode(gen_ids.clip(0))[0]
                
                # Post-process
                gen_text = gen_text.split('<s>')[-1].split('\n')[0].strip()
                gen_text = gen_text.replace(' in Chinese', '').replace(' Chinese', '')
            
            # Convert text to features
            features = self.text_to_features(gen_text)
            
            return features
            
        except Exception as e:
            print(f"Error: {e}")
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
    extractor = JMLASimpleExtractor()
    
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
        
        if success % 100 == 0:
            torch.cuda.empty_cache()
    
    print(f"\n✅ JMLA extraction complete!")
    print(f"   Success: {success:,}")
    print(f"   Errors: {errors:,}")
    
    db.close()

if __name__ == '__main__':
    extract_all()
