#!/usr/bin/env python3
"""
Extract FMA features (518 dims) - DATABASE VERSION
For production use with SQLite database integration
Use extract_fma_features.py for standalone/testing
"""

import librosa
import numpy as np
import sqlite3
from pathlib import Path
from tqdm import tqdm
import io
from src.utils.config import DATABASE_PATH, SAMPLE_RATE, DURATION, FMA_DIMS

def extract_fma_features(audio_path, sr=22050, duration=30):
    """Extract 518-dimensional FMA features"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
        
        features = []
        
        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend([np.mean(chroma), np.std(chroma), np.min(chroma), np.max(chroma)])
        features.extend(np.mean(chroma, axis=1))
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
        features.extend([np.mean(tonnetz), np.std(tonnetz), np.min(tonnetz), np.max(tonnetz)])
        features.extend(np.mean(tonnetz, axis=1))
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for coef in mfcc:
            features.extend([np.mean(coef), np.std(coef), np.min(coef), np.max(coef)])
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid), 
                        np.min(spectral_centroid), np.max(spectral_centroid)])
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth),
                        np.min(spectral_bandwidth), np.max(spectral_bandwidth)])
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=7)
        for band in spectral_contrast:
            features.extend([np.mean(band), np.std(band), np.min(band), np.max(band)])
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff),
                        np.min(spectral_rolloff), np.max(spectral_rolloff)])
        
        # Rhythm
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.extend([np.mean(zcr), np.std(zcr), np.min(zcr), np.max(zcr)])
        
        rms = librosa.feature.rms(y=y)[0]
        features.extend([np.mean(rms), np.std(rms), np.min(rms), np.max(rms)])
        
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(tempo)
        
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        for band in mel_spec_db:
            features.extend([np.mean(band), np.std(band), np.min(band), np.max(band)])
        
        # Ensure 518 dims
        feature_array = np.array(features)
        if len(feature_array) < FMA_DIMS:
            feature_array = np.pad(feature_array, (0, FMA_DIMS - len(feature_array)))
        elif len(feature_array) > FMA_DIMS:
            feature_array = feature_array[:FMA_DIMS]
        
        return feature_array
        
    except Exception as e:
        print(f"Error extracting features: {e}")
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
    ''', (song_id, 'fma', feature_blob, FMA_DIMS))
    db.commit()

def extract_all():
    """Extract FMA features for all songs"""
    db = sqlite3.connect(DATABASE_PATH)
    
    # Get unprocessed songs
    cursor = db.cursor()
    cursor.execute('''
        SELECT s.song_id, s.file_path 
        FROM songs s
        LEFT JOIN features f ON s.song_id = f.song_id AND f.feature_type = 'fma'
        WHERE f.feature_id IS NULL
    ''')
    songs = cursor.fetchall()
    
    print(f"Found {len(songs)} songs to process")
    
    success = 0
    errors = 0
    
    for song_id, file_path in tqdm(songs, desc="Extracting FMA features"):
        if not Path(file_path).exists():
            errors += 1
            continue
        
        features = extract_fma_features(file_path)
        if features is not None:
            save_features(db, song_id, features)
            success += 1
        else:
            errors += 1
    
    print(f"\n✅ FMA extraction complete!")
    print(f"   Success: {success:,}")
    print(f"   Errors: {errors:,}")
    
    db.close()

if __name__ == '__main__':
    extract_all()
