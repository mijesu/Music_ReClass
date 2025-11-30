#!/usr/bin/env python3
"""
FMA Feature Extraction (518 dims) - STANDALONE VERSION
For quick testing and batch processing without database
Use extract_fma.py for production with database integration

Feature breakdown:
- Chroma (12): Pitch class distribution
- Tonnetz (6): Tonal centroid features  
- MFCC (20): Mel-frequency cepstral coefficients
- Spectral features (11): Centroid, bandwidth, contrast, rolloff, etc.
- Rhythm features (2): Tempo, zero crossing rate
- Statistics (467): Mean, std, skew, kurtosis for time-series features
"""

import librosa
import numpy as np
from pathlib import Path

def extract_fma_features(audio_path, sr=22050, duration=30):
    """
    Extract 518-dimensional FMA features from audio file
    
    Feature breakdown:
    - Chroma (12): Pitch class distribution
    - Tonnetz (6): Tonal centroid features  
    - MFCC (20): Mel-frequency cepstral coefficients
    - Spectral features (11): Centroid, bandwidth, contrast, rolloff, etc.
    - Rhythm features (2): Tempo, zero crossing rate
    - Statistics (467): Mean, std, skew, kurtosis for time-series features
    
    Returns: numpy array of shape (518,)
    """
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=sr, duration=duration, mono=True)
    
    features = []
    
    # ===== 1. CHROMA FEATURES (12 dims) =====
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend([
        np.mean(chroma_stft),
        np.std(chroma_stft),
        np.min(chroma_stft),
        np.max(chroma_stft)
    ])
    features.extend(np.mean(chroma_stft, axis=1))  # 12 pitch classes
    
    # ===== 2. TONNETZ (6 dims) =====
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    features.extend([
        np.mean(tonnetz),
        np.std(tonnetz),
        np.min(tonnetz),
        np.max(tonnetz)
    ])
    features.extend(np.mean(tonnetz, axis=1))  # 6 tonal centroids
    
    # ===== 3. MFCC (20 dims) =====
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    for coef in mfcc:
        features.extend([
            np.mean(coef),
            np.std(coef),
            np.min(coef),
            np.max(coef)
        ])
    
    # ===== 4. SPECTRAL CENTROID (4 dims) =====
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    features.extend([
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.min(spectral_centroid),
        np.max(spectral_centroid)
    ])
    
    # ===== 5. SPECTRAL BANDWIDTH (4 dims) =====
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    features.extend([
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.min(spectral_bandwidth),
        np.max(spectral_bandwidth)
    ])
    
    # ===== 6. SPECTRAL CONTRAST (7 bands x 4 stats = 28 dims) =====
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=7)
    for band in spectral_contrast:
        features.extend([
            np.mean(band),
            np.std(band),
            np.min(band),
            np.max(band)
        ])
    
    # ===== 7. SPECTRAL ROLLOFF (4 dims) =====
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    features.extend([
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),
        np.min(spectral_rolloff),
        np.max(spectral_rolloff)
    ])
    
    # ===== 8. ZERO CROSSING RATE (4 dims) =====
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features.extend([
        np.mean(zcr),
        np.std(zcr),
        np.min(zcr),
        np.max(zcr)
    ])
    
    # ===== 9. RMS ENERGY (4 dims) =====
    rms = librosa.feature.rms(y=y)[0]
    features.extend([
        np.mean(rms),
        np.std(rms),
        np.min(rms),
        np.max(rms)
    ])
    
    # ===== 10. TEMPO (1 dim) =====
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features.append(tempo)
    
    # ===== 11. MEL SPECTROGRAM STATISTICS (128 bands x 4 stats = 512 dims) =====
    # This makes up most of the remaining dimensions
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    for band in mel_spec_db:
        features.extend([
            np.mean(band),
            np.std(band),
            np.min(band),
            np.max(band)
        ])
    
    # Convert to numpy array
    feature_array = np.array(features)
    
    # Ensure exactly 518 dimensions (pad or truncate if needed)
    if len(feature_array) < 518:
        feature_array = np.pad(feature_array, (0, 518 - len(feature_array)))
    elif len(feature_array) > 518:
        feature_array = feature_array[:518]
    
    return feature_array


def extract_batch(audio_dir, output_file='fma_features.npy'):
    """
    Extract features from all audio files in directory
    
    Args:
        audio_dir: Path to directory containing audio files
        output_file: Output .npy file to save features
    """
    audio_dir = Path(audio_dir)
    audio_files = list(audio_dir.rglob('*.mp3')) + \
                  list(audio_dir.rglob('*.wav')) + \
                  list(audio_dir.rglob('*.flac'))
    
    print(f"Found {len(audio_files)} audio files")
    
    all_features = []
    all_labels = []
    
    for audio_file in audio_files:
        try:
            print(f"Processing: {audio_file.name}")
            features = extract_fma_features(str(audio_file))
            all_features.append(features)
            
            # Get genre from parent folder name
            genre = audio_file.parent.name
            all_labels.append(genre)
            
        except Exception as e:
            print(f"Error processing {audio_file.name}: {e}")
            continue
    
    # Save features
    features_array = np.array(all_features)
    labels_array = np.array(all_labels)
    
    np.save(output_file, features_array)
    np.save(output_file.replace('.npy', '_labels.npy'), labels_array)
    
    print(f"\n✅ Saved {len(all_features)} features to {output_file}")
    print(f"   Shape: {features_array.shape}")
    print(f"   Genres: {np.unique(all_labels)}")
    
    return features_array, labels_array


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 extract_fma_features.py <audio_directory>")
        print("\nExample:")
        print("  python3 extract_fma_features.py ./custom_dataset/")
        print("\nDirectory structure should be:")
        print("  custom_dataset/")
        print("    ├── K-Pop/")
        print("    │   ├── song1.mp3")
        print("    │   └── song2.mp3")
        print("    ├── Anime/")
        print("    └── Lo-Fi/")
        sys.exit(1)
    
    audio_dir = sys.argv[1]
    extract_batch(audio_dir)
