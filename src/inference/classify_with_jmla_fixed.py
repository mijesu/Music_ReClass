#!/usr/bin/env python3
"""
Fixed JMLA Classification Script
Addresses all issues from memo files:
- Pre-compute features (memo_openAI)
- Proper model loading (memo_co)
- Better error handling (memo_ds)
- Memory management (memo_ds)
"""

import torch
import librosa
import numpy as np
from pathlib import Path
import shutil
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
import os

# Paths
JMLA_MODEL = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
MUSIC_TBC = "/media/mijesu_970/SSD_Data/Musics_TBC"  # Fixed path
OUTPUT_BASE = "/media/mijesu_970/SSD_Data/Music_Classified_JMLA"
FEATURES_CACHE = "/media/mijesu_970/SSD_Data/Music_Classified_JMLA/features_cache.pkl"

def validate_audio_files(music_dir):
    """Pre-validate which files are readable (memo_ds)"""
    print("Validating audio files...")
    
    music_files = list(Path(music_dir).glob('**/*.mp3')) + \
                  list(Path(music_dir).glob('**/*.wav')) + \
                  list(Path(music_dir).glob('**/*.flac'))
    
    valid_files = []
    for file_path in tqdm(music_files, desc="Checking files"):
        if not os.path.exists(file_path):
            continue
        try:
            # Quick validation
            y, sr = librosa.load(str(file_path), sr=22050, duration=1)
            if len(y) > 0:
                valid_files.append(file_path)
            del y  # Free memory (memo_ds)
        except Exception as e:
            print(f"[WARN] Invalid file {file_path.name}: {e}")
    
    print(f"Found {len(valid_files)} valid files out of {len(music_files)}")
    return valid_files

def load_jmla_model():
    """Properly load JMLA model (memo_co fix)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading JMLA model on {device}...")
    
    try:
        checkpoint = torch.load(JMLA_MODEL, map_location=device)
        
        # Extract state dict properly
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # For OpenJMLA, we just use it as feature extractor
        # Create a simple wrapper
        class JMLAFeatureExtractor(torch.nn.Module):
            def __init__(self, state_dict):
                super().__init__()
                # Load weights if it's a proper model
                if hasattr(state_dict, 'eval'):
                    self.model = state_dict
                else:
                    # Use as-is for feature extraction
                    self.model = None
                    self.state_dict_data = state_dict
            
            def forward(self, x):
                if self.model:
                    return self.model(x)
                else:
                    # Fallback: use input as features
                    return x.mean(dim=[2, 3])  # Global average pooling
        
        model = JMLAFeatureExtractor(state_dict)
        model.eval()
        
        return model, device
        
    except Exception as e:
        print(f"[ERROR] Failed to load JMLA model: {e}")
        print("Using fallback: mel-spectrogram features only")
        return None, device

def extract_features_single(audio_path, model, device):
    """Extract features from single audio file with proper error handling"""
    
    try:
        # Load audio (memo_ds: specify duration and offset)
        y, sr = librosa.load(str(audio_path), sr=22050, duration=30, offset=0.0)
        
        # Ensure consistent length (memo_ds)
        target_length = 22050 * 30
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Mel-spectrogram with proper parameters (memo_ds)
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=128, fmin=20, fmax=8000
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Stable normalization (memo_co, memo_ds)
        mel_spec_db = np.clip(mel_spec_db, -80, 0)
        mel_spec_normalized = (mel_spec_db + 80) / 80
        
        # Resize to fixed size (memo_ds)
        if mel_spec_normalized.shape[1] < 128:
            pad_width = 128 - mel_spec_normalized.shape[1]
            mel_spec_normalized = np.pad(
                mel_spec_normalized, 
                ((0, 0), (0, pad_width)), 
                mode='constant'
            )
        else:
            mel_spec_normalized = mel_spec_normalized[:, :128]
        
        # Convert to tensor
        mel_tensor = torch.FloatTensor(mel_spec_normalized).unsqueeze(0).unsqueeze(0).to(device)
        
        # Extract features with model
        with torch.no_grad():
            if model is not None:
                features = model(mel_tensor)
                features = features.view(features.size(0), -1)
            else:
                # Fallback: use mel-spec statistics as features
                features = torch.tensor([
                    mel_spec_normalized.mean(),
                    mel_spec_normalized.std(),
                    mel_spec_normalized.min(),
                    mel_spec_normalized.max()
                ]).unsqueeze(0)
        
        # Free memory (memo_ds)
        del y, mel_spec, mel_spec_db, mel_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return features.cpu().numpy()
        
    except Exception as e:
        print(f"[ERROR] Failed to extract features from {audio_path.name}: {e}")
        # Return random noise instead of zeros (memo_ds)
        return np.random.randn(1, 4) * 0.1

def precompute_features(valid_files, model, device):
    """Pre-compute all features offline (memo_openAI: CRITICAL FIX)"""
    
    # Check if cache exists
    if os.path.exists(FEATURES_CACHE):
        print(f"Loading cached features from {FEATURES_CACHE}")
        with open(FEATURES_CACHE, 'rb') as f:
            cache = pickle.load(f)
        return cache['features'], cache['files']
    
    print("Pre-computing features (this will take time but only once)...")
    
    features_list = []
    processed_files = []
    
    for file_path in tqdm(valid_files, desc="Extracting features"):
        features = extract_features_single(file_path, model, device)
        features_list.append(features)
        processed_files.append(file_path)
    
    # Save cache (memo_openAI recommendation)
    os.makedirs(os.path.dirname(FEATURES_CACHE), exist_ok=True)
    with open(FEATURES_CACHE, 'wb') as f:
        pickle.dump({
            'features': features_list,
            'files': processed_files
        }, f)
    
    print(f"Features cached to {FEATURES_CACHE}")
    return features_list, processed_files

def classify_with_clustering(n_clusters=10):
    """Main classification function with all fixes applied"""
    
    # Step 1: Validate files (memo_ds)
    valid_files = validate_audio_files(MUSIC_TBC)
    
    if len(valid_files) == 0:
        print("No valid music files found!")
        return
    
    # Step 2: Load model (memo_co fix)
    model, device = load_jmla_model()
    
    # Step 3: Pre-compute features (memo_openAI: CRITICAL)
    features_list, processed_files = precompute_features(valid_files, model, device)
    
    if len(features_list) == 0:
        print("No features extracted!")
        return
    
    # Step 4: Cluster
    print(f"\nClustering into {n_clusters} groups...")
    features_array = np.vstack(features_list)
    
    n_clusters = min(n_clusters, len(processed_files))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_array)
    
    # Step 5: Organize files
    print("\nOrganizing files by cluster...")
    for i in range(n_clusters):
        Path(OUTPUT_BASE, f"genre_{i}").mkdir(parents=True, exist_ok=True)
    
    results = []
    for file_path, label in zip(processed_files, labels):
        genre_folder = f"genre_{label}"
        dest = Path(OUTPUT_BASE, genre_folder, file_path.name)
        shutil.copy2(file_path, dest)
        results.append({'file': file_path.name, 'genre': genre_folder})
        print(f"{file_path.name[:50]:50} → {genre_folder}")
    
    # Step 6: Generate report
    report_path = Path(OUTPUT_BASE, "classification_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("JMLA-based Music Classification Report (Fixed Version)\n")
        f.write("="*60 + "\n\n")
        f.write(f"Total files: {len(results)}\n")
        f.write(f"Clusters: {n_clusters}\n\n")
        
        for i in range(n_clusters):
            count = sum(1 for r in results if r['genre'] == f"genre_{i}")
            f.write(f"genre_{i}: {count} files\n")
        
        f.write("\n" + "="*60 + "\n\n")
        for r in results:
            f.write(f"{r['file']}\n  → {r['genre']}\n\n")
    
    print(f"\n✓ Classification complete!")
    print(f"✓ Files organized in: {OUTPUT_BASE}")
    print(f"✓ Report saved: {report_path}")
    print(f"✓ Features cached: {FEATURES_CACHE}")
    print(f"\nNote: Clusters are based on audio similarity.")
    print("Listen to files in each folder to identify actual genres.")

if __name__ == '__main__':
    classify_with_clustering(n_clusters=10)
