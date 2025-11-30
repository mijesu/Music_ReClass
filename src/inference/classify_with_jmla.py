import torch
import librosa
import numpy as np
from pathlib import Path
import shutil
from sklearn.cluster import KMeans
import pickle

JMLA_MODEL = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
MUSIC_TBC = "/media/mijesu_970/SSD_Data/Music_TBC"
OUTPUT_BASE = "/media/mijesu_970/SSD_Data/Music_Classified_JMLA"

def load_jmla():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(JMLA_MODEL, map_location=device)
    
    # Extract model from checkpoint
    if isinstance(checkpoint, dict):
        model = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        model = checkpoint
    
    if hasattr(model, 'eval'):
        model.eval()
    
    return model, device

def extract_features(model, audio_path, device):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=22050, duration=30)
    
    # Pad/crop
    target_length = 22050 * 30
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    # Mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0).to(device)
    
    # Extract features with JMLA
    with torch.no_grad():
        if hasattr(model, 'forward'):
            features = model(mel_tensor)
        else:
            features = mel_tensor
        
        # Flatten to vector
        features = features.view(features.size(0), -1)
    
    return features.cpu().numpy()

def classify_with_clustering():
    print("Loading JMLA model...")
    model, device = load_jmla()
    
    # Find music files
    music_files = list(Path(MUSIC_TBC).glob('**/*.mp3')) + \
                  list(Path(MUSIC_TBC).glob('**/*.wav')) + \
                  list(Path(MUSIC_TBC).glob('**/*.flac'))
    
    print(f"Found {len(music_files)} files\n")
    
    if len(music_files) == 0:
        print("No music files found in Music_TBC folder!")
        return
    
    # Extract features for all files
    print("Extracting features...")
    features_list = []
    valid_files = []
    
    for i, file_path in enumerate(music_files, 1):
        try:
            features = extract_features(model, str(file_path), device)
            features_list.append(features)
            valid_files.append(file_path)
            print(f"[{i}/{len(music_files)}] {file_path.name[:50]}")
        except Exception as e:
            print(f"[{i}/{len(music_files)}] ERROR: {file_path.name} - {e}")
    
    if len(features_list) == 0:
        print("No valid features extracted!")
        return
    
    # Cluster into 10 groups (genres)
    print("\nClustering into genres...")
    features_array = np.vstack(features_list)
    n_clusters = min(10, len(valid_files))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_array)
    
    # Create output directories
    for i in range(n_clusters):
        Path(OUTPUT_BASE, f"genre_{i}").mkdir(parents=True, exist_ok=True)
    
    # Organize files
    print("\nOrganizing files...")
    results = []
    for file_path, label in zip(valid_files, labels):
        genre_folder = f"genre_{label}"
        dest = Path(OUTPUT_BASE, genre_folder, file_path.name)
        shutil.copy2(file_path, dest)
        results.append({'file': file_path.name, 'genre': genre_folder})
        print(f"{file_path.name[:50]:50} → {genre_folder}")
    
    # Save report
    report_path = Path(OUTPUT_BASE, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("JMLA-based Music Classification Report\n")
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
    print(f"\nNote: Files are grouped by similarity (genre_0 to genre_{n_clusters-1})")
    print("Listen to files in each folder to identify the actual genre")

if __name__ == '__main__':
    classify_with_clustering()
