import torch
import librosa
import numpy as np
from pathlib import Path
import shutil
from sklearn.cluster import KMeans

JMLA_MODEL = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
MUSIC_TBC = "/media/mijesu_970/SSD_Data/Musics_TBC"
OUTPUT_BASE = "/media/mijesu_970/SSD_Data/Music_Classified"
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

def process_audio(file_path, sr=22050, duration=30):
    audio, _ = librosa.load(file_path, sr=sr, duration=duration)
    target_length = sr * duration
    
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]
    
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)

def extract_features(model, file_path, device):
    mel_spec = process_audio(file_path).to(device)
    with torch.no_grad():
        if hasattr(model, 'forward'):
            features = model(mel_spec)
        else:
            features = mel_spec
        features = features.view(features.size(0), -1)
    return features.cpu().numpy()

def classify():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load JMLA
    print("Loading JMLA model...")
    checkpoint = torch.load(JMLA_MODEL, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict):
        model = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
    else:
        model = checkpoint
    if hasattr(model, 'eval'):
        model.eval()
    
    # Find music files
    music_files = list(Path(MUSIC_TBC).glob('**/*.mp3')) + \
                  list(Path(MUSIC_TBC).glob('**/*.wav')) + \
                  list(Path(MUSIC_TBC).glob('**/*.flac'))
    
    if len(music_files) == 0:
        print("No music files found in Music_TBC folder!")
        return
    
    print(f"Found {len(music_files)} files\n")
    print("="*70)
    
    # Extract features
    print("Extracting features with JMLA...")
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
    
    # Cluster into 10 genres
    print("\n" + "="*70)
    print("Clustering into 10 genres...")
    features_array = np.vstack(features_list)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_array)
    
    # Map clusters to genre names
    for i in range(10):
        Path(OUTPUT_BASE, GENRES[i]).mkdir(parents=True, exist_ok=True)
    
    # Organize files
    print("\n" + "="*70)
    print("Organizing files...")
    results = []
    for file_path, label in zip(valid_files, labels):
        genre = GENRES[label]
        dest = Path(OUTPUT_BASE, genre, file_path.name)
        shutil.copy2(file_path, dest)
        results.append({'file': file_path.name, 'genre': genre})
        print(f"{file_path.name[:50]:50} → {genre}")
    
    # Save report
    print("\n" + "="*70)
    report_path = Path(OUTPUT_BASE, "classification_report.txt")
    genre_counts = {}
    for r in results:
        genre_counts[r['genre']] = genre_counts.get(r['genre'], 0) + 1
    
    with open(report_path, 'w') as f:
        f.write("JMLA-based Music Classification Report\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total files: {len(results)}\n\n")
        f.write("Genre Distribution:\n")
        for genre in GENRES:
            count = genre_counts.get(genre, 0)
            f.write(f"  {genre:12} : {count:3} files\n")
        f.write("\n" + "="*70 + "\n\n")
        for r in results:
            f.write(f"{r['file']}\n  → {r['genre']}\n\n")
    
    print(f"\n✓ Classification complete!")
    print(f"✓ Files organized in: {OUTPUT_BASE}")
    print(f"✓ Report saved: {report_path}")
    print(f"\nGenre Distribution:")
    for genre in GENRES:
        count = genre_counts.get(genre, 0)
        print(f"  {genre:12} : {count:3} files")

if __name__ == '__main__':
    classify()
