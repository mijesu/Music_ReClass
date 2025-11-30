import torch
import librosa
import numpy as np
from pathlib import Path
import shutil

# Paths
JMLA_MODEL = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
GTZAN_MODEL = "/media/mijesu_970/SSD_Data/AI_models/ZTGAN/GTZAN.pth"
MUSIC_TBC = "/media/mijesu_970/SSD_Data/Music_TBC"
OUTPUT_BASE = "/media/mijesu_970/SSD_Data/Music_Classified"
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class GTZANClassifier(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = torch.nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

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

def phase1_jmla(file_path, jmla_model, device):
    """Phase 1: JMLA feature extraction"""
    mel_spec = process_audio(file_path).to(device)
    with torch.no_grad():
        if hasattr(jmla_model, 'forward'):
            features = jmla_model(mel_spec)
        else:
            features = mel_spec
        features = features.view(features.size(0), -1)
    return features

def phase2_gtzan(features, mel_spec, gtzan_model, device):
    """Phase 2: GTZAN genre classification"""
    with torch.no_grad():
        output = gtzan_model(mel_spec.to(device))
        pred_idx = output.argmax(1).item()
        confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
    return pred_idx, confidence

def classify_two_phase():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load Phase 1: JMLA
    print("Loading Phase 1: JMLA model...")
    jmla_checkpoint = torch.load(JMLA_MODEL, map_location=device)
    if isinstance(jmla_checkpoint, dict):
        jmla_model = jmla_checkpoint.get('model', jmla_checkpoint.get('state_dict', jmla_checkpoint))
    else:
        jmla_model = jmla_checkpoint
    if hasattr(jmla_model, 'eval'):
        jmla_model.eval()
    
    # Load Phase 2: GTZAN
    print("Loading Phase 2: GTZAN model...")
    gtzan_model = GTZANClassifier(num_classes=len(GENRES)).to(device)
    gtzan_model.load_state_dict(torch.load(GTZAN_MODEL, map_location=device, weights_only=False))
    gtzan_model.eval()
    
    # Create output directories
    for genre in GENRES:
        Path(OUTPUT_BASE, genre).mkdir(parents=True, exist_ok=True)
    
    # Find music files
    music_files = list(Path(MUSIC_TBC).glob('**/*.mp3')) + \
                  list(Path(MUSIC_TBC).glob('**/*.wav')) + \
                  list(Path(MUSIC_TBC).glob('**/*.flac'))
    
    print(f"\nFound {len(music_files)} files to classify\n")
    print("="*70)
    
    results = []
    for i, file_path in enumerate(music_files, 1):
        try:
            # Phase 1: JMLA feature extraction
            mel_spec = process_audio(str(file_path))
            jmla_features = phase1_jmla(str(file_path), jmla_model, device)
            
            # Phase 2: GTZAN classification
            pred_idx, confidence = phase2_gtzan(jmla_features, mel_spec, gtzan_model, device)
            genre = GENRES[pred_idx]
            
            # Copy to classified folder
            dest = Path(OUTPUT_BASE, genre, file_path.name)
            shutil.copy2(file_path, dest)
            
            results.append({
                'file': file_path.name,
                'genre': genre,
                'confidence': confidence
            })
            
            print(f"[{i}/{len(music_files)}] {file_path.name[:45]:45} → {genre:10} ({confidence:.1%})")
            
        except Exception as e:
            print(f"[{i}/{len(music_files)}] ERROR: {file_path.name} - {e}")
    
    # Save report
    print("\n" + "="*70)
    report_path = Path(OUTPUT_BASE, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Two-Phase Music Classification Report\n")
        f.write("Phase 1: JMLA Feature Extraction\n")
        f.write("Phase 2: GTZAN Genre Classification\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total files: {len(results)}\n\n")
        
        # Genre summary
        genre_counts = {}
        for r in results:
            genre_counts[r['genre']] = genre_counts.get(r['genre'], 0) + 1
        
        f.write("Genre Distribution:\n")
        for genre in GENRES:
            count = genre_counts.get(genre, 0)
            f.write(f"  {genre:12} : {count:3} files\n")
        
        f.write("\n" + "="*70 + "\n\n")
        f.write("Detailed Results:\n\n")
        for r in results:
            f.write(f"{r['file']}\n")
            f.write(f"  Genre: {r['genre']}\n")
            f.write(f"  Confidence: {r['confidence']:.1%}\n\n")
    
    print(f"\n✓ Classification complete!")
    print(f"✓ Files organized in: {OUTPUT_BASE}")
    print(f"✓ Report saved: {report_path}")
    print(f"\nGenre Distribution:")
    for genre in GENRES:
        count = genre_counts.get(genre, 0)
        print(f"  {genre:12} : {count:3} files")

if __name__ == '__main__':
    classify_two_phase()
