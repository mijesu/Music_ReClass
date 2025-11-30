import torch
import librosa
import numpy as np
from pathlib import Path
import shutil

MODEL_PATH = "best_model.pth"  # or genre_classifier.pth
MUSIC_TBC = "/media/mijesu_970/SSD_Data/Music_TBC"
OUTPUT_BASE = "/media/mijesu_970/SSD_Data/Music_Classified"
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

class GenreClassifier(torch.nn.Module):
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

def classify_music_tbc():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = GenreClassifier(num_classes=len(GENRES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Create output directories
    for genre in GENRES:
        Path(OUTPUT_BASE, genre).mkdir(parents=True, exist_ok=True)
    
    # Process files
    music_files = list(Path(MUSIC_TBC).glob('**/*.mp3')) + \
                  list(Path(MUSIC_TBC).glob('**/*.wav')) + \
                  list(Path(MUSIC_TBC).glob('**/*.flac'))
    
    print(f"Found {len(music_files)} files to classify\n")
    
    results = []
    for i, file_path in enumerate(music_files, 1):
        try:
            # Process and predict
            mel_spec = process_audio(str(file_path)).to(device)
            with torch.no_grad():
                output = model(mel_spec)
                pred_idx = output.argmax(1).item()
                confidence = torch.softmax(output, dim=1)[0][pred_idx].item()
            
            genre = GENRES[pred_idx]
            
            # Copy to classified folder
            dest = Path(OUTPUT_BASE, genre, file_path.name)
            shutil.copy2(file_path, dest)
            
            results.append({
                'file': file_path.name,
                'genre': genre,
                'confidence': f"{confidence:.2%}"
            })
            
            print(f"[{i}/{len(music_files)}] {file_path.name[:40]:40} → {genre:12} ({confidence:.2%})")
            
        except Exception as e:
            print(f"[{i}/{len(music_files)}] ERROR: {file_path.name} - {e}")
    
    # Save report
    report_path = Path(OUTPUT_BASE, "classification_report.txt")
    with open(report_path, 'w') as f:
        f.write("Music Classification Report\n")
        f.write("="*60 + "\n\n")
        for r in results:
            f.write(f"{r['file']}\n  Genre: {r['genre']}\n  Confidence: {r['confidence']}\n\n")
    
    print(f"\n✓ Classification complete!")
    print(f"✓ Files organized in: {OUTPUT_BASE}")
    print(f"✓ Report saved: {report_path}")

if __name__ == '__main__':
    classify_music_tbc()
