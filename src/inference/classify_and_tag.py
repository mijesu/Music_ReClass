#!/usr/bin/env python3
"""Classify music with 3 models and write results to ID3 tags"""

import torch
import torchaudio
import eyed3
from pathlib import Path

class MultiModelClassifier:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        
    def load_models(self, model_paths):
        """Load all 3 models"""
        for name, path in model_paths.items():
            self.models[name] = torch.load(path, map_location=self.device)
            self.models[name].eval()
    
    def classify(self, audio_path):
        """Get predictions from all 3 models"""
        results = {}
        
        for model_name, model in self.models.items():
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            if sr != 22050:
                waveform = torchaudio.transforms.Resample(sr, 22050)(waveform)
            
            # Extract features and predict
            mel_spec = torchaudio.transforms.MelSpectrogram()(waveform)
            with torch.no_grad():
                output = model(mel_spec.unsqueeze(0).to(self.device))
                probs = torch.softmax(output, dim=1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0][pred_idx].item()
            
            results[model_name] = {
                'genre': self.get_genre_name(model_name, pred_idx),
                'confidence': confidence
            }
        
        return results
    
    def get_genre_name(self, model_name, idx):
        """Map index to genre name based on model"""
        genres = {
            'msd': ['Blues', 'Country', 'Electronic', 'Folk', 'International', 
                    'Jazz', 'Latin', 'New Age', 'Pop_Rock', 'Rap', 'Reggae', 'RnB', 'Vocal'],
            'gtzan': ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 
                      'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock'],
            'ensemble': ['Blues', 'Classical', 'Country', 'Electronic', 'Folk', 
                         'Hip-Hop', 'Jazz', 'Pop', 'Rock']
        }
        return genres.get(model_name, [])[idx] if idx < len(genres.get(model_name, [])) else 'Unknown'
    
    def write_tags(self, audio_path, results):
        """Write classification results to ID3 tags"""
        audio = eyed3.load(audio_path)
        if audio.tag is None:
            audio.initTag()
        
        # Primary genre (highest confidence)
        best = max(results.items(), key=lambda x: x[1]['confidence'])
        audio.tag.genre = best[1]['genre']
        
        # All predictions in comments
        comment = "\n".join([
            f"{name}: {r['genre']} ({r['confidence']:.1%})"
            for name, r in results.items()
        ])
        audio.tag.comments.set(comment, description="AI Multi-Model")
        
        audio.tag.save()

def main():
    classifier = MultiModelClassifier()
    
    # Load your 3 models
    classifier.load_models({
        'msd': 'models/msd_model.pth',
        'gtzan': 'models/best_model.pth',
        'ensemble': 'models/ensemble_model.pth'
    })
    
    # Classify and tag all music files
    music_dir = Path('Music_TBC')
    for audio_file in music_dir.glob('**/*.mp3'):
        print(f"Processing: {audio_file.name}")
        results = classifier.classify(str(audio_file))
        classifier.write_tags(str(audio_file), results)
        
        # Print results
        for model, pred in results.items():
            print(f"  {model}: {pred['genre']} ({pred['confidence']:.1%})")

if __name__ == "__main__":
    main()
