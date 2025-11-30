#!/usr/bin/env python3
"""
JMLA Feature Extraction (768 dims) - STANDALONE VERSION
For quick testing and batch processing without database
Use extract_jmla.py for production with database integration
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class JMLAFeatureExtractor:
    """Extract features using OpenJMLA pre-trained model"""
    
    def __init__(self, model_path, sr=22050, duration=30, device=None):
        self.sr = sr
        self.duration = duration
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading JMLA model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            self.model = checkpoint.get('model', checkpoint.get('state_dict', checkpoint))
        else:
            self.model = checkpoint
            
        if hasattr(self.model, 'eval'):
            self.model.eval()
            self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def process_audio(self, audio_path):
        """Convert audio to mel spectrogram"""
        audio, _ = librosa.load(audio_path, sr=self.sr, duration=self.duration)
        target_length = self.sr * self.duration
        
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return torch.FloatTensor(mel_spec_db).unsqueeze(0).unsqueeze(0)
    
    def extract_features(self, audio_path):
        """Extract features from single audio file"""
        try:
            mel_spec = self.process_audio(audio_path).to(self.device)
            
            with torch.no_grad():
                if hasattr(self.model, 'forward'):
                    features = self.model(mel_spec)
                else:
                    features = mel_spec
                features = features.view(features.size(0), -1)
            
            return features.cpu().numpy().flatten().astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def extract_batch(self, audio_dir, output_file='JMLA_features.npy', file_extension='.wav'):
        """Extract features from all audio files in directory"""
        audio_dir = Path(audio_dir)
        audio_files = sorted(audio_dir.rglob(f'*{file_extension}'))
        
        if len(audio_files) == 0:
            print(f"No {file_extension} files found in {audio_dir}")
            return None, None
        
        print(f"Found {len(audio_files)} audio files")
        
        features_list = []
        filenames = []
        feature_dim = None
        
        for audio_file in tqdm(audio_files, desc="Extracting JMLA features"):
            features = self.extract_features(str(audio_file))
            
            if features is not None:
                if feature_dim is None:
                    feature_dim = len(features)
                    print(f"Feature dimension: {feature_dim}")
                
                features_list.append(features)
                filenames.append(audio_file.name)
        
        if len(features_list) == 0:
            print("No features extracted!")
            return None, None
        
        features_array = np.array(features_list)
        
        # Save
        if output_file:
            np.save(output_file, features_array)
            print(f"\nSaved features to {output_file} ({features_array.nbytes / 1024 / 1024:.2f} MB)")
        
        return features_array, filenames


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract JMLA features from audio files')
    parser.add_argument('--input', type=str, required=True, help='Input directory with audio files')
    parser.add_argument('--output', type=str, default='JMLA_features.npy', help='Output file (.npy)')
    parser.add_argument('--model', type=str, default='/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth', 
                        help='Path to JMLA model')
    parser.add_argument('--extension', type=str, default='.wav', help='Audio file extension')
    parser.add_argument('--sample-rate', type=int, default=22050, help='Sample rate')
    parser.add_argument('--duration', type=int, default=30, help='Audio duration in seconds')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = JMLAFeatureExtractor(
        model_path=args.model,
        sr=args.sample_rate,
        duration=args.duration
    )
    
    # Extract features
    features, filenames = extractor.extract_batch(
        audio_dir=args.input,
        output_file=args.output,
        file_extension=args.extension
    )
    
    if features is not None:
        print(f"\nExtraction complete!")
        print(f"Features shape: {features.shape}")
        print(f"Feature dimension: {features.shape[1]}")
        print(f"Number of files: {features.shape[0]}")
        
        print(f"\nFeature statistics:")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")
        print(f"  Min: {features.min():.4f}")
        print(f"  Max: {features.max():.4f}")


if __name__ == '__main__':
    main()
