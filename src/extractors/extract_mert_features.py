#!/usr/bin/env python3
"""
MERT Feature Extraction (768 dims) - STANDALONE VERSION
For quick testing and batch processing without database
Use extract_mert.py for production with database integration
"""

import torch
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, AutoModel
import warnings
warnings.filterwarnings('ignore')


class MERTFeatureExtractor:
    """Extract features using MERT pre-trained model"""
    
    def __init__(self, model_name='m-a-p/MERT-v1-330M', sr=24000, use_cpu=False):
        self.sr = sr
        self.device = torch.device('cpu' if use_cpu else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        print(f"Loading MERT model: {model_name}...")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        
        if not use_cpu:
            try:
                self.model = self.model.to(self.device)
            except:
                print("GPU failed, using CPU")
                self.device = torch.device('cpu')
        
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def extract_features(self, audio_path):
        """Extract features from single audio file"""
        try:
            audio, _ = librosa.load(audio_path, sr=self.sr, duration=30.0)
            inputs = self.processor(audio, sampling_rate=self.sr, return_tensors="pt")
            
            if self.device.type == 'cuda':
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                features = outputs.last_hidden_state.mean(dim=1)
            
            return features.cpu().numpy().flatten().astype(np.float32)
            
        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            return None
    
    def extract_batch(self, audio_dir, output_file='MERT_features.npy', file_extension='.wav'):
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
        
        for audio_file in tqdm(audio_files, desc="Extracting MERT features"):
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
        
        if output_file:
            np.save(output_file, features_array)
            print(f"\nSaved features to {output_file} ({features_array.nbytes / 1024 / 1024:.2f} MB)")
        
        return features_array, filenames


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract MERT features from audio files')
    parser.add_argument('--input', type=str, required=True, help='Input directory with audio files')
    parser.add_argument('--output', type=str, default='MERT_features.npy', help='Output file (.npy)')
    parser.add_argument('--model', type=str, default='m-a-p/MERT-v1-330M', help='MERT model name')
    parser.add_argument('--extension', type=str, default='.wav', help='Audio file extension')
    parser.add_argument('--sample-rate', type=int, default=24000, help='Sample rate')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    extractor = MERTFeatureExtractor(
        model_name=args.model,
        sr=args.sample_rate,
        use_cpu=args.cpu
    )
    
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
