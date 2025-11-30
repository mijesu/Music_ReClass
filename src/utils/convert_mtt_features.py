#!/usr/bin/env python3
import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
from tqdm import tqdm

def extract_features_from_xml(xml_path):
    """Extract audio features from Echonest XML file"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    track = root.find('track')
    
    features = {
        'duration': float(track.get('duration', 0)),
        'loudness': float(track.get('loudness', 0)),
        'tempo': float(track.get('tempo', 0)),
        'tempo_confidence': float(track.get('tempoConfidence', 0)),
        'time_signature': int(track.get('timeSignature', 0)),
        'time_signature_confidence': float(track.get('timeSignatureConfidence', 0)),
        'key': int(track.get('key', 0)),
        'key_confidence': float(track.get('keyConfidence', 0)),
        'mode': int(track.get('mode', 0)),
        'mode_confidence': float(track.get('modeConfidence', 0)),
    }
    
    # Extract segments features (timbre, pitch, loudness)
    segments = root.findall('.//segment')
    timbre_list = []
    pitch_list = []
    loudness_list = []
    
    for seg in segments[:50]:  # Limit to first 50 segments
        timbre = seg.find('timbre')
        if timbre is not None:
            timbre_vals = [float(c.text) for c in timbre.findall('coeff')]
            timbre_list.append(timbre_vals)
        
        pitch = seg.find('pitches')
        if pitch is not None:
            pitch_vals = [float(c.text) for c in pitch.findall('coeff')]
            pitch_list.append(pitch_vals)
        
        loudness_list.append(float(seg.get('loudnessMax', 0)))
    
    # Aggregate segment features
    if timbre_list:
        features['timbre_mean'] = np.mean(timbre_list, axis=0).tolist()
        features['timbre_std'] = np.std(timbre_list, axis=0).tolist()
    if pitch_list:
        features['pitch_mean'] = np.mean(pitch_list, axis=0).tolist()
        features['pitch_std'] = np.std(pitch_list, axis=0).tolist()
    if loudness_list:
        features['loudness_mean'] = np.mean(loudness_list)
        features['loudness_std'] = np.std(loudness_list)
    
    # Flatten to single feature vector
    feature_vector = [
        features['duration'], features['loudness'], features['tempo'],
        features['tempo_confidence'], features['time_signature'],
        features['time_signature_confidence'], features['key'],
        features['key_confidence'], features['mode'], features['mode_confidence']
    ]
    
    if 'timbre_mean' in features:
        feature_vector.extend(features['timbre_mean'])
        feature_vector.extend(features['timbre_std'])
    if 'pitch_mean' in features:
        feature_vector.extend(features['pitch_mean'])
        feature_vector.extend(features['pitch_std'])
    if 'loudness_mean' in features:
        feature_vector.extend([features['loudness_mean'], features['loudness_std']])
    
    return np.array(feature_vector, dtype=np.float32)

def main():
    xml_dir = '/media/mijesu_970/SSD_Data/DataSets/MTT/Data/Echonest_xml'
    output_dir = '/media/mijesu_970/SSD_Data/AI_models/MTT'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Scanning XML files...")
    xml_files = []
    for subdir in os.listdir(xml_dir):
        subdir_path = os.path.join(xml_dir, subdir)
        if os.path.isdir(subdir_path):
            for f in os.listdir(subdir_path):
                if f.endswith('.xml'):
                    xml_files.append(os.path.join(subdir_path, f))
    
    print(f"Found {len(xml_files)} XML files")
    
    features_list = []
    filenames = []
    
    print("Extracting features...")
    for xml_path in tqdm(xml_files):
        try:
            features = extract_features_from_xml(xml_path)
            features_list.append(features)
            filenames.append(os.path.basename(xml_path).replace('.mp3.xml', ''))
        except Exception as e:
            print(f"Error processing {xml_path}: {e}")
            continue
    
    # Convert to numpy array
    features_array = np.array(features_list, dtype=np.float32)
    print(f"Feature array shape: {features_array.shape}")
    
    # Save as .npy
    npy_path = os.path.join(output_dir, 'MTT.npy')
    np.save(npy_path, features_array)
    print(f"Saved: {npy_path} ({os.path.getsize(npy_path) / 1024 / 1024:.2f} MB)")
    
    # Save as .pth with metadata
    pth_path = os.path.join(output_dir, 'MTT.pth')
    torch.save({
        'features': torch.from_numpy(features_array),
        'filenames': filenames,
        'feature_dim': features_array.shape[1],
        'num_samples': features_array.shape[0]
    }, pth_path)
    print(f"Saved: {pth_path} ({os.path.getsize(pth_path) / 1024 / 1024:.2f} MB)")
    
    print("\nConversion complete!")
    print(f"Total samples: {len(features_list)}")
    print(f"Feature dimension: {features_array.shape[1]}")

if __name__ == '__main__':
    main()
