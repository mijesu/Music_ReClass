#!/usr/bin/env python3
import os
import numpy as np
import torch
import pandas as pd

# Paths
mtt_dir = '/media/mijesu_970/SSD_Data/DataSets/MTT'
features_dir = '/media/mijesu_970/SSD_Data/AI_models/MTT'

# Find annotation file
annotation_files = []
for root, dirs, files in os.walk(mtt_dir):
    for f in files:
        if 'annotation' in f.lower() or 'tag' in f.lower():
            if f.endswith('.csv') or f.endswith('.txt'):
                annotation_files.append(os.path.join(root, f))

print(f"Found {len(annotation_files)} annotation files:")
for f in annotation_files:
    print(f"  - {f}")

if not annotation_files:
    print("No annotation files found. Please specify the path.")
    exit(1)

# Use the first annotation file
annotation_path = annotation_files[0]
print(f"\nUsing: {annotation_path}")

# Load annotations
df = pd.read_csv(annotation_path, sep='\t' if annotation_path.endswith('.txt') else ',')
print(f"Annotation shape: {df.shape}")
print(f"Columns: {list(df.columns)[:10]}...")

# Load existing features
features_pth = torch.load(os.path.join(features_dir, 'MTT.pth'))
features = features_pth['features']
filenames = features_pth['filenames']

print(f"\nFeatures shape: {features.shape}")
print(f"Number of files: {len(filenames)}")

# Create combined dataset
combined_data = {
    'features': features,
    'filenames': filenames,
    'annotations': df,
    'feature_dim': features.shape[1],
    'num_samples': features.shape[0],
    'num_tags': len(df.columns) - 1 if 'clip_id' in df.columns else len(df.columns)
}

# Save combined
output_path = os.path.join(features_dir, 'MTT_combined.pth')
torch.save(combined_data, output_path)
print(f"\nSaved combined data: {output_path}")
print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
