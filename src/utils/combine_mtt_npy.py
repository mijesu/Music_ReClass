#!/usr/bin/env python3
import numpy as np
import pandas as pd
import os

# Paths
annotation_path = '/media/mijesu_970/SSD_Data/DataSets/MTT/annotations_final.csv'
features_path = '/media/mijesu_970/SSD_Data/AI_models/MTT/MTT.npy'
output_path = '/media/mijesu_970/SSD_Data/AI_models/MTT/MTT_combined.npz'

# Load features
features = np.load(features_path)
print(f"Features shape: {features.shape}")

# Load annotations
df = pd.read_csv(annotation_path, sep='\t')
print(f"Annotations shape: {df.shape}")

# Extract tag columns (all except clip_id and mp3_path)
tag_columns = [col for col in df.columns if col not in ['clip_id', 'mp3_path']]
tags = df[tag_columns].values
print(f"Tags shape: {tags.shape}")
print(f"Number of tags: {len(tag_columns)}")

# Save combined as compressed npz
np.savez_compressed(output_path,
                    features=features,
                    tags=tags,
                    tag_names=tag_columns,
                    clip_ids=df['clip_id'].values if 'clip_id' in df.columns else None)

print(f"\nSaved: {output_path}")
print(f"Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
print(f"\nContents:")
print(f"  - features: {features.shape}")
print(f"  - tags: {tags.shape}")
print(f"  - tag_names: {len(tag_columns)} tags")
