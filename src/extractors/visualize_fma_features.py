#!/usr/bin/env python3
"""
Visualize FMA Feature Structure - Analysis tool
Creates bar chart and pie chart showing 518-dimensional FMA feature breakdown
Outputs: fma_features_518dims.png
"""

import matplotlib.pyplot as plt
import numpy as np

# FMA Feature Structure (518 dimensions total)
features = {
    'Chroma': 12,
    'Tonnetz': 6,
    'MFCC': 20,
    'Spectral Centroid': 1,
    'Spectral Bandwidth': 1,
    'Spectral Contrast': 7,
    'Spectral Rolloff': 1,
    'Zero Crossing Rate': 1,
    'RMS Energy': 1,
    'Tempo': 1,
    'Statistics (mean/std/skew/kurtosis)': 467  # 4x stats for each feature
}

# Create visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
names = list(features.keys())
dims = list(features.values())
colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

ax1.barh(names, dims, color=colors)
ax1.set_xlabel('Number of Dimensions', fontsize=12)
ax1.set_title('FMA Feature Dimensions (Total: 518)', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

for i, (name, dim) in enumerate(zip(names, dims)):
    ax1.text(dim + 5, i, str(dim), va='center', fontsize=10)

# Pie chart
ax2.pie(dims, labels=names, autopct='%1.1f%%', colors=colors, startangle=90)
ax2.set_title('FMA Feature Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('fma_features_518dims.png', dpi=150, bbox_inches='tight')
print("✅ Saved: fma_features_518dims.png")
print(f"\nTotal dimensions: {sum(dims)}")
