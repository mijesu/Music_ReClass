#!/usr/bin/env python3
"""Quick baseline using XGBoost - runs in minutes"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time

FEATURES_PATH = "/media/mijesu_970/SSD_Data/datasets/FMA/Misc/fma_metadata/features.csv"
TRACKS_PATH = "/media/mijesu_970/SSD_Data/datasets/FMA/Misc/fma_metadata/tracks.csv"

print("Quick Baseline - XGBoost on FMA")
print("="*50)

start = time.time()

# Load
print("Loading data...")
features = pd.read_csv(FEATURES_PATH, index_col=0, header=[0, 1, 2])
tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])

# Filter
subset = tracks['set', 'subset'] == 'medium'
genre = tracks.loc[subset, ('track', 'genre_top')].dropna()
features = features.loc[genre.index].select_dtypes(include=[np.number]).dropna(axis=1)

print(f"Samples: {len(features)}, Features: {features.shape[1]}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    features, genre, test_size=0.2, random_state=42, stratify=genre
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train
print("Training XGBoost...")
model = xgb.XGBClassifier(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

elapsed = time.time() - start

print("="*50)
print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
print("="*50)
print("\nBaseline established! Now compare with deep learning.")
