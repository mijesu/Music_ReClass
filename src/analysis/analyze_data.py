import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
from collections import Counter

# Paths
TRACKS_PATH = "/media/mijesu_970/SSD_Data/datasets/FMA/Misc/fma_metadata/tracks.csv"
GTZAN_PATH = "/media/mijesu_970/SSD_Data/datasets/GTZAN/Data/genres_original"
FMA_AUDIO_PATH = "/media/mijesu_970/SSD_Data/datasets/FMA/Data/fma_medium"

def analyze_fma_metadata():
    """Analyze FMA genre distribution and class balance"""
    print("="*60)
    print("FMA METADATA ANALYSIS")
    print("="*60)
    
    tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
    
    # Medium subset
    medium = tracks[tracks['set', 'subset'] == 'medium']
    genres = medium['track', 'genre_top'].dropna()
    
    print(f"\nTotal tracks in medium subset: {len(medium)}")
    print(f"Tracks with genre labels: {len(genres)}")
    print(f"Unique genres: {genres.nunique()}")
    
    # Genre distribution
    genre_counts = genres.value_counts()
    print(f"\nGenre Distribution:")
    print(genre_counts)
    
    # Class imbalance ratio
    max_count = genre_counts.max()
    min_count = genre_counts.min()
    print(f"\nClass Imbalance Ratio: {max_count/min_count:.2f}:1")
    print(f"Most common: {genre_counts.index[0]} ({max_count})")
    print(f"Least common: {genre_counts.index[-1]} ({min_count})")
    
    # Plot distribution
    plt.figure(figsize=(12, 6))
    genre_counts.plot(kind='bar')
    plt.title('FMA Medium - Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Tracks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('fma_genre_distribution.png', dpi=150)
    print("\nSaved: fma_genre_distribution.png")
    plt.close()

def analyze_gtzan():
    """Analyze GTZAN genre distribution"""
    print("\n" + "="*60)
    print("GTZAN ANALYSIS")
    print("="*60)
    
    genres = os.listdir(GTZAN_PATH)
    genre_counts = {}
    
    for genre in genres:
        genre_path = os.path.join(GTZAN_PATH, genre)
        if os.path.isdir(genre_path):
            count = len([f for f in os.listdir(genre_path) if f.endswith('.wav')])
            genre_counts[genre] = count
    
    print(f"\nTotal genres: {len(genre_counts)}")
    print(f"\nGenre Distribution:")
    for genre, count in sorted(genre_counts.items()):
        print(f"  {genre:15s}: {count:3d}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(genre_counts.keys(), genre_counts.values())
    plt.title('GTZAN - Genre Distribution')
    plt.xlabel('Genre')
    plt.ylabel('Number of Tracks')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('gtzan_genre_distribution.png', dpi=150)
    print("\nSaved: gtzan_genre_distribution.png")
    plt.close()

def visualize_melspectrograms():
    """Visualize mel-spectrograms from different GTZAN genres"""
    print("\n" + "="*60)
    print("MEL-SPECTROGRAM VISUALIZATION")
    print("="*60)
    
    genres = sorted([g for g in os.listdir(GTZAN_PATH) if os.path.isdir(os.path.join(GTZAN_PATH, g))])[:6]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, genre in enumerate(genres):
        genre_path = os.path.join(GTZAN_PATH, genre)
        files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
        if files:
            audio_file = os.path.join(genre_path, files[0])
            y, sr = librosa.load(audio_file, duration=30)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[idx])
            axes[idx].set_title(f'{genre.capitalize()}')
            axes[idx].label_outer()
    
    plt.tight_layout()
    plt.savefig('melspectrogram_comparison.png', dpi=150)
    print("\nSaved: melspectrogram_comparison.png")
    plt.close()

def check_class_imbalance():
    """Check and report class imbalance metrics"""
    print("\n" + "="*60)
    print("CLASS IMBALANCE SUMMARY")
    print("="*60)
    
    # GTZAN
    gtzan_genres = {}
    for genre in os.listdir(GTZAN_PATH):
        genre_path = os.path.join(GTZAN_PATH, genre)
        if os.path.isdir(genre_path):
            gtzan_genres[genre] = len([f for f in os.listdir(genre_path) if f.endswith('.wav')])
    
    gtzan_counts = list(gtzan_genres.values())
    gtzan_imbalance = max(gtzan_counts) / min(gtzan_counts) if gtzan_counts else 0
    
    print(f"\nGTZAN:")
    print(f"  Imbalance Ratio: {gtzan_imbalance:.2f}:1")
    print(f"  Status: {'BALANCED' if gtzan_imbalance < 1.5 else 'IMBALANCED'}")
    
    # FMA
    tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])
    medium = tracks[tracks['set', 'subset'] == 'medium']
    genres = medium['track', 'genre_top'].dropna()
    genre_counts = genres.value_counts()
    
    fma_imbalance = genre_counts.max() / genre_counts.min()
    
    print(f"\nFMA Medium:")
    print(f"  Imbalance Ratio: {fma_imbalance:.2f}:1")
    print(f"  Status: {'BALANCED' if fma_imbalance < 2 else 'IMBALANCED'}")
    print(f"\n  Recommendation: {'No action needed' if fma_imbalance < 2 else 'Consider class weights or resampling'}")

if __name__ == "__main__":
    analyze_fma_metadata()
    analyze_gtzan()
    visualize_melspectrograms()
    check_class_imbalance()
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
