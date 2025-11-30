import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Paths
FEATURES_PATH = "/media/mijesu_970/SSD_Data/datasets/FMA/Misc/fma_metadata/features.csv"
TRACKS_PATH = "/media/mijesu_970/SSD_Data/datasets/FMA/Misc/fma_metadata/tracks.csv"

print("Loading data...")
features = pd.read_csv(FEATURES_PATH, index_col=0, header=[0, 1, 2])
tracks = pd.read_csv(TRACKS_PATH, index_col=0, header=[0, 1])

# Get medium subset and genre labels
subset = tracks['set', 'subset'] == 'medium'
genre = tracks.loc[subset, ('track', 'genre_top')]

# Remove rows with missing genre
valid = ~genre.isna()
features = features.loc[valid]
genre = genre.loc[valid]

print(f"Samples: {len(features)}, Genres: {genre.nunique()}")

# Remove non-numeric and highly correlated features
features = features.select_dtypes(include=[np.number])
features = features.dropna(axis=1)
corr = features.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
features = features.drop(columns=to_drop)

print(f"Features after cleaning: {features.shape[1]}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    features, genre, test_size=0.2, random_state=42, stratify=genre
)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PCA
pca = PCA(n_components=60)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train XGBoost
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=10)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*50}")
print(f"Test Accuracy: {accuracy*100:.2f}%")
print(f"{'='*50}\n")
print(classification_report(y_test, y_pred))

# Save model
with open('xgboost_fma_model.pkl', 'wb') as f:
    pickle.dump({'model': model, 'scaler': scaler, 'pca': pca}, f)

print("Model saved to xgboost_fma_model.pkl")
