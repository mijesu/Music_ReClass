import numpy as np
import matplotlib.pyplot as plt

# Load features
fma = np.load('FMA_features.npy')
jmla = np.load('JMLA_features.npy')
mert = np.load('MERT_features.npy')

print("=" * 60)
print("FEATURE COMPARISON")
print("=" * 60)

# Basic stats
print(f"\n1. FMA Features:")
print(f"   Shape: {fma.shape} (songs × features)")
print(f"   Size: {fma.nbytes / 1024:.2f} KB")
print(f"   Mean: {fma.mean():.4f}, Std: {fma.std():.4f}")
print(f"   Range: [{fma.min():.4f}, {fma.max():.4f}]")

print(f"\n2. JMLA Features:")
print(f"   Shape: {jmla.shape} (songs × features)")
print(f"   Size: {jmla.nbytes / 1024 / 1024:.2f} MB")
print(f"   Mean: {jmla.mean():.4f}, Std: {jmla.std():.4f}")
print(f"   Range: [{jmla.min():.4f}, {jmla.max():.4f}]")

print(f"\n3. MERT Features:")
print(f"   Shape: {mert.shape} (songs × features)")
print(f"   Size: {mert.nbytes / 1024:.2f} KB")
print(f"   Mean: {mert.mean():.4f}, Std: {mert.std():.4f}")
print(f"   Range: [{mert.min():.4f}, {mert.max():.4f}]")

# Combined features
combined = np.concatenate([fma, jmla, mert], axis=1)
print(f"\n4. Combined (FMA + JMLA + MERT):")
print(f"   Shape: {combined.shape}")
print(f"   Size: {combined.nbytes / 1024 / 1024:.2f} MB")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'Feature':<10} {'Dims':<10} {'Size':<12} {'Use Case'}")
print("-" * 60)
print(f"{'FMA':<10} {fma.shape[1]:<10} {'51 KB':<12} Fast training (77% acc)")
print(f"{'JMLA':<10} {jmla.shape[1]:<10} {'16 MB':<12} Deep features (high acc)")
print(f"{'MERT':<10} {mert.shape[1]:<10} {'100 KB':<12} Audio transformer")
print(f"{'Combined':<10} {combined.shape[1]:<10} {'16 MB':<12} Best accuracy (85-94%)")
print("=" * 60)
