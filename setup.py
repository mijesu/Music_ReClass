from setuptools import setup, find_packages

setup(
    name="music_reclass",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "pandas>=2.0.0",
        "h5py>=3.8.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
