import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm


def extract_features(file_path, sample_rate=22050):
    """
    Extracts audio features from a single .wav file.
    Returns a 1D numpy array of features.

    Features extracted:
    - MFCC (40 coefficients, mean + std)  → 80 values
    - Chroma STFT (mean)                  → 12 values
    - Mel Spectrogram (mean)              → 128 values
    - Zero Crossing Rate (mean)           →  1 value
    - RMS Energy (mean)                   →  1 value
    Total feature vector size             → 222 values
    """
    try:
        # Load audio file (cap at 3 seconds for consistency)
        audio, sr = librosa.load(file_path, sr=sample_rate, duration=3.0)

        # Pad if audio is shorter than 3 seconds
        target_length = sample_rate * 3
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # ── Feature 1: MFCC ──────────────────────────────
        # Captures vocal tract shape / phonetic content
        mfcc      = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_mean = np.mean(mfcc.T, axis=0)   # (40,)
        mfcc_std  = np.std(mfcc.T, axis=0)    # (40,)

        # ── Feature 2: Chroma STFT ───────────────────────
        # Captures pitch class / harmonic content
        chroma      = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)  # (12,)

        # ── Feature 3: Mel Spectrogram ───────────────────
        # Captures energy across perceptual frequency bands
        mel      = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_mean = np.mean(mel.T, axis=0)  # (128,)

        # ── Feature 4: Zero Crossing Rate ────────────────
        # How often signal crosses zero — indicates noisiness
        zcr      = librosa.feature.zero_crossing_rate(audio)
        zcr_mean = np.mean(zcr)  # (1,)

        # ── Feature 5: RMS Energy ────────────────────────
        # Overall loudness / intensity of speech
        rms      = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)  # (1,)

        # ── Concatenate all into one feature vector ───────
        features = np.concatenate([
            mfcc_mean,
            mfcc_std,
            chroma_mean,
            mel_mean,
            [zcr_mean],
            [rms_mean]
        ])

        return features

    except Exception as e:
        print(f"  Error extracting features from {file_path}: {e}")
        return None


def build_feature_matrix(df):
    """
    Loops through all files in the dataframe,
    extracts features for each, and returns
    X (feature matrix) and y (emotion labels).
    """
    X = []
    y = []
    failed = 0

    print(f"\nExtracting features from {len(df)} audio files...")
    print("This will take 3-6 minutes, please wait...\n")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting"):
        features = extract_features(row['path'])

        if features is not None:
            X.append(features)
            y.append(row['emotion'])
        else:
            failed += 1

    print(f"\nDone! Successfully processed: {len(X)} files")
    if failed > 0:
        print(f"Failed files: {failed}")

    X = np.array(X)
    y = np.array(y)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape        : {y.shape}")

    return X, y


if __name__ == "__main__":
    from preprocess import load_ravdess_data

    print("=" * 50)
    print("FEATURE EXTRACTION TEST (first 5 files only)")
    print("=" * 50)

    df = load_ravdess_data('Data_set')

    if df is not None:
        # Test on just 5 files first
        df_sample = df.head(5)
        X, y = build_feature_matrix(df_sample)

        print(f"\nSample feature vector (first file):")
        print(f"Shape : {X[0].shape}")
        print(f"Min   : {X[0].min():.4f}")
        print(f"Max   : {X[0].max():.4f}")
        print(f"Labels: {y}")
        print("\nfeatures.py is working correctly!")