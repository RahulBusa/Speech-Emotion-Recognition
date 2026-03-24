import os
import pandas as pd
import librosa
import numpy as np

# ─── Emotion mapping from RAVDESS filename convention ───
EMOTION_MAP = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def load_ravdess_data(data_path='Data_set'):
    """
    Walks through all Actor folders in Data_set,
    reads each .wav filename, extracts the emotion
    code from position [2] and builds a dataframe.
    """
    file_paths = []
    emotions   = []
    actors     = []

    # Check if path exists
    if not os.path.exists(data_path):
        print(f"ERROR: '{data_path}' folder not found!")
        print(f"Current directory: {os.getcwd()}")
        return None

    actor_folders = sorted(os.listdir(data_path))
    print(f"Found {len(actor_folders)} folders in {data_path}")

    for actor_folder in actor_folders:
        actor_path = os.path.join(data_path, actor_folder)

        # Skip if not a directory
        if not os.path.isdir(actor_path):
            continue

        wav_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]

        for file in wav_files:
            try:
                parts = file.split('-')

                # RAVDESS format: 03-01-06-01-02-01-12.wav
                # index:           0  1  2  3  4  5  6
                # parts[2] = emotion code
                emotion_code = parts[2]
                emotion      = EMOTION_MAP.get(emotion_code)

                if emotion is None:
                    print(f"  Skipping unknown emotion code '{emotion_code}' in {file}")
                    continue

                full_path = os.path.join(actor_path, file)
                file_paths.append(full_path)
                emotions.append(emotion)
                actors.append(actor_folder)

            except Exception as e:
                print(f"  Error parsing {file}: {e}")
                continue

    # Build dataframe
    df = pd.DataFrame({
        'path':   file_paths,
        'emotion': emotions,
        'actor':  actors
    })

    print(f"\nTotal audio files loaded: {len(df)}")
    print("\nEmotion distribution:")
    print(df['emotion'].value_counts())
    print("\nSample rows:")
    print(df.head(3))

    return df


def check_audio_file(file_path):
    """
    Loads a single audio file and prints basic info.
    Useful for sanity checking your data.
    """
    try:
        audio, sr = librosa.load(file_path, sr=22050, duration=3.0)
        duration  = librosa.get_duration(y=audio, sr=sr)
        print(f"File     : {os.path.basename(file_path)}")
        print(f"Sample rate: {sr} Hz")
        print(f"Duration   : {duration:.2f} seconds")
        print(f"Audio shape: {audio.shape}")
        return True
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return False


if __name__ == "__main__":
    # Run this file directly to test data loading
    print("=" * 50)
    print("RAVDESS DATA LOADING TEST")
    print("=" * 50)

    df = load_ravdess_data('Data_set')

    if df is not None and len(df) > 0:
        print("\n" + "=" * 50)
        print("AUDIO FILE SANITY CHECK (first file)")
        print("=" * 50)
        check_audio_file(df['path'].iloc[0])
        print("\npreprocess.py is working correctly!")
    else:
        print("\nNo data loaded. Check your Data_set folder structure.")