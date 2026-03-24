import numpy as np
import joblib
import tensorflow as tf
import librosa
import os
from features import extract_features

def predict_emotion(audio_path):
    """
    Takes any .wav file path and predicts the emotion.
    """

    # ── Check file exists ──
    if not os.path.exists(audio_path):
        print(f"ERROR: File not found → {audio_path}")
        return None

    # ── Load saved model, scaler, label encoder ──
    print("Loading model...")
    model  = tf.keras.models.load_model('models/best_model.h5')
    scaler = joblib.load('models/scaler.pkl')
    le     = joblib.load('models/label_encoder.pkl')

    # ── Extract features from audio ──
    print(f"Analyzing: {os.path.basename(audio_path)}")
    features = extract_features(audio_path)

    if features is None:
        print("ERROR: Could not extract features from this file.")
        return None

    # ── Scale and predict ──
    features_scaled = scaler.transform([features])
    prediction      = model.predict(features_scaled, verbose=0)

    # ── Get top 3 results ──
    top3_indices = np.argsort(prediction[0])[::-1][:3]

    print("\n" + "=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"File     : {os.path.basename(audio_path)}")
    print(f"\nTop 3 predictions:")
    for i, idx in enumerate(top3_indices):
        emotion    = le.inverse_transform([idx])[0]
        confidence = prediction[0][idx] * 100
        bar        = "█" * int(confidence / 5)
        print(f"  {i+1}. {emotion:<12} {confidence:5.1f}%  {bar}")

    # ── Final answer ──
    best_emotion    = le.inverse_transform([top3_indices[0]])[0]
    best_confidence = prediction[0][top3_indices[0]] * 100
    print(f"\nFinal prediction : {best_emotion.upper()} ({best_confidence:.1f}% confidence)")
    print("=" * 40)

    return best_emotion, best_confidence


def predict_from_dataset(n=5):
    """
    Tests the model on n random files from your own dataset.
    Great for checking how well the model works.
    """
    from preprocess import load_ravdess_data
    import random

    print("=" * 40)
    print(f"TESTING ON {n} RANDOM FILES FROM DATASET")
    print("=" * 40)

    df      = load_ravdess_data('Data_set')
    samples = df.sample(n, random_state=42)

    correct = 0
    for _, row in samples.iterrows():
        result = predict_emotion(row['path'])
        if result:
            predicted = result[0]
            actual    = row['emotion']
            status    = "✓" if predicted == actual else "✗"
            print(f"{status} Actual: {actual:<12} Predicted: {predicted}")
            if predicted == actual:
                correct += 1

    print(f"\nCorrect: {correct}/{n}")


if __name__ == "__main__":
    # Option 1: Test on random files from your dataset
    predict_from_dataset(n=10)

    # Option 2: Test on a specific file (uncomment and change path)
    # predict_emotion('Data_set/Actor_01/03-01-06-01-01-01-01.wav')