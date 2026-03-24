import sounddevice as sd
import soundfile as sf
import numpy as np
import time
from predict import predict_emotion

def reduce_noise(audio, sample_rate):
    """
    Simple noise reduction by removing
    low energy parts of the signal
    """
    # Calculate energy
    energy    = np.abs(audio)
    threshold = np.mean(energy) * 0.1

    # Keep only parts above threshold
    audio_cleaned = np.where(energy > threshold, audio, 0.0)
    return audio_cleaned


def record_and_predict(duration=5, sample_rate=22050):
    print("=" * 40)
    print("   VOICE EMOTION DETECTOR")
    print("=" * 40)
    print(f"\nTips for better results:")
    print("  → Speak LOUDLY and expressively")
    print("  → Exaggerate your emotion")
    print("  → Minimize background noise")
    print("  → Speak in English")
    print(f"\nRecording for {duration} seconds...")
    print("Recording starts after countdown!\n")

    # Countdown
    for i in range(3, 0, -1):
        print(f"  Starting in {i}...")
        time.sleep(1)

    print("\n RECORDING NOW — SPEAK LOUDLY!")
    print("-" * 40)

    # Record audio
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )

    # Live countdown
    for i in range(duration, 0, -1):
        print(f"  {i} seconds remaining...")
        time.sleep(1)

    sd.wait()
    print("-" * 40)
    print("Recording complete!\n")

    # Flatten audio array
    audio = audio.flatten()

    # Check if audio is too quiet
    volume = np.max(np.abs(audio))
    print(f"Volume level: {volume:.4f}")

    if volume < 0.01:
        print("WARNING: Audio is very quiet!")
        print("Please speak louder and try again.\n")
        return None

    if volume < 0.05:
        print("NOTE: Audio is a bit quiet, boosting volume...")
        audio = audio * (0.1 / volume)

    # Apply noise reduction
    audio = reduce_noise(audio, sample_rate)

    # Save recording
    output_path = 'my_recording.wav'
    sf.write(output_path, audio, sample_rate)
    print(f"Saved to: {output_path}")

    # Predict
    print("\nAnalyzing your voice...\n")
    result = predict_emotion(output_path)

    return result


if __name__ == "__main__":
    while True:
        record_and_predict(duration=5)
        print("\n")
        again = input("Record again? (y/n): ").strip().lower()
        if again != 'y':
            print("\nGoodbye!")
            break