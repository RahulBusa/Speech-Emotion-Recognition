
# 🎤 Speech Emotion Recognition (SER) — AI + NLP Inspired Project

An intelligent system that detects human emotions from speech using **Deep Learning, Audio Signal Processing, and concepts inspired by Natural Language Processing (NLP)**.

---

## 🚀 Project Overview

This project analyzes speech audio and predicts emotions such as:

* 😀 Happy
* 😢 Sad
* 😠 Angry
* 😐 Neutral
* 😨 Fearful
* 🤢 Disgust
* 😲 Surprise

It uses a **CNN + MLP ensemble model** trained on multiple datasets to achieve robust performance.

---

## 🧠 Key Features

* 🎤 Real-time voice emotion detection
* 📁 Upload audio file for prediction
* 🧪 Dataset-based testing
* 📊 Audio visualization (waveform, spectrogram, MFCC)
* 🤖 Ensemble Learning (CNN + MLP)
* 🌐 Interactive Web UI (Flask + HTML + JavaScript)

---

## 🧩 Relation to NLP (Important)

This project is part of the broader **Speech Processing & NLP ecosystem**.

Although it works on **audio signals instead of text**, it connects to NLP in the following way:

* Speech → (ASR) → Text → NLP processing
* Emotion can be detected from:

  * 🔊 Voice tone (this project)
  * 📝 Text meaning (NLP)

👉 This project focuses on **paralinguistic features (tone, pitch, energy)**
👉 NLP focuses on **linguistic features (words, meaning)**

🔗 Together, they form **Multimodal Emotion Recognition**

---

## 📂 Project Structure

```id="c0t3iy"
Speech-Emotion-Recognition/
│
├── speech_emotion_detection.ipynb   # ⭐ MAIN PROJECT (complete pipeline)
├── features.py                      # Feature extraction
├── preprocess.py                    # Dataset loading
├── train.py                         # Model training
├── predict.py                       # Prediction logic
├── record_and_predict.py            # Real-time prediction
├── main.py                          # Training pipeline
├── my_recording.wav                 # Sample audio
├── README.md                        # Documentation
```

👉 **Note:** Entire workflow is inside:

```id="k3u62y"
speech_emotion_detection.ipynb
```

---

## 📊 Datasets Used

* RAVDESS
* TESS
* CREMA-D
* SAVEE

These datasets contain **emotion-labeled speech recordings**.

---

## ⚙️ Technologies Used

* Python
* Librosa (Audio Processing)
* TensorFlow / Keras
* NumPy / Pandas
* Flask (Backend)
* HTML, CSS, JavaScript (Frontend)

---

## 🏗️ Model Architecture

### 🔹 CNN Model

* Works on Mel Spectrograms
* Captures spatial audio patterns

### 🔹 MLP Model

* Works on extracted numerical features
* Captures statistical properties of audio

### 🔥 Ensemble Model

* Combines CNN + MLP predictions
* Improves accuracy and robustness

---

## 📈 Results

* ✅ Accuracy: ~80%
* ✅ Strong performance on dataset audio
* ⚠️ Moderate performance on real-world speech

---

## ▶️ How to Run

### 🔹 Run in Google Colab (Recommended)

1. Open `speech_emotion_detection.ipynb`
2. Run all cells
3. Use UI for:

   * 🎤 Voice recording
   * 📁 Upload audio
   * 🧪 Dataset testing

---

### 🔹 Run Locally

```bash id="l6t6wq"
pip install -r requirements.txt
python main.py
python predict.py <audio_file.wav>
```

---

## ⚠️ Limitations

* Uses **acted datasets** (not fully natural speech)
* Sensitive to noise and low-quality audio
* Emotion detection depends on expressiveness

---

## 🔮 Future Enhancements

* 🔊 Add noise reduction
* 🧠 Integrate Speech-to-Text (ASR)
* 📝 Add NLP-based emotion detection
* 🔥 Build Multimodal Emotion AI (Audio + Text + Face)
* 🌐 Deploy as web/mobile application

---

## 👨‍💻 Author

**Rahul Busa**

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub!
