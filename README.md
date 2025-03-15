# MyAgent
{{ ... }}

<p align="center">
  <img src="https://via.placeholder.com/200x200.png?text=MyAgent+Logo" alt="MyAgent Logo" />
</p>

<h1 align="center">MyAgent</h1>
<p align="center">Advanced Open-Source Voice Analysis & Transcription Framework</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#api">API</a> •
  <a href="#contribution">Contribution</a> •
  <a href="#license">License</a>
</p>

---

## 🌟 Overview

**MyAgent** is a comprehensive open-source framework designed for intelligent audio recording, speaker recognition, text transcription, and adaptive voice model learning. It creates a personal, searchable voice archive that allows users to retrieve and analyze their conversations for informed decision-making.

### 🔍 What Makes MyAgent Unique?

Unlike traditional voice recognition systems, **MyAgent** is built on a hybrid architecture:

1. **Local Processing on Powerful Hardware** - Leverages state-of-the-art models without cloud dependency
2. **Mobile Integration via REST API** - Ensures universal compatibility with Android and iOS devices
3. **Adaptive Speaker Recognition** - Continuously improves accuracy through reinforcement learning
4. **Shareable Voice Models** - Facilitates recognition across different devices without extensive retraining

---

## ✨ Features

### 🎤 Voice Activity Detection (VAD)
- Automatically identifies when human speech is present
- Prevents unnecessary continuous recording
- Implemented with **Silero VAD**, a neural network trained for precise human voice detection

### 🔊 Noise Suppression and Audio Enhancement
- Advanced audio signal filtering for more reliable transcription
- Algorithms:
  - **Noisereduce**: Removes light background noise
  - **Demucs**: Separates voices from noise for maximum quality

### 👥 Speaker Identification
- Creates unique voice fingerprints for each speaker
- Based on **SpeechBrain (ECAPA-TDNN)** model, capable of distinguishing voices even in noisy environments
- Continuous reinforcement: Recognition becomes more accurate as users validate or correct results

### 📝 Automatic Transcription with Whisper
- Converts audio signals to text using **Whisper**, OpenAI's model known for accuracy even in noisy environments
- Supports:
  - **Local Mode (Faster-Whisper)**: Optimized for speed and efficiency
  - **API Mode (OpenAI Whisper API)**: Cloud solution for users preferring remote processing

### 🧠 Voice Model Learning and Sharing System
- Export and share voice models between users
- Enables cross-recognition (e.g., a user can send their voice model to a friend for direct recognition without retraining)

### 🔄 REST API for Seamless Mobile Integration
- Functions as a local server that handles all processing and communicates with mobile devices through an API

#### API Endpoints:
- **`POST /upload`**: Mobile app sends audio file for processing
- **`GET /transcription/{id}`**: Retrieves transcription and detected speakers
- **`POST /train_speaker`**: Adds a new voice user to improve recognition
- **`GET /export_model`**: Exports and shares a voice model

---

## 🏗️ Architecture

MyAgent is structured in a modular and extensible way to facilitate adding new features:

```
📂 myagent-framework/
├── 📂 vad/                  # Voice Activity Detection
│   ├── vad_silero.py        # Silero-based detection
│   ├── __init__.py
├── 📂 noise_reduction/      # Noise Suppression
│   ├── noisereduce.py       # Spectral noise reduction
│   ├── demucs.py            # Advanced suppression with Demucs
│   ├── __init__.py
├── 📂 speaker_id/           # Speaker Recognition
│   ├── train_model.py       # Local voice model training
│   ├── recognize.py         # Speaker identification
│   ├── reinforcement.py     # Continuous model improvement
│   ├── export_model.py      # Voice model export/import
│   ├── __init__.py
├── 📂 transcription/        # Audio-to-Text Transcription
│   ├── whisper_local.py     # Local transcription (Faster-Whisper)
│   ├── whisper_api.py       # OpenAI API transcription (optional)
│   ├── __init__.py
├── 📂 api/                  # Mobile Communication API
│   ├── server.py            # Main API (FastAPI)
│   ├── routes.py            # Interaction endpoints
│   ├── __init__.py
├── 📂 utils/                # Miscellaneous Tools
│   ├── audio_utils.py       # Audio conversion, normalization
│   ├── config.py            # Configuration file
│   ├── __init__.py
├── setup.py                 # Installation with pip install .
├── README.md                # Main documentation
├── requirements.txt         # Dependencies
├── .gitignore               # Files to ignore in GitHub
```

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/myagent.git
cd myagent

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install .
```

## 🚀 Usage

### Basic Usage

```python
from myagent import VAD, NoiseReduction, SpeakerID, Transcription

# Initialize components
vad = VAD()
noise_reducer = NoiseReduction()
speaker_id = SpeakerID()
transcriber = Transcription()

# Process an audio file
audio_path = "path/to/audio.wav"
speech_segments = vad.detect(audio_path)
clean_audio = noise_reducer.process(audio_path)
speakers = speaker_id.identify(clean_audio)
transcription = transcriber.transcribe(clean_audio)

print(f"Transcription: {transcription}")
print(f"Speakers identified: {speakers}")
```

### Starting the API Server

```python
from myagent.api import server

# Start server on default port 8000
server.run()

# Or specify a custom port
server.run(port=5000)
```

---

## 🔌 API Reference

### `POST /upload`
Upload an audio file for processing.

**Request:**
```
POST /upload
Content-Type: multipart/form-data

file: [audio_file]
```

**Response:**
```json
{
  "id": "task_12345",
  "status": "processing"
}
```

### `GET /transcription/{id}`
Get the transcription and speaker information for a processed audio file.

**Response:**
```json
{
  "id": "task_12345",
  "status": "completed",
  "transcription": "Hello, this is a test message.",
  "speakers": [
    {
      "id": "speaker_1",
      "name": "John",
      "segments": [{"start": 0.0, "end": 2.5}]
    }
  ]
}
```

### `POST /train_speaker`
Add a new speaker to the recognition model.

**Request:**
```
POST /train_speaker
Content-Type: multipart/form-data

name: "John Doe"
file: [audio_file]
```

**Response:**
```json
{
  "speaker_id": "speaker_12345",
  "status": "trained"
}
```

### `GET /export_model`
Export a trained speaker model.

**Response:**
```json
{
  "model_data": "base64_encoded_model",
  "speaker_id": "speaker_12345"
}
```

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by the MyAgent Team
</p>