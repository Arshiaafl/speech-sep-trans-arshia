# Speech Separation and Transcription API

This repository contains a FastAPI application that separates mixed audio from two speakers and transcribes their speech into text on master branch, outputting results in the format "Speaker1: ..." and "Speaker2: ...". It uses SpeechBrain for audio separation and Whisper for transcription, perfect for processing two-speaker audio files like `.wav`.

## Prerequisites
- Python 3.8 or later
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/Arshiaafl/speech-sep-trans-arshia
cd speech-sep-api
```

### 2. Create a Virtual Environment
Set up a virtual environment to isolate dependencies:
```bash
python -m venv venv
```

### 3. Activate the Virtual Environment
- **Windows (PowerShell/CMD)**:
  ```bash
  .\venv\Scripts\activate
  ```
- Your terminal prompt should now show `(venv)`.

### 4. Install Dependencies
Install the required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Run the API
With the virtual environment activated, start the FastAPI server:
```bash
python api.py
```
- The API will run on `http://0.0.0.0:8000`.

### 2. Test the API
You can test the API using Postman, cURL, or any HTTP client. Ensure you have a `.wav` audio file (mono, 8kHz or 16kHz, up to 5 seconds) to upload.

#### Using Postman
- Create a new POST request to `http://localhost:8000/transcribe`.
- In the Body tab, select `form-data`.
- Add a key `file`, set the type to "File", and upload your `.wav` file (e.g., `new/1.wav`).
- Send the request—you’ll receive a JSON response like:
  ```json
  {
    "Speaker1": "BUT IN SUCH A CASE MISS MILNER'S ELECTION OF A HUSBAND SHALL NOT DIRECT MINE",
    "Speaker2": "HE SPOKE SIMPLY BUT PACED UP AND DOWN THE NARROW CELL IN FRONT OF THEM"
  }
  ```

#### Using cURL (if installed)
If you have `curl` installed, run:
```bash
curl -X POST -F "file=@path\to\your\audio.wav" http://localhost:8000/transcribe
```
- Replace `path\to\your\audio.wav` with the path to your `.wav` file.

## Notes
- The API assumes input audio files contain two speakers mixed together.
- Ensure your `.wav` files are in a compatible format (mono, 8kHz or 16kHz) for optimal results.
- The API uses temporary files during processing and cleans them up automatically.

## Troubleshooting
- If you encounter errors during installation, ensure Python 3.8+ is installed and try:
  ```bash
  pip install -r requirements.txt --no-cache-dir
  ```
- For `torchaudio` issues on CPU-only systems, ensure you have `soundfile` installed or use `torchaudio[ffmpeg]`:
  ```bash
  pip install soundfile
  ```
  or
  ```bash
  pip install torchaudio[ffmpeg]
  ```

