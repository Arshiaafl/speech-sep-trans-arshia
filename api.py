from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import torch
import whisper
import logging
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Load models globally
logger.info("Loading SpeechBrain SepFormer model...")
sep_model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir='pretrained_models/sepformer-wham')

logger.info("Loading Whisper model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large-v3", device=device)

def separate_and_transcribe(audio_path):
    """
    Separate audio into two speaker streams and transcribe with Whisper.
    
    Args:
        audio_path (str): Path to the mixed audio file (e.g., 'new/1.wav').
    
    Returns:
        dict: Transcriptions per speaker.
    """
    try:
        # Log audio info before separation
        logger.info(f"Loading audio file: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)
        logger.info(f"Audio shape before processing: {waveform.shape}, Sample rate: {sample_rate}")
        
        # Ensure audio is mono and at 8kHz for SepFormer
        if waveform.shape[0] > 1:  # Stereo to mono
            waveform = waveform.mean(dim=0, keepdim=True)  # [channels, samples] -> [1, samples]
        if sample_rate != 8000:
            waveform = torchaudio.transforms.Resample(sample_rate, 8000)(waveform)
            sample_rate = 8000
        
        logger.info(f"Audio shape after processing: {waveform.shape}")
        
        # Separate audio file (exactly as your working script)
        logger.info("Separating audio...")
        est_sources = sep_model.separate_file(path=audio_path)
        
        # Save separated sources temporarily to disk (like your script)
        logger.info("Saving separated sources temporarily...")
        temp_dir = "temp_separated"
        os.makedirs(temp_dir, exist_ok=True)
        torchaudio.save(os.path.join(temp_dir, "source1hat.wav"), est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save(os.path.join(temp_dir, "source2hat.wav"), est_sources[:, :, 1].detach().cpu(), 8000)
        
        # Transcribe each speaker stream
        transcriptions = {}
        for i, audio_file in enumerate(["source1hat.wav", "source2hat.wav"]):
            audio_path = os.path.join(temp_dir, audio_file)
            logger.info(f"Transcribing {audio_file}...")
            waveform, sr = torchaudio.load(audio_path)
            logger.info(f"Separated audio shape: {waveform.shape}, Sample rate: {sr}")
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                sr = 16000
            
            result = whisper_model.transcribe(waveform.squeeze().numpy(), language="en")
            speaker_id = f"Speaker{i + 1}"
            transcriptions[speaker_id] = result["text"]
        
        # Clean up temporary files
        for audio_file in ["source1hat.wav", "source2hat.wav"]:
            os.remove(os.path.join(temp_dir, audio_file))
        os.rmdir(temp_dir)
        
        return transcriptions
    
    except Exception as e:
        logger.error(f"Processing failed with error: {str(e)}")
        raise

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    """
    API endpoint to transcribe a mixed audio file into two speaker streams.
    
    Args:
        file (UploadFile): Uploaded audio file (.wav).
    
    Returns:
        JSONResponse: Transcriptions for each speaker.
    """
    try:
        # Read audio file directly into memory and save temporarily
        logger.info(f"Received file: {file.filename}")
        if not file.filename.endswith('.wav'):
            raise HTTPException(status_code=400, detail="Only .wav files are supported.")
        
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        # Process audio
        transcriptions = separate_and_transcribe(temp_path)
        
        # Clean up temporary file
        os.remove(temp_path)
        
        # Format response
        response = {speaker: text for speaker, text in transcriptions.items()}
        return JSONResponse(content=response)
    
    except Exception as e:
        logger.error(f"API request failed with error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)