from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
import torch  # Added missing import
import whisper
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def separate_and_transcribe(audio_path, sep_model, whisper_model_name="large-v3", language="en"):
    """
    Separate audio into two speaker streams and transcribe with Whisper.
    
    Args:
        audio_path (str): Path to the mixed audio file (e.g., 'new/1.wav').
        sep_model: Loaded SpeechBrain SepFormer model.
        whisper_model_name (str): Whisper model size (e.g., 'large-v3').
        language (str): Language code (e.g., 'en' for English).
    
    Returns:
        dict: Transcriptions per speaker with timestamps.
    """
    try:
        # Separate audio file
        logger.info(f"Separating audio file: {audio_path}")
        est_sources = sep_model.separate_file(path=audio_path)
        
        # Save separated sources
        logger.info("Saving separated sources...")
        torchaudio.save("source1hat.wav", est_sources[:, :, 0].detach().cpu(), 8000)
        torchaudio.save("source2hat.wav", est_sources[:, :, 1].detach().cpu(), 8000)
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {whisper_model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        whisper_model = whisper.load_model(whisper_model_name, device=device)
        
        # Transcribe each speaker stream
        transcriptions = {}
        for i, audio_file in enumerate(["source1hat.wav", "source2hat.wav"]):
            logger.info(f"Transcribing {audio_file}...")
            # Load audio with torchaudio (Whisper expects 16kHz)
            waveform, sr = torchaudio.load(audio_file)
            if sr != 16000:
                waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                sr = 16000
            
            # Transcribe with Whisper
            result = whisper_model.transcribe(waveform.squeeze().numpy(), language=language)
            
            # Store transcription
            speaker_id = f"Speaker{i + 1}"
            transcriptions[speaker_id] = result["text"]
        
        return transcriptions
    
    except Exception as e:
        logger.error(f"Processing failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Load SpeechBrain SepFormer model
        logger.info("Loading SpeechBrain SepFormer model...")
        model = separator.from_hparams(source="speechbrain/sepformer-wham", savedir='pretrained_models/sepformer-wham')
        
        # Process audio file
        audio_path = 'new/1.wav'
        transcriptions = separate_and_transcribe(audio_path, model)
        
        # Print results in desired format
        for speaker, text in transcriptions.items():
            print(f"{speaker}: {text}")
        
    except Exception as e:
        logger.error(f"Failed with error: {str(e)}")
        raise