import subprocess
import json
import numpy as np
import librosa
import os
import logging
from pathlib import Path
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)

def analyze_audio_energy(video_id: str, video_path: str, output_dir: str) -> str:
    """
    Extracts full audio from video, calculates per-second metrics:
    - rms_energy
    - emotion
    - music_intensity
    - speech_probability
    Saves the JSON to the audio output directory.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    full_audio_wav = audio_dir / "full_audio.wav"
    output_json_path = audio_dir / "full_audio_intelligence.json"
    
    # 1. Extract Audio using FFmpeg
    if not full_audio_wav.exists():
        logger.info(f"Extracting full audio from {video_path} to {full_audio_wav}")
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",          # No video
            "-ac", "1",     # Mono
            "-ar", "16000", # 16kHz required for wav2vec2
            "-c:a", "pcm_s16le", # 16-bit PCM
            str(full_audio_wav)
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg extraction failed: {e}")
            raise RuntimeError("Failed to extract audio using FFmpeg")
            
    # 2. Load SER Model
    logger.info("Loading Wav2Vec2 SER model for per-second analysis...")
    device = 0 if torch.cuda.is_available() else -1
    try:
        ser_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=device)
    except Exception as e:
        logger.error(f"Failed to load SER model: {e}")
        ser_model = None

    # 3. Analyze waveform & Calculate Metrics per second
    logger.info("Analyzing waveform and calculating per-second intelligence...")
    
    # Load audio with librosa
    try:
        y, sr = librosa.load(str(full_audio_wav), sr=16000)
    except Exception as e:
        logger.error(f"Failed to load audio with librosa: {e}")
        raise RuntimeError("Audio analysis failed")
        
    samples_per_sec = sr
    num_seconds = len(y) // samples_per_sec
    
    # Pre-calculate spectral features for heuristics
    # Zero Crossing Rate (proxy for speech fricatives)
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    # Spectral Centroid (proxy for music vs speech balance)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    # Onset Strength (proxy for music beats/intensity)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    
    intelligence_data = []
    
    # Frame lengths for spectral features
    frames_per_sec = len(zcr) / (len(y) / sr) if len(y) > 0 else 0
    
    for i in range(num_seconds):
        start_sample = i * samples_per_sec
        end_sample = (i + 1) * samples_per_sec
        sec_samples = y[start_sample:end_sample]
        
        if len(sec_samples) == 0:
            continue
            
        # --- RMS Energy ---
        rms_energy = float(np.sqrt(np.mean(sec_samples**2)))
        
        # --- Emotion ---
        emotion = "neutral"
        if ser_model and rms_energy > 0.005:  # Only run if not silent
            try:
                # Provide the raw numpy array directly
                res = ser_model(sec_samples)
                # res is a list of dicts: [{'score': 0.9, 'label': 'neu'}, ...]
                if res and isinstance(res, list) and len(res) > 0:
                    emotion = res[0]['label']
                    # Map labels if necessary
                    label_map = {'neu': 'neutral', 'ang': 'anger', 'hap': 'happiness', 'sad': 'sadness'}
                    emotion = label_map.get(emotion.lower(), emotion)
            except Exception as e:
                logger.warning(f"SER failed at sec {i}: {e}")
                
        # --- Spectral Heuristics ---
        start_frame = int(i * frames_per_sec)
        end_frame = int((i + 1) * frames_per_sec)
        
        # --- Speech Probability (Heuristic: ZCR + Centroid) ---
        # High ZCR and mid-high centroid indicates speech
        if end_frame > start_frame and len(zcr[start_frame:end_frame]) > 0:
            avg_zcr = np.mean(zcr[start_frame:end_frame])
            avg_centroid = np.mean(centroid[start_frame:end_frame])
            # Normalize heuristically
            speech_prob = min(1.0, max(0.0, float(avg_zcr * 5.0)))
        else:
            speech_prob = 0.0
            
        # --- Music Intensity (Heuristic: Onset Strength + Energy) ---
        if end_frame > start_frame and len(onset_env[start_frame:end_frame]) > 0:
            avg_onset = np.mean(onset_env[start_frame:end_frame])
            # Normalize heuristically
            music_intensity = min(1.0, float((avg_onset / 10.0) + (rms_energy * 2.0)))
        else:
            music_intensity = 0.0
            
        # Zero out if silent
        if rms_energy < 0.005:
            speech_prob = 0.0
            music_intensity = 0.0

        intelligence_data.append({
            "start": float(i),
            "end": float(i + 1),
            "rms_energy": round(rms_energy, 4),
            "emotion": emotion,
            "music_intensity": round(music_intensity, 4),
            "speech_probability": round(speech_prob, 4)
        })
        
    # 4. Save JSON in storage folder
    with open(output_json_path, 'w') as f:
        json.dump(intelligence_data, f, indent=2)
        
    logger.info(f"Saved detailed audio intelligence to {output_json_path}")
    return str(output_json_path)
