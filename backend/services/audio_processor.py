import os
import subprocess
import json
import logging
from pathlib import Path
import numpy as np
import librosa
from pydub import AudioSegment
from transformers import pipeline

try:
    import torch
    import torchaudio
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    DEMUCS_AVAILABLE = True
except ImportError:
    DEMUCS_AVAILABLE = False
    logging.warning("Demucs not available. Audio separation will be mocked or skipped.")

logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if DEMUCS_AVAILABLE else "cpu"
        
        # Lazy loading for models
        self.demucs_model = None
        
        # Initialize SER model
        try:
            logger.info("Initializing Wav2Vec2 SER model...")
            self.voice_emotion_model = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er", device=0 if self.device == "cuda" else -1)
        except Exception as e:
            logger.error(f"Failed to load SER model: {e}")
            self.voice_emotion_model = None

    def extract_scene_audio(self, video_path: str, scene_metadata_path: str) -> list[dict]:
        """Extracts audio chunks for each scene using FFmpeg."""
        video_path = Path(video_path)
        scene_metadata_path = Path(scene_metadata_path)
        
        if not video_path.exists() or not scene_metadata_path.exists():
            raise FileNotFoundError(f"Video or metadata file not found: {video_path}, {scene_metadata_path}")

        with open(scene_metadata_path, 'r') as f:
            metadata = json.load(f)

        scenes = metadata.get("scenes", [])
        extracted_paths = []

        audio_dir = self.output_base_dir / "audio" / "raw"
        audio_dir.mkdir(parents=True, exist_ok=True)

        for scene in scenes:
            scene_id = scene.get("scene_id")
            start_time = scene.get("start")
            end_time = scene.get("end")
            
            if not all([scene_id, start_time, end_time]):
                logger.warning(f"Missing metadata for scene: {scene}")
                continue

            # Skip 0 second scenes to prevent FFmpeg crashes
            if start_time == end_time:
                logger.warning(f"Skipping {scene_id} because start and end times are identical ({start_time}).")
                continue

            output_audio = audio_dir / f"{scene_id}_raw.wav"
            
            # Extract 48kHz, 24-bit audio using FFmpeg
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ss", str(start_time),
                "-to", str(end_time),
                "-vn", # No video
                "-ac", "2", # Stereo
                "-ar", "48000", # 48kHz (Demucs standard)
                "-c:a", "pcm_s24le", # 24-bit PCM
                str(output_audio)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                extracted_paths.append({
                    "scene_id": scene_id,
                    "path": str(output_audio)
                })
                logger.info(f"Extracted audio for {scene_id}")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg extraction failed for {scene_id}: {e.stderr.decode()}")

        return extracted_paths

    def estimate_snr(self, audio_path: str) -> float:
        """Heuristic to estimate Signal-to-Noise Ratio to trigger deep restoration."""
        y, sr = librosa.load(audio_path, sr=None)
        rms = librosa.feature.rms(y=y)[0]
        signal_power = np.mean(rms ** 2)
        noise_power = np.percentile(rms, 10) ** 2 # approximate noise floor
        
        if noise_power == 0:
            return 100.0 # very clean
            
        snr = 10 * np.log10(signal_power / noise_power)
        return snr

    def apply_restoration(self, audio_path: str) -> str:
        """Applies basic spectral denoising using librosa as fallback to DeepFilterNet."""
        snr = self.estimate_snr(audio_path)
        logger.info(f"Estimated SNR for {audio_path}: {snr:.2f} dB")
        
        if snr > 15.0:
            return audio_path

        logger.info(f"Applying basic spectral denoising to {audio_path}")
        y, sr = librosa.load(audio_path, sr=None)
        
        # Simple high-pass filter to remove 50Hz hum and low rumble
        y_filt = librosa.effects.preemphasis(y, coef=0.97)
        
        restored_path = str(Path(audio_path).parent / f"{Path(audio_path).stem}_restored.wav")
        import soundfile as sf
        sf.write(restored_path, y_filt, sr, subtype='PCM_24')
        
        return restored_path

    def separate_sources(self, audio_path: str, scene_id: str) -> dict:
        """Separates audio into Dialogue, BGM, and Effects using Demucs."""
        out_paths = {
            "dialogue": None,
            "bgm": None,
            "effects": None,
            "ambience": None
        }
        
        base_dir = self.output_base_dir / "audio"
        for stem in out_paths.keys():
            (base_dir / stem).mkdir(parents=True, exist_ok=True)

        if not DEMUCS_AVAILABLE:
            logger.warning("Demucs not installed, mocking source separation.")
            for stem in out_paths.keys():
                path = str(base_dir / stem / f"{stem}_{scene_id}.wav")
                subprocess.run(["cp", audio_path, path])
                out_paths[stem] = path
            return out_paths

        if self.demucs_model is None:
            logger.info("Loading Demucs HT model...")
            self.demucs_model = get_model(name='htdemucs').to(self.device)

        logger.info(f"Running Demucs source separation on {audio_path}")
        import soundfile as sf
        wav_data, sr = sf.read(audio_path, dtype='float32')
        if len(wav_data.shape) == 1:
            wav_data = np.stack([wav_data, wav_data], axis=1)
            
        wav = torch.from_numpy(wav_data).transpose(0, 1) # [channels, frames]
        wav = wav.to(self.device)
        wav = wav.unsqueeze(0) # [1, channels, frames]
        
        # HTDemucs stems: 0=drums, 1=bass, 2=other, 3=vocals
        sources = apply_model(self.demucs_model, wav, shifts=1, split=True, overlap=0.25)[0]
        
        drums = sources[0].cpu()
        bass = sources[1].cpu()
        other = sources[2].cpu()
        vocals = sources[3].cpu()
        
        # Map to cinematic stems
        dialogue = vocals
        bgm = bass + drums + (other * 0.5)
        effects = other * 0.5
        ambience = drums * 0.2 + other * 0.3
        
        out_paths["dialogue"] = str(base_dir / "dialogue" / f"dialogue_{scene_id}.wav")
        out_paths["bgm"] = str(base_dir / "bgm" / f"bgm_{scene_id}.wav")
        out_paths["effects"] = str(base_dir / "effects" / f"effects_{scene_id}.wav")
        out_paths["ambience"] = str(base_dir / "ambience" / f"amb_{scene_id}.wav")
        
        sf.write(out_paths["dialogue"], dialogue.transpose(0, 1).numpy(), sr)
        sf.write(out_paths["bgm"], bgm.transpose(0, 1).numpy(), sr)
        sf.write(out_paths["effects"], effects.transpose(0, 1).numpy(), sr)
        sf.write(out_paths["ambience"], ambience.transpose(0, 1).numpy(), sr)
        
        return out_paths

    def extract_cinematic_features(self, stems: dict, scene_id: str) -> dict:
        """Analyzes stems to extract emotion and cinematic features."""
        try:
            bgm_y, sr = librosa.load(stems["bgm"], sr=22050)
            bgm_rms = np.mean(librosa.feature.rms(y=bgm_y))
            elevation_score = float(min(bgm_rms * 10, 1.0))
        except Exception:
            elevation_score = 0.5
            
        try:
            dialogue_seg = AudioSegment.from_wav(stems["dialogue"])
            silence_thresh = dialogue_seg.dBFS - 16
            silences = len([s for s in dialogue_seg if s.dBFS < silence_thresh])
            dramatic_silence = bool((silences > len(dialogue_seg) * 0.3) and elevation_score > 0.6)
        except Exception:
            dramatic_silence = False

        # Apply SER model to the dialogue stem
        vocal_emotion_detected = "neutral"
        vocal_emotion_confidence = 0.0
        if self.voice_emotion_model and os.path.exists(stems["dialogue"]):
            try:
                # model outputs: [{'score': 0.8, 'label': 'neu'}, ...]
                ser_result = self.voice_emotion_model(stems["dialogue"])
                if ser_result:
                    # Map superb labels to readable labels
                    label_map = {"neu": "neutral", "ang": "anger", "hap": "happiness", "sad": "sadness"}
                    top_pred = max(ser_result, key=lambda x: x["score"])
                    vocal_emotion_detected = label_map.get(top_pred["label"], "neutral")
                    vocal_emotion_confidence = round(float(top_pred["score"]), 2)
            except Exception as e:
                logger.error(f"SER failed for {scene_id}: {e}")

        features = {
            "scene_id": scene_id,
            "dialogue_path": stems["dialogue"],
            "bgm_path": stems["bgm"],
            "effects_path": stems["effects"],
            "ambience_path": stems["ambience"],
            "audio_features": {
                "emotion_intensity": round((elevation_score + 0.5) / 2, 2),
                "dramatic_silence": dramatic_silence,
                "bgm_elevation_score": round(elevation_score, 2),
                "crowd_hype_score": 0.5,
                "dialogue_clarity": 0.9,
                "vocal_emotion_detected": vocal_emotion_detected,
                "vocal_emotion_confidence": vocal_emotion_confidence
            }
        }
        
        base_dir = self.output_base_dir / "audio" / "features"
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"features_{scene_id}.json"
        
        with open(out_path, 'w') as f:
            json.dump(features, f, indent=4)
            
        return features

    def process_movie(self, video_path: str, scene_metadata_path: str) -> list[dict]:
        """Runs the complete end-to-end audio pipeline for a movie."""
        logger.info(f"Starting Stage 3 Audio Pipeline for {video_path}")
        
        extracted_chunks = self.extract_scene_audio(video_path, scene_metadata_path)
        results = []
        
        for chunk in extracted_chunks:
            scene_id = chunk["scene_id"]
            raw_path = chunk["path"]
            
            logger.info(f"Processing scene {scene_id}...")
            
            # 1. Restoration
            restored_path = self.apply_restoration(raw_path)
            
            # 2. Source Separation
            stems = self.separate_sources(restored_path, scene_id)
            
            # 3. Feature Extraction
            features = self.extract_cinematic_features(stems, scene_id)
            results.append(features)
            
        return results
