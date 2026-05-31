import os
import json
import logging
from pathlib import Path
import numpy as np
import librosa
import essentia.standard as es
from google import genai

logger = logging.getLogger(__name__)

class BgmIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.music_dir = self.output_base_dir / "music"
        self.music_dir.mkdir(parents=True, exist_ok=True)
        
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.location = os.getenv("GCP_LOCATION")
        
        try:
            self.llm_client = genai.Client(
                vertexai=True, 
                project=self.project_id, 
                location=self.location
            )
        except Exception as e:
            logger.warning(f"Failed to init GenAI client: {e}")
            self.llm_client = None

    def process_bgm(self, video_id: str, scene_metadata_path: str) -> list:
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {scene_metadata_path}")
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        shots = metadata.get("shots", [])
        intelligence_results = []
        
        for shot in shots:
            shot_id = shot.get("shot_id")
            
            bgm_path = self.output_base_dir / "audio" / "bgm" / f"bgm_{shot_id}.wav"
            if not bgm_path.exists():
                logger.warning(f"BGM audio not found for {shot_id}, skipping.")
                continue
                
            # Read Stage 6 output
            emotion_path = self.output_base_dir / "emotions" / f"emotion_{shot_id}.json"
            emotion_data = {}
            if emotion_path.exists():
                with open(emotion_path, 'r') as f:
                    emotion_data = json.load(f)
                    
            # 1. Extract BGM features with librosa
            music_features = self._analyze_audio(str(bgm_path))
            
            # 2. LLM Prompting for Music Curve Classification
            music_output = self._extract_music_curve(
                shot_id, music_features, emotion_data
            )
            
            if isinstance(music_output, list) and len(music_output) > 0:
                music_output = music_output[0]
            elif not isinstance(music_output, dict):
                music_output = {}
            
            scene_payload = music_output
            
            out_path = self.music_dir / f"bgm_intelligence_{shot_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            intelligence_results.append(scene_payload)

        # Save master JSON
        master_payload = {
            "status": "completed",
            "message": f"Stage 7 BGM Intelligence completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "bgm_intelligence_results": intelligence_results
        }
        
        master_path = self.music_dir / "master_bgm_intelligence.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return intelligence_results

    def _analyze_audio(self, audio_path: str) -> dict:
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=22050)
            
            if len(y) == 0:
                return {}
                
            # Energy/Loudness over time
            rms = librosa.feature.rms(y=y)[0]
            
            # Divide into 5 temporal segments to see progression
            num_segments = 5
            segment_length = max(1, len(rms) // num_segments)
            energy_progression = []
            for i in range(num_segments):
                segment = rms[i*segment_length : (i+1)*segment_length]
                if len(segment) > 0:
                    energy_progression.append(round(float(np.mean(segment)) * 100, 2))
                else:
                    energy_progression.append(0.0)
                    
            # Peak energy timestamp
            peak_idx = np.argmax(rms)
            peak_time = round(float(librosa.frames_to_time(peak_idx, sr=sr)), 2)
            
            # Essentia Processing
            try:
                audio_es = es.MonoLoader(filename=audio_path, sampleRate=22050)()
                dyn_comp, _ = es.DynamicComplexity()(audio_es)
                rhythm_extractor = es.RhythmExtractor2013(method='multifeature')
                bpm, _, _, _, _ = rhythm_extractor(audio_es)
            except Exception as e:
                logger.error(f"Essentia failed: {e}")
                dyn_comp, bpm = 0, 0

            # Spectral Centroid (High = Violins/Vocals, Low = Drums/Bass)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            avg_centroid = round(float(np.mean(centroid)), 2)
            
            # Onset Strength for Beat Drops
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            peak_onset = round(float(np.max(onset_env)), 2)
            
            # Silence
            silence_thresh = 0.01
            silence_ratio = round(float(np.sum(rms < silence_thresh) / len(rms)), 2)
            
            return {
                "energy_progression": energy_progression,
                "peak_energy_time": peak_time,
                "tempo": round(float(bpm), 2),
                "spectral_centroid": avg_centroid,
                "dynamic_complexity": round(float(dyn_comp), 2),
                "peak_onset_strength": peak_onset,
                "silence_ratio": silence_ratio
            }
        except Exception as e:
            logger.error(f"Error analyzing audio {audio_path}: {e}")
            return {}

    def _extract_music_curve(self, shot_id, music_features, emotion_data):
        # We now just return the raw features for Multimodal Fusion
        return music_features
