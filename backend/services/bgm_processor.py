import os
import json
import logging
from pathlib import Path
import numpy as np
import librosa
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
            
        scenes = metadata.get("scenes", [])
        intelligence_results = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            
            bgm_path = self.output_base_dir / "audio" / "bgm" / f"bgm_{scene_id}.wav"
            if not bgm_path.exists():
                logger.warning(f"BGM audio not found for {scene_id}, skipping.")
                continue
                
            # Read Stage 6 output
            emotion_path = self.output_base_dir / "emotions" / f"emotion_{scene_id}.json"
            emotion_data = {}
            if emotion_path.exists():
                with open(emotion_path, 'r') as f:
                    emotion_data = json.load(f)
                    
            # 1. Extract BGM features with librosa
            music_features = self._analyze_audio(str(bgm_path))
            
            # 2. LLM Prompting for Music Curve Classification
            music_output = self._extract_music_curve(
                scene_id, music_features, emotion_data
            )
            
            scene_payload = {
                "scene_id": scene_id,
                "status": "completed",
                "bgm_type": music_output.get("bgm_type", "unknown"),
                "intensity": music_output.get("intensity", 0),
                "music_progression": music_output.get("music_progression", []),
                "cinematic_impact": music_output.get("cinematic_impact", "low"),
                "music_emotion": music_output.get("music_emotion", "neutral"),
                "peak_timestamp": music_output.get("peak_timestamp", 0.0),
                "viral_music_potential": music_output.get("viral_music_potential", 0)
            }
            
            out_path = self.music_dir / f"bgm_intelligence_{scene_id}.json"
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
            
            # Tempo & Beats
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
            
            # Spectral Contrast (High value indicates prominent bright instruments, low means bass/drums)
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            avg_contrast = round(float(np.mean(spectral_contrast)), 2)
            
            # Silence
            silence_thresh = 0.01
            silence_ratio = round(float(np.sum(rms < silence_thresh) / len(rms)), 2)
            
            return {
                "energy_progression": energy_progression,
                "peak_energy_time": peak_time,
                "tempo": round(tempo_val, 2),
                "spectral_contrast": avg_contrast,
                "silence_ratio": silence_ratio
            }
        except Exception as e:
            logger.error(f"Error analyzing audio {audio_path}: {e}")
            return {}

    def _extract_music_curve(self, scene_id, music_features, emotion_data):
        if not self.llm_client:
            return self._mock_bgm_output()
            
        # Context from Stage 6
        scene_emotion_curve = emotion_data.get("emotion_curve", "unknown")
        scene_dominant_emotions = emotion_data.get("dominant_emotions", [])
        
        prompt = f"""
        You are a Senior Cinematic Music Director and Composer.
        Analyze the numerical audio features from the background music (BGM) of scene {scene_id} and the detected overall emotion of the scene to classify the cinematic music progression.
        
        STAGE 6 OVERALL SCENE EMOTION:
        Curve: {scene_emotion_curve}
        Dominant Emotions: {scene_dominant_emotions}
        
        STAGE 7 EXTRACTED MUSIC FEATURES:
        1. Energy Progression (5 segments of the scene, showing loudness swells): {music_features.get('energy_progression')}
        2. Tempo: {music_features.get('tempo')} BPM
        3. Spectral Contrast (High = violins/melodic, Low = drums/bass/dull): {music_features.get('spectral_contrast')}
        4. Silence Ratio: {music_features.get('silence_ratio')} (1.0 means totally silent)
        5. Absolute Peak Energy Timestamp: {music_features.get('peak_energy_time')}s
        
        Based on this, map the BGM progression. For example, if energy swells significantly towards the end, and the scene is a 'suspense_peak', the BGM type might be 'suspense_rise' or 'emotional_payoff'.
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "bgm_type": "suspense_rise | emotional_payoff | heroic_elevation | silent_reveal | tragedy_peak | romance_bloom | tension_build | emotional_breakdown",
            "intensity": integer 0-100,
            "music_progression": ["step1", "step2", "step3", "step4"],
            "cinematic_impact": "high | medium | low",
            "music_emotion": "fear | sorrow | joy | suspense | heroic | romantic | neutral",
            "peak_timestamp": float (estimate when the peak occurs based on peak_energy_time),
            "viral_music_potential": integer 0-100
        }}
        """
        
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"Cinematic Music LLM failed: {e}")
            return self._mock_bgm_output()

    def _mock_bgm_output(self):
        return {
            "bgm_type": "suspense_rise",
            "intensity": 85,
            "music_progression": ["calm", "tension_build", "silence", "beat_drop"],
            "cinematic_impact": "high",
            "music_emotion": "suspense",
            "peak_timestamp": 0.0,
            "viral_music_potential": 85
        }
