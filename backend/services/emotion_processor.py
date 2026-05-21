import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class EmotionIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.emotion_dir = self.output_base_dir / "emotions"
        self.emotion_dir.mkdir(parents=True, exist_ok=True)
        
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

    def process_emotion(self, video_id: str, scene_metadata_path: str) -> list:
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
            
            # Read Stage outputs
            face_data = self._read_json(self.output_base_dir / "faces" / f"face_intelligence_{scene_id}.json")
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            audio_data = self._read_json(self.output_base_dir / "audio" / "features" / f"features_{scene_id}.json")
            
            if not all([face_data, speech_data, audio_data]):
                logger.warning(f"Missing multimodal data for {scene_id}, skipping.")
                continue

            # Extract intelligence from Speech
            speech_intel = speech_data.get("intelligence_results", [{}])[0]
            dialogue_type = speech_intel.get("dialogue_type", "general")
            dialogue_emotion = speech_intel.get("emotion", "neutral")
            dialogue_impact = speech_intel.get("impact_score", 0)
            delivery_intensity = speech_intel.get("delivery_intensity", 0)
            
            # Extract intelligence from Audio
            audio_features = audio_data.get("audio_features", {})
            bgm_elevation = audio_features.get("bgm_elevation_score", 0)
            dramatic_silence = audio_features.get("dramatic_silence", False)
            audio_emotion_intensity = audio_features.get("emotion_intensity", 0)
            
            # Extract intelligence from Faces
            reaction_timeline = face_data.get("reaction_timeline", [])
            closeups_count = 0
            for t in reaction_timeline:
                for face in t.get("faces", []):
                    if face.get("is_closeup"):
                        closeups_count += 1
                        
            # Weighted Scoring Heuristic
            # Example: dialogue emotion=35%, face=25%, voice=20%, bgm=15%, silence=5%
            # Normalize inputs to 0-100 scale
            n_dialogue = dialogue_impact
            n_face = min(100, closeups_count * 5) # Heuristic for face impact
            n_voice = delivery_intensity
            n_bgm = bgm_elevation * 100
            n_silence = 100 if dramatic_silence else 0
            
            weighted_score = (n_dialogue * 0.35) + (n_face * 0.25) + (n_voice * 0.20) + (n_bgm * 0.15) + (n_silence * 0.05)
            
            cinematic_scores = face_data.get("cinematic_scores", {})
            
            # LLM Prompting for Curve Tracking
            emotion_output = self._extract_emotional_curve(
                scene_id,
                dialogue_type, dialogue_emotion, dialogue_impact, delivery_intensity,
                bgm_elevation, dramatic_silence, closeups_count,
                reaction_timeline, cinematic_scores, weighted_score
            )
            
            scene_payload = {
                "scene_id": scene_id,
                "status": "completed",
                "weighted_emotional_score": round(weighted_score, 2),
                "emotion_curve": emotion_output.get("emotion_curve", "unknown"),
                "dominant_emotions": emotion_output.get("dominant_emotions", []),
                "emotion_progression": emotion_output.get("emotion_progression", []),
                "peak_intensity": emotion_output.get("peak_intensity", 0),
                "viral_potential": emotion_output.get("viral_potential", 0),
                "peak_timestamp": emotion_output.get("peak_timestamp", 0.0)
            }
            
            out_path = self.emotion_dir / f"emotion_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            intelligence_results.append(scene_payload)

        # Save master JSON
        master_payload = {
            "status": "completed",
            "message": f"Stage 6 Emotion Intelligence completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "emotion_intelligence_results": intelligence_results
        }
        
        master_path = self.emotion_dir / "master_emotion_intelligence.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return intelligence_results

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        return {}
        
    def _extract_emotional_curve(self, scene_id, d_type, d_emo, d_impact, v_intensity, 
                                 bgm_elevation, silence, closeups, timeline, cinematic_scores, weighted_score):
        if not self.llm_client:
            return self._mock_emotion_output()
            
        # Simplify timeline
        simplified_timeline = []
        for t in timeline:
            faces_str = [f"{f['actor_id']} (Closeup: {f['is_closeup']})" for f in t.get("faces", [])]
            if faces_str:
                simplified_timeline.append(f"At {t['timestamp_sec']}s: " + " | ".join(faces_str))
            
        prompt = f"""
        You are a Senior Cinematic Director specializing in emotional intelligence for film.
        Analyze the fused multimodal signals for scene {scene_id} and map its emotional curve.
        
        We are tracking cinematic emotional progression patterns such as:
        - calm -> tension rise -> silence -> explosion
        - calm -> romance -> betrayal -> emotional collapse
        - fear build-up -> suspense -> reveal -> silence
        
        MULTIMODAL SIGNALS:
        1. Dialogue Meaning: {d_type}, Impact Score: {d_impact}/100, Emotion: {d_emo}
        2. Voice Emotion/Delivery Intensity: {v_intensity}/100
        3. Background Music (BGM) Elevation Score: {bgm_elevation} (0.0 to 1.0)
        4. Dramatic Silence Detected: {silence}
        5. Face Reactions: {closeups} close-up shots detected.
        6. Aggregated Weighted Emotion Score: {weighted_score}/100
        
        Face Timeline:
        {json.dumps(simplified_timeline[:30], indent=2)} # Truncated for context
        
        Provide your analysis ONLY as a valid JSON object matching this schema exactly:
        {{
            "emotion_curve": "explosive_payoff | silent_breakdown | suspense_peak | romance_peak | betrayal_moment | emotional_reveal | heroic_elevation",
            "dominant_emotions": ["list", "of", "strings"],
            "emotion_progression": ["step1", "step2", "step3", "step4"],
            "peak_intensity": integer 0-100,
            "viral_potential": integer 0-100,
            "peak_timestamp": float (estimate when the peak occurs based on timeline, or 0.0)
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
            logger.error(f"Cinematic Curve LLM failed: {e}")
            return self._mock_emotion_output()

    def _mock_emotion_output(self):
        return {
            "emotion_curve": "suspense_peak",
            "dominant_emotions": ["suspense", "tension"],
            "emotion_progression": ["calm", "tension_rise", "silence", "explosion"],
            "peak_intensity": 85,
            "viral_potential": 80,
            "peak_timestamp": 0.0
        }
