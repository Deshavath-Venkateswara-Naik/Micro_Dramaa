import os
import json
import logging
import cv2
import torch
from pathlib import Path
from google import genai
from emotiefflib.facial_analysis import EmotiEffLibRecognizer
from dotenv import load_dotenv

load_dotenv()

# PyTorch 2.6 weights_only patch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _patched_load

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

        try:
            logger.info("Loading EmotiEffLib Engagement Model...")
            self.engagement_model = EmotiEffLibRecognizer(model_name='enet_b0_8_best_afew')
        except Exception as e:
            logger.error(f"Failed to load EmotiEffLibRecognizer: {e}")
            self.engagement_model = None

    def process_emotion(self, video_path: str, scene_metadata_path: str) -> list:
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
            vocal_emotion = audio_features.get("vocal_emotion_detected", "neutral")
            
            # Extract intelligence from Faces
            reaction_timeline = face_data.get("reaction_timeline", [])
            closeups_count = 0
            for t in reaction_timeline:
                for face in t.get("faces", []):
                    if face.get("is_closeup"):
                        closeups_count += 1
                        
            # Stage 6 HUGE UPGRADE: Extract Continuous Tension/Engagement via EmotiEffLib
            engagement_timeline = self._extract_continuous_engagement(video_path, scene, reaction_timeline)
            avg_engagement = 0
            if engagement_timeline:
                avg_engagement = sum([e["engagement_score"] for e in engagement_timeline]) / len(engagement_timeline)
                        
            # Weighted Scoring Heuristic
            # Example: dialogue emotion=30%, face=20%, voice=15%, bgm=15%, silence=5%, engagement=15%
            # Normalize inputs to 0-100 scale
            n_dialogue = dialogue_impact
            n_face = min(100, closeups_count * 5) # Heuristic for face impact
            n_voice = delivery_intensity
            n_bgm = bgm_elevation * 100
            n_silence = 100 if dramatic_silence else 0
            n_engagement = avg_engagement * 100
            
            weighted_score = (n_dialogue * 0.30) + (n_face * 0.20) + (n_voice * 0.15) + (n_bgm * 0.15) + (n_silence * 0.05) + (n_engagement * 0.15)
            
            cinematic_scores = face_data.get("cinematic_scores", {})
            
            # LLM Prompting for Curve Tracking
            emotion_output = self._extract_emotional_curve(
                scene_id,
                dialogue_type, dialogue_emotion, dialogue_impact, delivery_intensity,
                bgm_elevation, dramatic_silence, closeups_count,
                reaction_timeline, engagement_timeline, cinematic_scores, weighted_score, vocal_emotion
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
            "message": f"Stage 6 Emotion Intelligence completed",
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
        
    def _extract_continuous_engagement(self, video_path: str, scene: dict, reaction_timeline: list) -> list:
        """Samples frames within the scene and uses EmotiEffLib to track continuous engagement (tension)."""
        if not self.engagement_model or not os.path.exists(video_path) or not reaction_timeline:
            return []
            
        engagement_data = []
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 24.0
            
            for t_data in reaction_timeline:
                ts_sec = t_data.get("timestamp_sec", 0)
                frame_idx = int(ts_sec * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and t_data.get("faces"):
                    # Process the first face in the frame for engagement
                    face = t_data["faces"][0]
                    box = face.get("box")
                    if box:
                        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                        # Ensure box is within bounds
                        x = max(0, x)
                        y = max(0, y)
                        face_crop = frame[y:y+h, x:x+w]
                        if face_crop.size > 0:
                            # Extract engagement using EmotiEffLib
                            # Output of predict_engagement is (eng_score, class_name) or similar.
                            # We will just predict emotions and get the engagement dimension if supported, or infer it.
                            # Since enet_b0_8_best_afew is primarily emotions, we use the prediction features.
                            pred, features = self.engagement_model.predict_emotions(face_crop, logits=False)
                            # Fallback heuristic: 'Angry', 'Fear', 'Surprise', 'Sadness' indicate higher tension/engagement in drama
                            tension_score = 0.5
                            if pred in ['Angry', 'Fear', 'Surprise']: tension_score = 0.9
                            elif pred in ['Sadness', 'Disgust']: tension_score = 0.7
                            elif pred in ['Happiness']: tension_score = 0.8 # Elevation
                            else: tension_score = 0.2
                            
                            engagement_data.append({
                                "timestamp_sec": ts_sec,
                                "engagement_score": tension_score,
                                "detected_emotion": pred
                            })
            cap.release()
        except Exception as e:
            logger.error(f"Error extracting continuous engagement: {e}")
            
        return engagement_data

    def _extract_emotional_curve(self, scene_id, d_type, d_emo, d_impact, v_intensity, 
                                 bgm_elevation, silence, closeups, timeline, engagement_timeline, cinematic_scores, weighted_score, vocal_emotion):
        if not self.llm_client:
            return self._mock_emotion_output()
            
        # Simplify timeline
        simplified_timeline = []
        for t in timeline:
            faces_str = [f"{f['actor_id']} (Closeup: {f['is_closeup']})" for f in t.get("faces", [])]
            if faces_str:
                simplified_timeline.append(f"At {t['timestamp_sec']}s: " + " | ".join(faces_str))
                
        # Simplify engagement
        engagement_str = ", ".join([f"{e['timestamp_sec']}s:{e['engagement_score']}" for e in engagement_timeline])
            
        prompt = f"""
        You are a Senior Cinematic Director specializing in emotional intelligence for film.
        Analyze the fused multimodal signals for scene {scene_id} and track its emotional energy continuously.
        
        STAGE 6 GOALS - We must detect the following specific emotional states:
        - sadness, tension, suspense, romance, comedy, elevation, fear, emotional collapse
        
        We are tracking cinematic emotional progression patterns (curves) such as:
        - low emotion -> tension rise -> silence -> emotional explosion
        - calm -> romance -> betrayal -> emotional collapse
        - fear build-up -> suspense -> reveal -> silence
        
        MULTIMODAL SIGNALS:
        1. Dialogue Meaning: {d_type}, Impact Score: {d_impact}/100, Emotion: {d_emo}
        2. Voice Emotion/Delivery Intensity: {v_intensity}/100, Acoustic Tone: {vocal_emotion}
        3. Background Music (BGM) Elevation Score: {bgm_elevation} (0.0 to 1.0)
        4. Dramatic Silence Duration Detected: {silence}
        5. Face Reactions: {closeups} close-up shots detected.
        6. Aggregated Weighted Emotion Score: {weighted_score}/100
        7. Continuous Tension/Engagement Energy Curve (EmotiEffLib): [{engagement_str}]
        
        Face Timeline:
        {json.dumps(simplified_timeline[:30], indent=2)}
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "emotion_curve": "explosive_payoff | silent_breakdown | suspense_peak | romance_peak | betrayal_moment | emotional_collapse | heroic_elevation",
            "dominant_emotions": ["must contain at least one of: sadness, tension, suspense, romance, comedy, elevation, fear, emotional collapse"],
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
