import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class MultimodalFusionProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.fusion_dir = self.output_base_dir / "fusion"
        self.fusion_dir.mkdir(parents=True, exist_ok=True)
        
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

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        return {}

    def process_fusion(self, video_id: str, scene_metadata_path: str) -> list:
        if not self.llm_client:
            logger.error("GenAI client not initialized.")
            return []
            
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
            
            # Read Stage outputs
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{shot_id}.json")
            bgm_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{shot_id}.json")
            
            if not all([speech_data, bgm_data]):
                logger.warning(f"Missing speech or BGM data for {shot_id}, skipping.")
                continue
                
            speech_intel = speech_data.get("intelligence_results", [{}])[0] if "intelligence_results" in speech_data else speech_data
            
            # Formulate the prompt
            prompt = f"""
            You are a Master Cinematic Analyst specializing in Telugu Micro-Dramas.
            Analyze the following raw audio/speech signals for Scene {shot_id} and provide a holistic, combined intelligence score.
            
            --- MULTIMODAL SIGNALS ---
            1. Transcript & Speech Delivery: {json.dumps(speech_intel, indent=2)}
            2. BGM Features: {json.dumps(bgm_data, indent=2)}
            
            --- INSTRUCTIONS ---
            Provide a holistic analysis. Cross-reference the signals. For example, if BGM peaks and Speech delivery intensity is high, it's highly cinematic.
            Infer the expected visual emotional reaction based purely on the dialogue and the BGM.
            
            Return ONLY a valid JSON object matching this exact schema:
            {{
                "dialogue_impact": {{
                    "dialogue_type": "mass_elevation | emotional_breakdown | suspense | comedy | romance | general",
                    "impact_score": int (0-100)
                }},
                "cinematic_bgm": {{
                    "bgm_type": "elevation_bgm | emotional_violin | suspense_drums | beat_drop | silence_before_reveal | romantic_melody | tragedy_music | suspense_rise",
                    "intensity": int (0-100)
                }},
                "face_and_emotion": {{
                    "emotion_curve": "explosive_payoff | silent_breakdown | suspense_peak | romance_peak | betrayal_moment | heroic_elevation",
                    "hero_elevation_score": int (0-100),
                    "mother_sentiment_score": int (0-100)
                }},
                "virality_score": int (0-100),
                "nostalgia_score": int (0-100)
            }}
            """
            
            try:
                response = self.llm_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                    
                scene_payload = json.loads(response_text.strip())
                scene_payload["shot_id"] = shot_id
                
                out_path = self.fusion_dir / f"multimodal_intelligence_{shot_id}.json"
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                    
                intelligence_results.append(scene_payload)
            except Exception as e:
                logger.error(f"Fusion LLM failed for {shot_id}: {e}")

        # Save master JSON
        master_payload = {
            "status": "completed",
            "message": f"Multimodal Fusion Engine completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "fusion_intelligence_results": intelligence_results
        }
        
        master_path = self.fusion_dir / "master_fusion_intelligence.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return intelligence_results
