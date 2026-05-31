import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class EmotionIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.faces_dir = self.output_base_dir / "faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
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
        if not self.llm_client:
            logger.error("GenAI client not initialized.")
            return []
            
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        shots = metadata.get("shots", [])
        intelligence_results = []
        
        for shot in shots:
            shot_id = shot.get("shot_id")
            raw_path = self.faces_dir / f"raw_faces_{shot_id}.json"
            
            if not raw_path.exists():
                logger.warning(f"Raw faces missing for {shot_id}, skipping.")
                continue
                
            with open(raw_path, 'r') as f:
                raw_faces = json.load(f)
                
            timeline = raw_faces.get("timeline", [])
            
            # Condense the timeline to a text summary for the LLM
            timeline_str = ""
            for item in timeline:
                timestamp = item.get("timestamp", 0)
                faces = item.get("face_details", [])
                if not faces:
                    timeline_str += f"- At {timestamp}s: No faces visible\n"
                else:
                    face_desc = []
                    for face in faces:
                        desc = f"{face['emotion']} (Closeup: {face['is_closeup']})"
                        face_desc.append(desc)
                    timeline_str += f"- At {timestamp}s: {', '.join(face_desc)}\n"

            system_prompt = """You are a Senior Entertainment AI Director specializing in Indian Cinema.
Analyze the following chronological face/emotion sequence for a scene.
Identify cinematic impact metrics.
- Hero Elevation Score (0-100): High if anger/intense closeups or dramatic stares are present.
- Mother Sentiment Score (0-100): High if crying, sadness, or emotional closeups are present.
- Virality Reaction Potential (0-100): High if there are strong emotional shifts or intense facial expressions.

Return ONLY valid JSON matching this schema:
{
    "hero_elevation_score": 0,
    "mother_sentiment_score": 0,
    "virality_reaction_potential": 0,
    "primary_cinematic_cue": "Brief description of the emotional peak"
}
"""
            user_prompt = f"Scene Timeline:\n{timeline_str}\n\nAnalyze and return JSON."

            try:
                response = self.llm_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[user_prompt],
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.4,
                        response_mime_type="application/json"
                    )
                )
                
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                    
                scene_payload = json.loads(response_text.strip())
                scene_payload["shot_id"] = shot_id
                
                out_path = self.faces_dir / f"face_intelligence_{shot_id}.json"
                with open(out_path, 'w', encoding='utf-8') as f:
                    json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                    
                intelligence_results.append(scene_payload)
            except Exception as e:
                logger.error(f"Emotion LLM failed for {shot_id}: {e}")

        # Store aggregated emotion JSON in the storage root
        emotion_json_path = self.output_base_dir / "emotion.json"
        with open(emotion_json_path, 'w', encoding='utf-8') as f:
            json.dump(intelligence_results, f, indent=4, ensure_ascii=False)

        return intelligence_results
