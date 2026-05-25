import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class ViralityIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.virality_dir = self.output_base_dir / "virality"
        self.virality_dir.mkdir(parents=True, exist_ok=True)
        
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

    def process_virality(self, video_id: str, scene_metadata_path: str) -> list:
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
            
            # Read Stage 6, 7, 8 outputs
            emotion_data = self._read_json(self.output_base_dir / "emotions" / f"emotion_{scene_id}.json")
            bgm_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{scene_id}.json")
            narrative_data = self._read_json(self.output_base_dir / "narrative" / f"narrative_intelligence_{scene_id}.json")
            
            # Base Heuristics
            emo_score = emotion_data.get("weighted_emotional_score", 0)
            music_viral = bgm_data.get("viral_music_potential", 0)
            narrative_payoff = narrative_data.get("audience_payoff_strength", 0)
            
            base_virality_score = (emo_score * 0.3) + (music_viral * 0.3) + (narrative_payoff * 0.4)
            
            scene_context = {
                "emotion_curve": emotion_data.get("emotion_curve", "neutral"),
                "dominant_emotions": emotion_data.get("dominant_emotions", []),
                "bgm_type": bgm_data.get("bgm_type", "neutral"),
                "music_progression": bgm_data.get("music_progression", []),
                "arc_type": narrative_data.get("arc_type", "neutral"),
                "story_role": narrative_data.get("story_role", "neutral"),
                "character_dynamics": narrative_data.get("character_dynamics", {}),
                "base_score": base_virality_score
            }
            
            scene_payload = {
                "viral_probability": int(base_virality_score)
            }
            
            out_path = self.virality_dir / f"virality_intelligence_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            scene_payload_for_master = scene_payload.copy()
            scene_payload_for_master["scene_id"] = scene_id
            intelligence_results.append(scene_payload_for_master)

        # Rank scenes by viral probability for the master file
        ranked_scenes = sorted(intelligence_results, key=lambda x: x["viral_probability"], reverse=True)

        master_payload = {
            "status": "completed",
            "message": f"Stage 9 Virality & Audience Psychology Engine completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "top_viral_scenes": [s["scene_id"] for s in ranked_scenes[:3]],
            "virality_intelligence_results": ranked_scenes
        }
        
        master_path = self.virality_dir / "master_virality_intelligence.json"
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
        
    pass
