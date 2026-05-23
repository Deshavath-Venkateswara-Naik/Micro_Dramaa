import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class DramaScoringProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.drama_dir = self.output_base_dir / "drama"
        self.drama_dir.mkdir(parents=True, exist_ok=True)

    def process_drama_scoring(self, video_id: str, scene_metadata_path: str, genre: str = "Action") -> list:
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
            
            # Aggregate intelligence from all stages
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            face_data = self._read_json(self.output_base_dir / "faces" / f"face_intelligence_{scene_id}.json")
            emotion_data = self._read_json(self.output_base_dir / "emotions" / f"emotion_{scene_id}.json")
            audio_data = self._read_json(self.output_base_dir / "audio" / "features" / f"features_{scene_id}.json")
            
            # --- DRAMA SCORE ALGORITHMIC FORMULA ---
            # drama_score = (
            #     emotion_intensity    * 0.35  +
            #     face_reaction_peak   * 0.20  +
            #     dialogue_aggression  * 0.20  +
            #     silence_tension      * 0.15  +
            #     audio_peak           * 0.10
            # )
            
            # 1. Emotion Intensity (0.0 to 1.0)
            emo_score = emotion_data.get("weighted_emotional_score", 0) / 100.0
            
            # 2. Face Reaction Peak (0.0 to 1.0)
            face_reaction_peak = 0.0
            if face_data and "reaction_timeline" in face_data:
                # Basic heuristic based on closeups and detected faces
                if len(face_data["reaction_timeline"]) > 5:
                    face_reaction_peak = 0.9
                else:
                    face_reaction_peak = 0.5
                    
            # 3. Dialogue Aggression (0.0 to 1.0)
            speech_intel = speech_data.get("intelligence_results", [{}])[0] if speech_data else {}
            dialogue_aggression = speech_intel.get("impact_score", 0) / 100.0
            
            # 4. Silence Tension (0.0 to 1.0)
            audio_features = audio_data.get("audio_features", {})
            silence_tension = 1.0 if audio_features.get("dramatic_silence") else 0.0
            
            # 5. Audio Peak (0.0 to 1.0)
            audio_peak = float(audio_features.get("bgm_elevation_score", 0.0))
            
            # Calculate composite score
            composite_score = (
                emo_score * 0.35 +
                face_reaction_peak * 0.20 +
                dialogue_aggression * 0.20 +
                silence_tension * 0.15 +
                audio_peak * 0.10
            )
            
            final_drama_score = min(max(int(composite_score * 100), 0), 100)
            
            scene_payload = {
                "final_drama_score": final_drama_score
            }
            
            out_path = self.drama_dir / f"drama_score_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            scene_payload_for_master = scene_payload.copy()
            scene_payload_for_master["scene_id"] = scene_id
            scene_payload_for_master["genre"] = genre
            intelligence_results.append(scene_payload_for_master)

        # Save master JSON
        ranked_scenes = sorted(intelligence_results, key=lambda x: x["final_drama_score"], reverse=True)
        
        master_payload = {
            "status": "completed",
            "message": f"Stage 11 Drama Scoring Engine completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "top_drama_scenes": [s["scene_id"] for s in ranked_scenes[:3]],
            "drama_scoring_results": ranked_scenes
        }
        
        master_path = self.drama_dir / "master_drama_scoring.json"
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
