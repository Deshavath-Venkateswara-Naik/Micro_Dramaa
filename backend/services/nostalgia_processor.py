import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class NostalgiaIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.nostalgia_dir = self.output_base_dir / "nostalgia"
        self.nostalgia_dir.mkdir(parents=True, exist_ok=True)
        
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

    def process_nostalgia(self, video_id: str, scene_metadata_path: str) -> list:
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
            
            # Read previous stage outputs
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            face_data = self._read_json(self.output_base_dir / "faces" / f"face_intelligence_{scene_id}.json")
            bgm_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{scene_id}.json")
            narrative_data = self._read_json(self.output_base_dir / "narrative" / f"narrative_intelligence_{scene_id}.json")
            
            # Extract relevant semantic information
            speech_intel = speech_data.get("intelligence_results", [{}])[0] if speech_data else {}
            dialogue_text = speech_intel.get("text", "No dialogue")
            
            # Face identities
            identities = []
            if face_data and "scene_timeline" in face_data:
                for t in face_data["scene_timeline"]:
                    if t.get("face_identity") and t["face_identity"] != "Unknown":
                        identities.append(t["face_identity"])
            unique_identities = list(set(identities))
            
            scene_context = {
                "dialogue": dialogue_text,
                "faces_detected": unique_identities,
                "bgm_type": bgm_data.get("bgm_type", "neutral"),
                "arc_type": narrative_data.get("arc_type", "unknown")
            }
            
            scene_payload = scene_context
            
            out_path = self.nostalgia_dir / f"nostalgia_intelligence_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            scene_payload_for_master = scene_payload.copy()
            scene_payload_for_master["scene_id"] = scene_id
            intelligence_results.append(scene_payload_for_master)

        # Save master JSON
        master_payload = {
            "status": "completed",
            "message": f"Stage 10 Nostalgia Intelligence Engine completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "top_nostalgic_scenes": [],
            "nostalgia_intelligence_results": intelligence_results
        }
        
        master_path = self.nostalgia_dir / "master_nostalgia_intelligence.json"
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
