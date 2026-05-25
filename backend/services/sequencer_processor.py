import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class EpisodicSequencingEngine:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.sequencer_dir = self.output_base_dir / "sequencer"
        self.sequencer_dir.mkdir(parents=True, exist_ok=True)
        
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

    def process_sequencer(self, video_id: str, scene_metadata_path: str) -> dict:
        if not self.llm_client:
            logger.error("GenAI client not initialized.")
            return {"error": "GenAI client not initialized"}
            
        story_candidates_path = self.output_base_dir / "story" / "story_candidates.json"
        
        if not story_candidates_path.exists():
            logger.error(f"Story candidates not found at {story_candidates_path}. Run Layer 3 first.")
            return {"error": "Story candidates not found. Run Story Engine first."}
            
        with open(story_candidates_path, 'r') as f:
            story_data = json.load(f)
            
        candidates = story_data.get("microdrama_candidates", [])
        
        # Load all transcripts for context
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            return {"error": "Metadata not found"}
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        scenes = metadata.get("scenes", [])
        transcript_data = []
        for scene in scenes:
            scene_id = scene.get("scene_id")
            t_path = self.output_base_dir / "speech" / f"transcript_{scene_id}.json"
            if t_path.exists():
                with open(t_path, 'r') as f:
                    transcript_data.append(json.load(f))
                    
        # Extract episodic sequencing logic
        logger.info("Extracting episodic sequencing from Story Engine candidates...")
        # Since story_engine.py now generates the full sequence, we just pass it through
        llm_response = story_data.get("microdrama_candidates", {})
        
        master_payload = {
            "status": "completed",
            "message": f"Layer 4 Episodic Sequencing Engine completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "episodic_series": llm_response
        }
        
        master_path = self.sequencer_dir / "master_series_sequence.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return master_payload
