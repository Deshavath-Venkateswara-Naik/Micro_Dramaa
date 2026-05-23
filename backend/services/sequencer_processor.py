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
        logger.info("Sending candidates to Episodic Sequencing Engine (Gemini 2.5 Pro)...")
        llm_response = self._extract_episodic_logic(video_id, candidates, transcript_data)
        
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

    def _extract_episodic_logic(self, video_id, candidates, transcript_data):
        prompt = f"""
        You are the Master Showrunner (Episodic Sequencing Engine) for an OTT Microdrama series.
        Your goal is to organize isolated story candidates into a coherent, serialized short-form series.
        
        --- CANDIDATE CLIPS (From Story Engine) ---
        {json.dumps(candidates, indent=2)}
        
        --- TRANSCRIPTS (For Character Context) ---
        {json.dumps(transcript_data, indent=2)}
        
        STAGE 4 SEQUENCING RULES:
        1. Character Graph: Identify characters and their relationships based on the text.
        2. Filter & Order: Group candidates by shared arcs. Order them logically: setup -> escalation -> confrontation -> peak -> unresolved.
        3. Episodes: Assign sequential episode numbers.
        4. Cliffhanger Linking: Ensure the cliffhanger of Episode N naturally leads into the hook of Episode N+1.
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "series_title": "Binge-worthy title for the whole series",
            "character_graph": {{
                "characters": [
                    {{ "id": "char_01", "name": "Name/Role", "role": "protagonist/antagonist/etc" }}
                ],
                "relationships": [
                    {{
                        "from": "char_01",
                        "to": "char_02",
                        "type": "parent_child/lovers/rivals/etc",
                        "arc": "betrayal_discovery/etc",
                        "tension": "high/medium/low"
                    }}
                ]
            }},
            "episodes": [
                {{
                    "episode_number": 1,
                    "episode_title": "Title",
                    "candidate_reference": "Include candidate start_time or hook description so we know which clip to use",
                    "narrative_role": "setup/escalation/confrontation/peak",
                    "cliffhanger_link": "How this connects to Episode 2"
                }}
            ]
        }}
        """
        
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.7
                )
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"Episodic Sequencing LLM failed: {e}")
            return {"error": str(e)}
