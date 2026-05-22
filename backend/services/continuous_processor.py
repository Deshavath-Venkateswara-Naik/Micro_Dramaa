import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class ContinuousIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.continuous_dir = self.output_base_dir / "continuous"
        self.continuous_dir.mkdir(parents=True, exist_ok=True)
        
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

    def process_continuous_story(self, video_id: str, scene_metadata_path: str) -> list:
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {scene_metadata_path}")
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        scenes = metadata.get("scenes", [])
        
        # Build global movie timeline context
        movie_timeline = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            
            # Load Intelligence
            emotion_data = self._read_json(self.output_base_dir / "emotions" / f"emotion_{scene_id}.json")
            bgm_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{scene_id}.json")
            narrative_data = self._read_json(self.output_base_dir / "narrative" / f"narrative_intelligence_{scene_id}.json")
            drama_data = self._read_json(self.output_base_dir / "drama" / f"drama_score_{scene_id}.json")
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            
            # Extract basic dialog to understand the story
            speech_intel = speech_data.get("intelligence_results", [{}])[0] if speech_data else {}
            dialogue_text = speech_intel.get("text", "No dialogue")
            
            timeline_event = {
                "scene_id": scene_id,
                "start_time": scene.get("start", "00:00:00"),
                "end_time": scene.get("end", "00:00:00"),
                "dialogue_summary": dialogue_text,
                "emotion_curve": emotion_data.get("emotion_curve", "neutral"),
                "bgm_intensity": bgm_data.get("intensity", 0),
                "narrative_arc": narrative_data.get("arc_type", "neutral"),
                "drama_score": drama_data.get("final_drama_score", 0)
            }
            movie_timeline.append(timeline_event)
            
        # Feed the entire movie timeline to LLM
        llm_response = self._extract_continuous_logic(movie_timeline)
        
        episodes = llm_response.get("episodes", [])
        
        # Save output
        out_path = self.continuous_dir / "continuous_micro_drama_roadmap.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(llm_response, f, indent=4, ensure_ascii=False)
            
        return episodes

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        return {}
        
    def _extract_continuous_logic(self, movie_timeline):
        if not self.llm_client:
            return self._mock_continuous_output()
            
        # Convert timeline to string for prompt
        timeline_str = json.dumps(movie_timeline, indent=2, ensure_ascii=False)
        
        prompt = f"""
        You are a professional OTT short-drama editor. Your goal is to transform a 3-hour movie into an emotionally addictive, continuous short-form dramatic series.
        
        --- MOVIE TIMELINE CUES ---
        {timeline_str}
        
        --- CONTINUOUS ENGINE RULES ---
        1. Merge related scenes to create continuous 30 to 90-second episodes.
        2. Preserve story continuity, emotional flow, and suspense.
        3. Identify parts to trim: Remove dead air, slow walking, filler dialogues.
        4. Every episode must end on a cliffhanger or a strong hook.
        5. Assign specific continuity scores to ensure the flow makes sense.
        
        INSTRUCTIONS:
        Analyze the full timeline. Group the `scene_id`s into logical "episodes". Provide exact instructions on what to trim and identify the hook/cliffhanger for the episode. 
        Assign scores to each episode.
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "series_title": "string",
            "total_episodes": integer,
            "episodes": [
                {{
                    "episode_number": integer,
                    "scenes_included": ["SC_001", "SC_002"],
                    "trim_instructions": "e.g., Remove first 5 seconds of SC_001",
                    "hook": "string description",
                    "cliffhanger": "string description",
                    "scores": {{
                        "emotion_score": integer 0-100,
                        "hook_score": integer 0-100,
                        "suspense_score": integer 0-100,
                        "viral_score": integer 0-100,
                        "dialogue_impact": integer 0-100,
                        "reaction_strength": integer 0-100,
                        "continuity_score": integer 0-100
                    }}
                }}
            ]
        }}
        """
        
        try:
            # Using Gemini 2.5 Pro for massive context window and complex reasoning
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"Continuous Engine LLM failed: {e}")
            return self._mock_continuous_output()

    def _mock_continuous_output(self):
        return {
            "series_title": "The Betrayal Series",
            "total_episodes": 1,
            "episodes": [
                {
                    "episode_number": 1,
                    "scenes_included": ["SC_001", "SC_002"],
                    "trim_instructions": "Remove 00:05-00:10 walking scene from SC_001",
                    "hook": "Shocking dialogue intro",
                    "cliffhanger": "Desperate plea ends the episode",
                    "scores": {
                        "emotion_score": 92,
                        "hook_score": 88,
                        "suspense_score": 91,
                        "viral_score": 86,
                        "dialogue_impact": 90,
                        "reaction_strength": 95,
                        "continuity_score": 89
                    }
                }
            ]
        }
