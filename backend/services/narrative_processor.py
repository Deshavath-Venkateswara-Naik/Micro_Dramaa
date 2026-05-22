import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class NarrativeIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.narrative_dir = self.output_base_dir / "narrative"
        self.narrative_dir.mkdir(parents=True, exist_ok=True)
        
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

    def process_narrative(self, video_id: str, scene_metadata_path: str) -> list:
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {scene_metadata_path}")
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        scenes = metadata.get("scenes", [])
        intelligence_results = []
        
        # Long-Term Story Memory
        story_memory = {
            "active_arcs": [],
            "character_dynamics": {},
            "unresolved_tensions": [],
            "past_scenes": []
        }
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            
            # Gather all previous stage data for this scene
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            emotion_data = self._read_json(self.output_base_dir / "emotions" / f"emotion_{scene_id}.json")
            bgm_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{scene_id}.json")
            
            # Extract relevant semantic information
            speech_intel = speech_data.get("intelligence_results", [{}])[0] if speech_data else {}
            dialogue_text = speech_intel.get("text", "No dialogue")
            dialogue_type = speech_intel.get("dialogue_type", "general")
            
            scene_emotion_curve = emotion_data.get("emotion_curve", "unknown")
            scene_dominant_emotions = emotion_data.get("dominant_emotions", [])
            
            bgm_type = bgm_data.get("bgm_type", "unknown")
            music_progression = bgm_data.get("music_progression", [])
            
            scene_context = {
                "dialogue": dialogue_text[:500] + "..." if len(dialogue_text) > 500 else dialogue_text,
                "dialogue_type": dialogue_type,
                "emotion_curve": scene_emotion_curve,
                "dominant_emotions": scene_dominant_emotions,
                "bgm_type": bgm_type,
                "music_progression": music_progression
            }
            
            # Feed current scene + memory to LLM
            llm_response = self._extract_narrative_logic(scene_id, scene_context, story_memory)
            
            # Update memory for next scene
            story_memory["active_arcs"] = llm_response.get("updated_active_arcs", story_memory["active_arcs"])
            story_memory["character_dynamics"] = llm_response.get("updated_character_dynamics", story_memory["character_dynamics"])
            story_memory["unresolved_tensions"] = llm_response.get("updated_unresolved_tensions", story_memory["unresolved_tensions"])
            
            # Save scene to history
            story_memory["past_scenes"].append({
                "scene_id": scene_id,
                "role": llm_response.get("story_role", "unknown"),
                "arc": llm_response.get("arc_type", "unknown")
            })
            
            scene_payload = {
                "arc_type": llm_response.get("arc_type", "unknown"),
                "narrative_importance": llm_response.get("narrative_importance", 0)
            }
            
            out_path = self.narrative_dir / f"narrative_intelligence_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            intelligence_results.append(scene_payload)

        # Save master JSON
        master_payload = {
            "status": "completed",
            "message": f"Stage 8 Narrative Intelligence completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "final_story_memory": {
                "active_arcs": story_memory["active_arcs"],
                "character_dynamics": story_memory["character_dynamics"],
                "unresolved_tensions": story_memory["unresolved_tensions"]
            },
            "narrative_intelligence_results": intelligence_results
        }
        
        master_path = self.narrative_dir / "master_narrative_intelligence.json"
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
        
    def _extract_narrative_logic(self, scene_id, scene_context, story_memory):
        if not self.llm_client:
            return self._mock_narrative_output()
            
        prompt = f"""
        You are the Core Story Brain (Senior Cinematic Narrative Director).
        Your job is to understand long-range story structure, character dynamics, and emotional payoffs.
        
        CURRENT SCENE ID: {scene_id}
        
        STAGE 8 GOALS - AI Must Understand:
        - revenge arcs
        - betrayal arcs
        - emotional arcs
        - comedy arcs
        - romance arcs
        - suspense buildup
        - payoff scenes
        - character relationships
        
        CRITICAL EXAMPLE OF STORY CONNECTION:
        Hero insulted earlier -> emotional buildup -> revenge payoff later.
        You must understand story connection, emotional meaning, and audience payoff.
        Without this narrative intelligence, we only get random highlights. With it, we get cinematic storytelling!
        
        --- CURRENT SCENE DATA ---
        Dialogue Type: {scene_context['dialogue_type']}
        Dialogue Snippet: "{scene_context['dialogue']}"
        Emotion Curve: {scene_context['emotion_curve']}
        Dominant Emotions: {scene_context['dominant_emotions']}
        BGM Type: {scene_context['bgm_type']}
        BGM Progression: {scene_context['music_progression']}
        
        --- LONG-TERM STORY MEMORY ---
        Active Arcs: {json.dumps(story_memory['active_arcs'])}
        Character Dynamics: {json.dumps(story_memory['character_dynamics'])}
        Unresolved Tensions: {json.dumps(story_memory['unresolved_tensions'])}
        Past Scenes Summary: {json.dumps(story_memory['past_scenes'])}
        
        INSTRUCTIONS:
        1. Determine how this scene connects to the past scenes (e.g., does it payoff an earlier tension?).
        2. Identify the core cinematic arc type.
        3. Score the narrative importance of this scene to the overall movie (0-100).
        4. Update the long-term memory so you remember what happened for the next scenes.
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "arc_type": "revenge_payoff | betrayal_arc | emotional_arc | comedy_arc | romance_arc | suspense_buildup | narrative_setup",
            "narrative_importance": integer 0-100,
            "story_role": "setup | buildup | climax | payoff",
            "updated_active_arcs": ["list of ongoing story threads"],
            "updated_character_dynamics": {{"char_A": "current state with char_B"}},
            "updated_unresolved_tensions": ["list of pending conflicts"]
        }}
        """
        
        try:
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
            logger.error(f"Cinematic Narrative LLM failed: {e}")
            return self._mock_narrative_output()

    def _mock_narrative_output(self):
        return {
            "arc_type": "revenge_payoff",
            "narrative_importance": 90,
            "story_role": "payoff",
            "connected_scenes": ["SC_001"],
            "scene_character_dynamics": {"hero": "dominant"},
            "narrative_progression": ["humiliation", "buildup", "payoff"],
            "audience_payoff_strength": 95,
            "cinematic_significance": "high",
            "updated_active_arcs": ["revenge completed"],
            "updated_character_dynamics": {},
            "updated_unresolved_tensions": []
        }
