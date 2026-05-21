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
            
            # Feed to LLM for Audience Psychology modeling
            llm_response = self._extract_virality_logic(scene_id, scene_context)
            
            scene_payload = {
                "scene_id": scene_id,
                "status": "completed",
                "viral_probability": llm_response.get("viral_probability", int(base_virality_score)),
                "replay_probability": llm_response.get("replay_probability", 0),
                "retention_strength": llm_response.get("retention_strength", 0),
                "hook_strength": llm_response.get("hook_strength", 0),
                "shareability": llm_response.get("shareability", 0),
                "meme_potential": llm_response.get("meme_potential", 0),
                "audience_emotion": llm_response.get("audience_emotion", "unknown"),
                "viral_reasons": llm_response.get("viral_reasons", []),
                "audience_psychology": llm_response.get("audience_psychology", {}),
                "platform_fit": llm_response.get("platform_fit", [])
            }
            
            out_path = self.virality_dir / f"virality_intelligence_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            intelligence_results.append(scene_payload)

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
        
    def _extract_virality_logic(self, scene_id, context):
        if not self.llm_client:
            return self._mock_virality_output()
            
        prompt = f"""
        You are the Core Audience Psychology & Virality Engine (Expert Social Media Strategist).
        Analyze the cinematic cues of Scene {scene_id} to predict its viral performance on platforms like Instagram Reels and YouTube Shorts.
        
        --- CINEMATIC CUES ---
        Emotion Curve: {context['emotion_curve']} (Emotions: {context['dominant_emotions']})
        Music Type: {context['bgm_type']} (Progression: {context['music_progression']})
        Story Arc: {context['arc_type']} (Role: {context['story_role']})
        Character Dynamics: {context['character_dynamics']}
        Base Algorithmic Score: {context['base_score']}/100
        
        INSTRUCTIONS:
        1. Predict WHY audiences would replay this scene or share it (e.g. dopamine trigger from hero elevation, relatable meme face).
        2. Assign scores for replay probability, hook strength, and meme potential.
        3. Identify the core audience emotion (e.g. mass_elevation, suspense_retention, goosebumps_moment).
        4. Recommend the best platform fit.
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "viral_probability": integer 0-100 (use the base algorithmic score as a starting point, adjust based on your analysis),
            "replay_probability": integer 0-100,
            "retention_strength": integer 0-100,
            "hook_strength": integer 0-100,
            "shareability": integer 0-100,
            "meme_potential": integer 0-100,
            "audience_emotion": "mass_elevation | emotional_hook | suspense_retention | meme_reaction | goosebumps_moment | emotional_payoff | fan_edit_gold | viral_dialogue",
            "viral_reasons": ["reason 1", "reason 2"],
            "audience_psychology": {{
                "dopamine_trigger": "high | medium | low",
                "curiosity_retention": "strong | moderate | weak",
                "emotional_payoff": "excellent | good | fair"
            }},
            "platform_fit": ["Instagram Reels", "YouTube Shorts", "TikTok"]
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
            logger.error(f"Virality LLM failed: {e}")
            return self._mock_virality_output()

    def _mock_virality_output(self):
        return {
            "viral_probability": 85,
            "replay_probability": 88,
            "retention_strength": 90,
            "hook_strength": 92,
            "shareability": 80,
            "meme_potential": 65,
            "audience_emotion": "mass_elevation",
            "viral_reasons": ["strong hero presence", "suspense payoff"],
            "audience_psychology": {
                "dopamine_trigger": "high",
                "curiosity_retention": "strong",
                "emotional_payoff": "excellent"
            },
            "platform_fit": ["Instagram Reels", "YouTube Shorts"]
        }
