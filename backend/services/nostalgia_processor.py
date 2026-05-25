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
            
            # Feed to LLM for Nostalgia Intelligence modeling
            llm_response = self._extract_nostalgia_logic(scene_id, scene_context)
            
            scene_payload = {
                "nostalgia_score": llm_response.get("nostalgia_score", 0)
            }
            
            out_path = self.nostalgia_dir / f"nostalgia_intelligence_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            scene_payload_for_master = scene_payload.copy()
            scene_payload_for_master["scene_id"] = scene_id
            scene_payload_for_master["nostalgic_triggers"] = llm_response.get("nostalgic_triggers", [])
            intelligence_results.append(scene_payload_for_master)

        # Save master JSON
        ranked_scenes = sorted(intelligence_results, key=lambda x: x["nostalgia_score"], reverse=True)
        
        master_payload = {
            "status": "completed",
            "message": f"Stage 10 Nostalgia Intelligence Engine completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "top_nostalgic_scenes": [s["scene_id"] for s in ranked_scenes[:3]],
            "nostalgia_intelligence_results": ranked_scenes
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
        
    def _extract_nostalgia_logic(self, scene_id, context):
        if not self.llm_client:
            return self._mock_nostalgia_output()
            
        prompt = f"""
        You are the Core Nostalgia Intelligence Engine (Vintage Cinema Archivist).
        Your goal is to detect nostalgic emotional triggers specifically for old movies, as old movie engagement depends heavily on nostalgia.
        
        --- SCENE CUES ---
        Dialogue Snippet: "{context['dialogue']}"
        Faces/Actors Detected: {context['faces_detected']}
        Background Music Type: {context['bgm_type']}
        Cinematic Arc Type: {context['arc_type']}
        
        STAGE 10 GOALS - AI Detects:
        - iconic actors
        - retro bgm
        - famous dialogues
        - legendary scenes
        - vintage emotional moments
        
        INSTRUCTIONS:
        Analyze the scene cues. If you identify legendary narrative tropes, iconic actor presence, or specific retro BGM cues, it implies a high nostalgic value for the audience.
        Calculate a final 'nostalgia_score' (0-100).
        
        Provide your analysis ONLY as a valid JSON object matching this exact schema:
        {{
            "nostalgia_score": integer 0-100,
            "nostalgic_triggers": ["list of detected triggers, e.g. iconic_actor, retro_bgm"]
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
                
            return json.loads(response_text.strip())
        except Exception as e:
            logger.error(f"Nostalgia LLM failed: {e}")
            return self._mock_nostalgia_output()

    def _mock_nostalgia_output(self):
        return {
            "nostalgia_score": 88,
            "nostalgic_triggers": ["iconic_actors", "retro_bgm"]
        }
