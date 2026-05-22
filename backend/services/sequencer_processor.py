import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class DramaSequencerProcessor:
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

    def process_sequencer(self, video_id: str, scene_metadata_path: str) -> list:
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
            scene_duration = scene.get("duration", 0)
            
            # Aggregate intelligence from relevant stages
            speech_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            emotion_data = self._read_json(self.output_base_dir / "emotions" / f"emotion_{scene_id}.json")
            narrative_data = self._read_json(self.output_base_dir / "narrative" / f"narrative_intelligence_{scene_id}.json")
            drama_data = self._read_json(self.output_base_dir / "drama" / f"drama_score_{scene_id}.json")
            
            # Extract transcript to find dialogue timings
            speech_intel = speech_data.get("intelligence_results", [{}])[0] if speech_data else {}
            dialogue_text = speech_intel.get("text", "No dialogue")
            # In a real scenario, we would pass word-level timestamps here if WhisperX generated them.
            
            scene_context = {
                "duration": scene_duration,
                "dialogue": dialogue_text,
                "emotion_curve": emotion_data.get("emotion_curve", "neutral"),
                "emotional_peak_time": emotion_data.get("peak_timestamp", 0),
                "narrative_arc": narrative_data.get("arc_type", "neutral"),
                "narrative_role": narrative_data.get("story_role", "neutral"),
                "drama_score": drama_data.get("final_drama_score", 0)
            }
            
            # Feed to LLM for Sequence mapping
            llm_response = self._extract_sequencer_logic(scene_id, scene_context)
            
            scene_payload = {
                "hook": llm_response.get("hook", "0s-3s (Unknown Hook)"),
                "buildup": llm_response.get("buildup", "3s-12s (Unknown Buildup)"),
                "peak": llm_response.get("peak", "12s-20s (Unknown Peak)"),
                "cliffhanger": llm_response.get("cliffhanger", "20s-28s (Unknown Cliffhanger)")
            }
            
            out_path = self.sequencer_dir / f"sequencer_intelligence_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            scene_payload_for_master = scene_payload.copy()
            scene_payload_for_master["scene_id"] = scene_id
            intelligence_results.append(scene_payload_for_master)

        # Save master JSON
        master_payload = {
            "status": "completed",
            "message": f"Stage 12 AI Drama Sequencer completed for {video_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "sequencer_results": intelligence_results
        }
        
        master_path = self.sequencer_dir / "master_sequencer_intelligence.json"
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
        
    def _extract_sequencer_logic(self, scene_id, context):
        if not self.llm_client:
            return self._mock_sequencer_output()
            
        prompt = f"""
        You are the Master Reel Editor (AI Drama Sequencer).
        Your goal is to create the perfect cinematic short-form structure for a viral reel.
        This is the MOST IMPORTANT step for virality.
        
        --- SCENE CUES ---
        Scene Duration: {context['duration']} seconds
        Dialogue Snippet: "{context['dialogue']}"
        Emotion Curve: {context['emotion_curve']}
        Calculated Emotional Peak Time: {context['emotional_peak_time']} seconds
        Narrative Arc: {context['narrative_arc']} (Role: {context['narrative_role']})
        Final Drama Score: {context['drama_score']}
        
        STAGE 12 VIRAL STRUCTURE:
        1. HOOK: First 3 seconds. Needs shocking dialogue or immediate visual impact.
        2. BUILDUP: Tension escalation leading up to the peak.
        3. EMOTIONAL PEAK: The confrontation, massive elevation, or core emotional release (should align near {context['emotional_peak_time']}s).
        4. CLIFFHANGER: The final seconds. Must end abruptly or on a shocking note to force replay.
        
        INSTRUCTIONS:
        Analyze the scene data and map out the precise timestamps and descriptions for each of the 4 structural phases. Make sure the timestamps logically flow from 0s to {context['duration']}s.
        
        Provide your analysis ONLY as a valid JSON object matching this exact strict schema:
        {{
            "hook": "string indicating time range and description (e.g. '0s-3s (shocking dialogue)')",
            "buildup": "string indicating time range and description",
            "peak": "string indicating time range and description",
            "cliffhanger": "string indicating time range and description"
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
            logger.error(f"Sequencer LLM failed: {e}")
            return self._mock_sequencer_output()

    def _mock_sequencer_output(self):
        return {
            "hook": "0s-3s (shocking dialogue)",
            "buildup": "3s-12s (emotional buildup)",
            "peak": "12s-20s (confrontation)",
            "cliffhanger": "20s-28s (cliffhanger)"
        }
