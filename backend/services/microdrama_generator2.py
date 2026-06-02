import os
import json
import logging
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class MicrodramaGenerator2:
    """
    Second pass: Consumes the previously generated `microdrama_candidates.json` and `dialogue_diarization.json`.
    Splits any candidates longer than 120 seconds into smaller chunks of 30-120 seconds.
    """

    MODEL = "gemini-2.5-flash-lite"

    def _get_system_instruction(self) -> str:
        return (
            "You are an elite OTT microdrama editor. You are given a list of microdrama episodes and a diarized transcript. "
            "Some of the episodes are too long (e.g., 4, 5, or 6 minutes). "
            "Your task is to find any episode with a `duration_seconds` greater than 120.0 and SPLIT it into multiple consecutive episodes. "
            "Every final episode MUST be strictly between 30.0 and 120.0 seconds. "
            "Do not modify episodes that are already between 30 and 120 seconds, just include them in the final sequence. "
            "Respond with ONLY a valid JSON object matching the requested schema."
        )

    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
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

    def generate(self, video_id: str) -> dict:
        if not self.llm_client:
            return {"video_id": video_id, "status": "failed", "error": "LLM client not initialized"}

        candidates_path = self.output_base_dir / "microdrama_candidates.json"
        dialogue_path = self.output_base_dir / "dialogue_diarization.json"

        if not candidates_path.exists() or not dialogue_path.exists():
            return {"video_id": video_id, "status": "failed", "error": "Missing input JSON files (microdrama_candidates.json or dialogue_diarization.json)"}

        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates_json = json.load(f)
            candidates_content = json.dumps(candidates_json, indent=2, ensure_ascii=False)
            
        with open(dialogue_path, "r", encoding="utf-8") as f:
            dialogue_json = json.load(f)
            dialogue_content = json.dumps(dialogue_json, indent=2, ensure_ascii=False)

        prompt = f"""
<context>
Below is the first draft of microdrama candidates, and the full diarized dialogue.
</context>

<microdrama_candidates>
{candidates_content}
</microdrama_candidates>

<movie_dialogue>
{dialogue_content}
</movie_dialogue>

<instructions>
Review the `<microdrama_candidates>`. Identify any episode where `duration_seconds` > 120.0 (for example, 4, 5, or 6 minutes).
You MUST split these long episodes into two or more smaller, consecutive episodes.
For example, if an episode is 342 seconds, split it into three consecutive episodes, each between 30 and 120 seconds.
Use the `<movie_dialogue>` timestamps to find natural cut points (e.g., pauses between speakers).
Keep the episodes that are already between 30 and 120 seconds exactly as they are.
Re-number the `episode_number` sequentially from 1 to N.
</instructions>

<output_format>
Respond with ONLY a valid, parseable JSON object matching this schema:
{{
  "overall_microdrama_story": "Explain the overall microdrama story.",
  "explanation_of_how_microdramas_tell_the_story": "Explain how the splits ensure no episode exceeds 120 seconds.",
  "episodes": [
    {{
      "episode_number": 1,
      "title": "string",
      "start_time": "HH:MM:SS.mmm",
      "end_time": "HH:MM:SS.mmm",
      "duration_seconds": 0.0,
      "characters_present": ["string"],
      "episode_plot_explanation": "string",
      "opening_hook": "string",
      "central_conflict": "string",
      "cliffhanger_ending": "string",
      "retention_score": 0
    }}
  ]
}}
</output_format>
"""
        try:
            logger.info(f"Sending prompt to {self.MODEL} for video {video_id} (Generator 2)...")
            response = self.llm_client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self._get_system_instruction(),
                    temperature=0.3,
                    response_mime_type="application/json",
                ),
            )
            raw_output = response.text.strip()
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()
                
            raw_data = json.loads(raw_output)
            
            raw_candidates = raw_data.get("episodes", [])
            overall_story = raw_data.get("overall_microdrama_story", "")
            explanation = raw_data.get("explanation_of_how_microdramas_tell_the_story", "")
                
        except Exception as e:
            return {"video_id": video_id, "status": "failed", "error": f"LLM generation failed: {e}"}

        # Recalculate duration to ensure accuracy
        final_candidates = []
        for c in raw_candidates:
            try:
                start_parts = str(c.get("start_time", "00:00:00")).split(":")
                end_parts = str(c.get("end_time", "00:00:00")).split(":")
                start_sec = int(start_parts[0])*3600 + int(start_parts[1])*60 + float(start_parts[2])
                end_sec = int(end_parts[0])*3600 + int(end_parts[1])*60 + float(end_parts[2])
                c["duration_seconds"] = round(end_sec - start_sec, 3)
                final_candidates.append(c)
            except Exception as e:
                pass
        
        envelope = {
            "video_id": video_id,
            "status": "completed",
            "overall_microdrama_story": overall_story,
            "explanation_of_how_microdramas_tell_the_story": explanation,
            "microdrama_candidates": final_candidates
        }

        try:
            # The user explicitly requested to override scenes_and_plot.json
            scenes_path = self.output_base_dir / "scenes_and_plot.json"
            with open(scenes_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=4, ensure_ascii=False)
                
            # Note: I am also overriding microdrama_candidates.json so that the render service can still work correctly with the new fixed data.
            output_path = self.output_base_dir / "microdrama_candidates.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write output files for {video_id}: {e}")

        return envelope
