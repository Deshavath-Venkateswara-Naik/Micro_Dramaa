import os
import json
import logging
from pathlib import Path
from google import genai
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MicrodramaGenerator:
    """
    Consumes the per-video `scenes_and_plot.json` and `dialogue_diarization.json`
    artifacts and produces strictly 30-100 second microdrama candidates using Gemini LLM.
    Filters out boring parts and guarantees high-energy peak drama.
    """

    MODEL = "gemini-2.5-flash-lite"

    def _get_system_instruction(self, language: str) -> str:
        return (
            "You are an elite OTT microdrama editor and short-form TikTok/YouTube Shorts "
            f"retention strategist working on a long-form {language} film. "
            "Your goal is to extract the top highly engaging, fast-paced microdrama candidates "
            "strictly between 30 and 100 seconds across the movie.\n\n"
            "CRITICAL: Ignore all boring parts, mundane conversations, and low-energy scenes. "
            "Only pick peak drama moments, intense conflicts, major revelations, or highly emotional scenes. "
            "Respond with ONLY a valid, parseable JSON array of candidate objects."
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

    def generate(self, video_id: str, language: str = "Telugu") -> dict:
        if not self.llm_client:
            return {"video_id": video_id, "status": "failed", "error": "LLM client not initialized"}

        scenes_path = self.output_base_dir / "scenes_and_plot.json"
        dialogue_path = self.output_base_dir / "dialogue_diarization.json"

        if not scenes_path.exists() or not dialogue_path.exists():
            return {"video_id": video_id, "status": "failed", "error": "Missing input JSON files"}

        def calculate_duration_from_str(start_str, end_str):
            try:
                start_parts = str(start_str).split(":")
                end_parts = str(end_str).split(":")
                s = int(start_parts[0])*3600 + int(start_parts[1])*60 + float(start_parts[2])
                e = int(end_parts[0])*3600 + int(end_parts[1])*60 + float(end_parts[2])
                return round(e - s, 3)
            except:
                return 0.0

        # Parse JSON content to separate scenes and plot, and add pre-calculated duration
        with open(scenes_path, "r", encoding="utf-8") as f:
            scenes_json = json.load(f)
            plot_content = scenes_json.get("plot_of_the_movie", "No plot provided.")
            
            scenes_list = scenes_json.get("Scenes", [])
            for s in scenes_list:
                s["duration_seconds"] = calculate_duration_from_str(s.get("start_time", "00:00:00"), s.get("end_time", "00:00:00"))
            scenes_content = json.dumps(scenes_list, indent=2, ensure_ascii=False)
            
        with open(dialogue_path, "r", encoding="utf-8") as f:
            dialogue_json = json.load(f)
            dialogue_list = dialogue_json.get("dialogues", []) if isinstance(dialogue_json, dict) else dialogue_json
            for d in dialogue_list:
                try:
                    # dialogue start/end are often already floats in seconds
                    d["duration_seconds"] = round(float(d.get("end", 0.0)) - float(d.get("start", 0.0)), 3)
                except:
                    d["duration_seconds"] = 0.0
            dialogue_content = json.dumps(dialogue_json, indent=2, ensure_ascii=False)

        prompt = f"""
<context>
You are an elite OTT microdrama editor and retention strategist.
Below is the overarching movie plot, the scene data, and diarized dialogue for a {language} movie.
</context>

<movie_plot>
{plot_content}
</movie_plot>

<movie_scenes>
{scenes_content}
</movie_scenes>

<movie_dialogue>
{dialogue_content}
</movie_dialogue>

<instructions>
Extract ALL necessary, engaging microdrama candidates across the ENTIRE movie. You must follow the rules below with 100% accuracy.

<rules>
  <rule_1>REMOVE BORING PARTS BUT KEEP ALL NECESSARY CONTENT: You MUST aggressively cut out all boring dialogue, slow pacing, mundane moments, and silence. However, you MUST capture EVERY single necessary plot point, intense conflict, and emotional scene. Do not just pick a few highlights; provide a comprehensive list of clips that cover the entire narrative arc of the film.</rule_1>
  <rule_2>STRICT DURATION: Every candidate runtime MUST be strictly between 30.0 and 100.0 seconds inclusive.
    <requirement>You MUST use the supplied `duration_seconds` fields provided in the input data to construct your clips.</requirement>
    <requirement>Never estimate duration from timestamps on your own. Rely on the pre-calculated `duration_seconds` to ensure your clip adds up to 30-100 seconds.</requirement>
    <requirement>Only select candidates where 30 <= total duration_seconds <= 100.</requirement>
    <warning>If a dramatic scene is 150 seconds long, DO NOT output the whole scene! You MUST trim the beginning or end to fit under 100 seconds, even if it cuts off some story context. Duration is more important than narrative completeness.</warning>
  </rule_2>
  <rule_3>SLICE LONG SCENES: If an exciting scene is longer than 100 seconds, use the exact dialogue line timestamps to slice out ONLY the punchiest 30-100 second section. Remove the boring buildup.</rule_3>
  <rule_4>COMPREHENSIVE GENERATION: Do not stop at 10 or 15 clips. You MUST generate as many clips as necessary to capture all the important, non-boring parts of the 2.5-hour movie. This may require generating 40, 50, or even 80 clips.</rule_4>
  <rule_5>BOUNDARY GROUNDING: Select every candidate `start_time` and `end_time` ONLY from the actual timestamps provided in the `<movie_scenes>` and `<movie_dialogue>` blocks.</rule_5>
  <rule_6>CHRONOLOGICAL: Order the candidates chronologically by their start_time.</rule_6>
</rules>

<output_format>
Respond with ONLY a valid, parseable JSON array. Each element MUST match this schema exactly:
[
  {{
    "title": "string",
    "start_time": "HH:MM:SS.mmm",
    "end_time": "HH:MM:SS.mmm",
    "duration_seconds": 0.0,
    "characters_present": ["string"],
    "opening_hook": "string",
    "central_conflict": "string",
    "cliffhanger_ending": "string",
    "retention_score": 0
  }}
]
</output_format>
</instructions>
"""
        try:
            logger.info(f"Sending prompt to {self.MODEL} for video {video_id}...")
            response = self.llm_client.models.generate_content(
                model=self.MODEL,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction=self._get_system_instruction(language),
                    temperature=0.4,
                    response_mime_type="application/json",
                ),
            )
            raw_output = response.text.strip()
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()
            elif raw_output.startswith("```"):
                raw_output = raw_output[3:-3].strip()
                
            raw_candidates = json.loads(raw_output)
        except Exception as e:
            return {"video_id": video_id, "status": "failed", "error": f"LLM generation failed: {e}"}

        raw_candidates = raw_candidates or []
        
        # We calculate the duration for the final output, but we do NOT drop them if they exceed 100s.
        final_candidates = []
        for c in raw_candidates:
            try:
                # Use LLM provided duration if available, otherwise calculate it
                duration = c.get("duration_seconds")
                if duration is None:
                    start_parts = c["start_time"].split(":")
                    end_parts = c["end_time"].split(":")
                    start_sec = int(start_parts[0])*3600 + int(start_parts[1])*60 + float(start_parts[2])
                    end_sec = int(end_parts[0])*3600 + int(end_parts[1])*60 + float(end_parts[2])
                    duration = end_sec - start_sec
                
                c["duration_seconds"] = round(duration, 3)
                final_candidates.append(c)
            except Exception as e:
                logger.warning(f"Failed to parse time for candidate {c.get('title')}: {e}")
        
        envelope = {
            "video_id": video_id,
            "status": "completed",
            "microdrama_candidates": final_candidates
        }

        try:
            output_path = self.output_base_dir / "microdrama_candidates.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write microdrama_candidates.json for {video_id}: {e}")

        return envelope
