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
            "strictly between 30 and 120 seconds across the movie. THIS IS THE TOP MOST RULE.\n\n"
            "CRITICAL: The plot of the movie is extremely important! Every single microdrama you generate MUST be directly and heavily related to the core narrative plot of the movie.\n"
            "You MUST base your extraction on the provided Scenes timing, Plot of the movie, and the Cinematic Script Transcript (which includes speaker diarization, emotions, and sound events).\n"
            "You MUST absolutely EXCLUDE any advertisements, sponsor integrations, title cards, end credits, and unnecessary or boring scenes. "
            "Only pick peak drama moments, intense conflicts, major revelations, or highly emotional scenes that are central to the plot. "
            "Respond with ONLY a valid, parseable JSON object."
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
        script_path = self.output_base_dir / "script_transcript.json"
        
        # Fallback to older diarization file if the new cinematic script is missing
        if not script_path.exists():
            script_path = self.output_base_dir / "dialogue_diarization.json"

        if not scenes_path.exists() or not script_path.exists():
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
            
        with open(script_path, "r", encoding="utf-8") as f:
            script_json = json.load(f)
            
            # Handle both new script_transcript.json format and old dialogue_diarization.json format
            if isinstance(script_json, dict) and ("dialogues" in script_json or "sound_events" in script_json):
                dialogue_list = script_json.get("dialogues", [])
                sound_events = script_json.get("sound_events", [])
            else:
                dialogue_list = script_json.get("dialogues", []) if isinstance(script_json, dict) else script_json
                sound_events = []
                
            for d in dialogue_list:
                try:
                    # dialogue start/end are often already floats in seconds
                    d["duration_seconds"] = round(float(d.get("end", 0.0)) - float(d.get("start", 0.0)), 3)
                except:
                    d["duration_seconds"] = 0.0
                    
            script_content = json.dumps({
                "dialogues": dialogue_list,
                "sound_events": sound_events
            }, indent=2, ensure_ascii=False)

        # Load explicit dialogue diarization file as requested by the user
        diarization_path = self.output_base_dir / "dialogue_diarization copy.json"
        if not diarization_path.exists():
            diarization_path = self.output_base_dir / "dialogue_diarization.json"
        
        diarization_content = "No dialogue diarization file found."
        if diarization_path.exists():
            with open(diarization_path, "r", encoding="utf-8") as f:
                diarization_content = json.dumps(json.load(f), indent=2, ensure_ascii=False)

        prompt = f"""
<context>
You are an elite OTT microdrama editor and retention strategist.
Below is the overarching movie plot, the scene data, the cinematic script transcript, AND the raw dialogue diarization for a {language} movie.
</context>

<movie_plot>
{plot_content}
</movie_plot>

<movie_scenes>
{scenes_content}
</movie_scenes>

<movie_script_transcript>
{script_content}
</movie_script_transcript>

<movie_dialogue_diarization>
{diarization_content}
</movie_dialogue_diarization>

<instructions>
Your task is to extract a sequential series of highly engaging microdrama 'episodes' that together tell the COMPLETE story of the movie from start to finish. 
You are receiving a massive amount of multimodal data. You MUST understand the purpose of each data source and use them together:

1. <movie_plot>: This is your NARRATIVE COMPASS. It tells you the core story. Every episode you create MUST drive this plot forward. Ignore any scenes that do not contribute to this plot.
2. <movie_scenes>: This is your STRUCTURAL MAP. It summarizes the movie scene-by-scene, showing characters, settings, and emotions. Use this to quickly locate the dramatic peaks, fights, and emotional resolutions without getting lost in the raw text.
3. <movie_script_transcript> & <movie_dialogue_diarization>: These are your PRECISION SCALPELS. They contain the exact millisecond timestamps of every spoken word, the speaker's emotion, and background sound events (like Music or Sirens). Use these to find the EXACT start and end boundaries for your episodes so that no dialogue is cut off mid-sentence.

Methodology: 
Step 1: Read the plot to know the story.
Step 2: Look at the scenes to find the chunks of time that tell that story.
Step 3: Dive into the transcript/diarization for those specific times to set your exact episode 'start_time' and 'end_time' to the millisecond.
</instructions>

<rules>
  <rule_1>COMPLETE PLOT COVERAGE & RELEVANCE: You MUST capture the entire movie's plot. Every single generated episode MUST be directly related to the core plot. Do not select random side-scenes or disconnected engaging moments if they do not advance the main plot. The combination of all your selected microdramas must seamlessly tell the whole main story in sync, from beginning to end.</rule_1>
  <rule_2>STRICT DURATION (TOP MOST RULE): Every single episode MUST be strictly between 30.0 and 120.0 seconds inclusive. This is non-negotiable.
    <requirement>You MUST use the supplied `duration_seconds` fields provided in the input data to calculate the exact length of your episodes.</requirement>
    <requirement>Only select episodes where 30 <= total duration_seconds <= 120.</requirement>
    <warning>If a dramatic sequence or scene is longer than 120 seconds, you MUST split it into two or more separate, consecutive episodes (e.g., 'Episode 4: Part 1' and 'Episode 5: Part 2') to ensure NO episode exceeds the 120-second limit.</warning>
  </rule_2>
  <rule_3>REMOVE UNNECESSARY SCENES: Aggressively ignore and cut out all boring dialogue, slow pacing, mundane moments, silence, and ANY form of advertisement, brand promotion, or sponsor message. The output must be purely cinematic story.</rule_3>
  <rule_4>CLEAN CUTS & BOUNDARY GROUNDING: DO NOT cut videos abruptly mid-sentence or mid-action. You must ensure that the `start_time` and `end_time` align EXACTLY with the timestamps of actual spoken dialogue or sound events found in the <movie_script_transcript> or <movie_dialogue_diarization>. Do not hallucinate timestamps.</rule_4>
  <rule_5>CHRONOLOGICAL SEQUENCE: Order the episodes chronologically by their start_time.</rule_5>
</rules>

<output_format>
Respond with ONLY a valid, parseable JSON object. The object MUST match this schema exactly:
{{
  "overall_microdrama_story": "Explain the overall microdrama story that you picked.",
  "explanation_of_how_microdramas_tell_the_story": "Explain how you told the whole story with these microdramas, without unnecessary scenes and with engaging scenes.",
  "episodes": [
    {{
      "episode_number": 1,
      "title": "string",
      "start_time": "HH:MM:SS.mmm",
      "end_time": "HH:MM:SS.mmm",
      "duration_seconds": 0.0,
      "characters_present": ["string"],
      "episode_plot_explanation": "Explain the plot for this specific microdrama and how it syncs with the overall story.",
      "opening_hook": "string",
      "central_conflict": "string",
      "cliffhanger_ending": "string",
      "retention_score": 0
    }}
  ]
}}
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
                
            raw_data = json.loads(raw_output)
            
            if isinstance(raw_data, list):
                raw_candidates = raw_data
                overall_story = ""
                explanation = ""
            else:
                raw_candidates = raw_data.get("episodes", [])
                overall_story = raw_data.get("overall_microdrama_story", "")
                explanation = raw_data.get("explanation_of_how_microdramas_tell_the_story", "")
                
        except Exception as e:
            return {"video_id": video_id, "status": "failed", "error": f"LLM generation failed: {e}"}

        raw_candidates = raw_candidates or []
        
        # We calculate the duration for the final output, but we do NOT drop them if they exceed 120s.
        final_candidates = []
        for c in raw_candidates:
            try:
                # Always recalculate duration from timestamps to avoid LLM math errors
                start_parts = str(c.get("start_time", "00:00:00")).split(":")
                end_parts = str(c.get("end_time", "00:00:00")).split(":")
                
                start_sec = int(start_parts[0])*3600 + int(start_parts[1])*60 + float(start_parts[2])
                end_sec = int(end_parts[0])*3600 + int(end_parts[1])*60 + float(end_parts[2])
                duration = end_sec - start_sec
                
                c["duration_seconds"] = round(duration, 3)
                
                # Inject precise dialogues and sound events for this episode's timeframe
                episode_dialogues = []
                for d in dialogue_list:
                    d_start = float(d.get("start", 0.0))
                    d_end = float(d.get("end", 0.0))
                    # Overlap condition: dialogue ends after episode starts, and starts before episode ends
                    if d_end > start_sec and d_start < end_sec:
                        episode_dialogues.append(d)
                c["dialogues"] = episode_dialogues
                
                episode_sound_events = []
                for se in sound_events:
                    se_start = float(se.get("start", 0.0))
                    se_end = float(se.get("end", 0.0))
                    if se_end > start_sec and se_start < end_sec:
                        episode_sound_events.append(se)
                c["sound_events"] = episode_sound_events
                
                final_candidates.append(c)
            except Exception as e:
                logger.warning(f"Failed to parse time for candidate {c.get('title')}: {e}")
        
        envelope = {
            "video_id": video_id,
            "status": "completed",
            "overall_microdrama_story": overall_story,
            "explanation_of_how_microdramas_tell_the_story": explanation,
            "microdrama_candidates": final_candidates
        }

        try:
            output_path = self.output_base_dir / "microdrama_candidates.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(envelope, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write microdrama_candidates.json for {video_id}: {e}")

        return envelope
