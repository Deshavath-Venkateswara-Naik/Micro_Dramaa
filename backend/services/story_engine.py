import os
import json
import logging
from pathlib import Path
from moviepy.editor import VideoFileClip
from google import genai
from services.video_chunker import VideoChunker

logger = logging.getLogger(__name__)

class StoryIntelligenceEngine:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.story_dir = self.output_base_dir / "story"
        self.story_dir.mkdir(parents=True, exist_ok=True)
        
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

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        return {}

    def process_story(self, video_id: str, video_path: str, scene_metadata_path: str) -> dict:
        if not self.llm_client:
            logger.error("GenAI client not initialized.")
            return {"error": "GenAI client not initialized"}
            
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            return {"error": "Metadata not found"}
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        shots = metadata.get("shots", [])
        
        # 1. Extract Video Chunks
        chunker = VideoChunker(str(self.output_base_dir))
        _ = chunker.extract_chunks(video_path, scene_metadata_path)
        
        # 2. Gather all Data
        shot_list_data = shots
        fusion_data = []
        transcript_data = []
        bgm_data = []
        face_emotion_data = []
        
        for shot in shots:
            shot_id = shot.get("shot_id")
            
            f_data = self._read_json(self.output_base_dir / "fusion" / f"multimodal_intelligence_{shot_id}.json")
            if f_data: fusion_data.append(f_data)
            
            t_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{shot_id}.json")
            if t_data: transcript_data.append(t_data)
            
            b_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{shot_id}.json")
            if b_data: bgm_data.append(b_data)

            fe_data = self._read_json(self.output_base_dir / "faces" / f"face_intelligence_{shot_id}.json")
            if fe_data: face_emotion_data.append(fe_data)

        # Get video duration to prevent hallucinations
        try:
            with VideoFileClip(video_path) as clip:
                video_duration_sec = clip.duration
        except Exception as e:
            logger.warning(f"Could not get video duration: {e}")
            video_duration_sec = 0
            
        if video_duration_sec > 0:
            video_duration_formatted = f"{int(video_duration_sec//3600):02d}:{int((video_duration_sec%3600)//60):02d}:{int(video_duration_sec%60):02d}"
        else:
            video_duration_formatted = "UNKNOWN"

        # 3. Extract Keyframes instead of full video
        uploaded_files = []
        logger.info(f"Extracting keyframes from {video_path} for visual context...")
        
        import tempfile
        import shutil
        temp_dir = tempfile.mkdtemp(prefix="microdrama_keyframes_")
        
        def t_to_s(t):
            h,m,s = map(float, t.split(':'))
            return h*3600 + m*60 + s
            
        try:
            with VideoFileClip(video_path) as clip:
                for idx, shot in enumerate(shots):
                    s_start = t_to_s(shot.get("start") or shot.get("start_time", "00:00:00"))
                    s_end = t_to_s(shot.get("end") or shot.get("end_time", "00:00:00"))
                    
                    midpoint = (s_start + s_end) / 2.0
                    if midpoint > clip.duration:
                        midpoint = clip.duration - 0.1
                        
                    frame_path = os.path.join(temp_dir, f"frame_{idx:03d}.jpg")
                    clip.save_frame(frame_path, t=midpoint)
                    
                    with open(frame_path, "rb") as f:
                        image_bytes = f.read()
                        
                    part = genai.types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
                    uploaded_files.append(part)
                    
            logger.info(f"Successfully extracted {len(uploaded_files)} keyframes.")
        except Exception as e:
            logger.error(f"Failed to extract keyframes from {video_path}: {e}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # 4. Prompt Engineering
        system_prompt = """You are an elite OTT microdrama editor, cinematic storytelling analyst, and viral short-form strategist.

--- CRITICAL DEFINITION & CHARACTERISTICS OF A MICRO DRAMA ---
A Micro Drama is a highly condensed, emotionally driven narrative video format (typically 30-90 seconds) engineered for mobile-first consumption, high audience retention, and rapid emotional impact. 

Your task is to analyze structured entertainment scene data to extract the perfect Micro Drama episodes.

1. DURATION & PACING & EDITING RULES (STRICTLY ENFORCED): 
   - Runtime MUST be strictly between 30 and 90 seconds. This is a hard limit.
   - You must look at the timestamps of the `included_shot_ids` you select. The total time from the first scene's start to the last scene's end MUST NOT exceed 90 seconds! If the story arc is too long, cut it into two separate episodes.
   - As an editor (human or AI), you can:
     * remove unnecessary time
     * skip filler moments
     * jump across small gaps
     * compress conversations
     * cut slow pacing
   - AS LONG AS:
     * emotional continuity remains
     * the conflict still makes sense
     * the audience can emotionally follow the story

2. EMOTIONAL & NARRATIVE STRUCTURE:
   - Strong Opening Hook: MUST grab attention in the first 3 seconds.
   - Focused Conflict: Revolves around ONE central conflict or dramatic turning point.
   - Emotional Continuity: The story must feel connected, not random.
   - Minimal Exposition: Jump directly into conflict.

3. AUDIENCE RETENTION & PSYCHOLOGY:
   - High Retention Design: Utilize suspense loops and emotional peaks.
   - Cliffhanger/Payoff Ending: MUST end with an unresolved twist, reveal, or suspense hook.
   - Virality Optimization: Prioritize highly relatable themes and dramatic reactions.

4. MULTIMODAL SIGNALS TO PRIORITIZE:
   - Dialogue Efficiency: Short, emotionally powerful lines.
   - Visual Emotional Emphasis: Intense facial close-ups, crying, strong reactions.
   - Audio Importance: Emotional music swells, suspense sounds, tension cues.
   - Scene Density: High concentration of emotional moments per second.

5. PRECISION BOUNDARY CAPTURING (CRITICAL):
   - EXACT START_TIME: Must align perfectly with a distinct visual scene shift, the beginning of a crucial dialogue line, or a dramatic audio cue. DO NOT cut a character off mid-sentence. Start precisely when the emotional hook begins.
   - EXACT END_TIME: Must cut to black exactly at the peak of the cliffhanger or reaction shot. DO NOT let the clip bleed into the next unrelated scene or conversation.
   - Cross-reference Transcript timestamps with Face and Audio Intelligence timestamps to guarantee frame-accurate, narratively cohesive boundaries.
   - NOTE: You are provided with keyframe images (one per scene). Use these keyframes to verify visual continuity, character presence, and setting, but rely on the precise timestamps in the JSON for your output boundaries.

6. EPISODIC SEQUENCING RULES:
   - Identify characters and map out their relational graph.
   - You MUST order the extracted episodes strictly in chronological order based on their start_time. Never jump backward in time. (e.g., Episode 5 MUST have a start_time that occurs after Episode 4).
   - Ensure the cliffhanger of Episode N naturally links into the hook of Episode N+1.

--- OUTPUT FORMAT REQUIREMENTS ---
You MUST respond with ONLY valid, parseable JSON. Do not include markdown formatting like ```json or any prose outside the JSON.
The JSON must be a raw array of candidate objects matching this exact schema:

[
  {
    "binge_worthy_title": "string",
    "included_shot_ids": ["SC_001", "SC_002"],
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS",
    "duration_seconds": 0,
    "first_3_second_hook_caption": "string",
    "emotional_hook_description": "string",
    "central_conflict_type": "string",
    "relatable_theme": "string",
    "dramatic_peak_timestamp": "HH:MM:SS",
    "cliffhanger_ending_description": "string",
    "retention_score_0_to_100": 0,
    "boring_part_time_seconds": 0,
    "characters_present": ["string"],
    "virality_and_psychology_analysis": "string"
  }
]

Failure to adhere to this JSON array schema will result in system failure.
"""
        
        user_prompt = f"""Analyze the provided multimodal intelligence data and chronological keyframes from a long-form Telugu video.
Identify the BEST microdrama candidates based on the defined characteristics.

Video Metadata:
- Job ID: {video_id}
- Language: Telugu
- Total Duration: {video_duration_formatted} ({video_duration_sec} seconds)

--- MULTIMODAL INTELLIGENCE DATA ---
1. SCENE LIST (Each scene corresponds sequentially to the uploaded keyframe images):
{json.dumps(shot_list_data, indent=2)}

2. MULTIMODAL FUSION INTELLIGENCE:
{json.dumps(fusion_data, indent=2)}

3. FACE & EMOTION INTELLIGENCE (Cinematic Cues & Hero Elevation):
{json.dumps(face_emotion_data, indent=2)}

4. TRANSCRIPT DATA:
{json.dumps(transcript_data, indent=2)}

5. RAW BGM INTELLIGENCE DATA:
{json.dumps(bgm_data, indent=2)}

--- INSTRUCTIONS ---
Extract ALL valid high-impact microdrama candidates from the provided video.
Identify every emotionally engaging, suspenseful, retention-optimized narrative window that satisfies the defined microdrama characteristics.
Do not artificially limit the number of candidates.
Identify any filler segments within candidates and specify their duration in `boring_part_time_seconds`.
CRITICAL DURATION RULE: Each episode MUST strictly be between 30 and 100 seconds. When selecting `included_shot_ids`, manually verify that the total duration does not exceed 100 seconds. Do not create 2-minute or 5-minute episodes. Keep them extremely short and punchy.
For each episode, you MUST select the exact scene IDs from the SCENE LIST that make up the episode and put them in `included_shot_ids`.
CRITICAL TIMING RULE: DO NOT hallucinate timestamps! The video is exactly {video_duration_formatted} ({video_duration_sec}s) long. Any timestamp exceeding this limit is physically impossible and will crash the pipeline. All timestamps MUST match exactly with the provided transcript chunks.
Ensure each candidate ends on a cliffhanger.
Return strictly the JSON array of chronologically ordered episodes as defined in the system prompt. No prose.
"""

        # Build contents array with all uploaded video chunks followed by the user prompt
        contents = uploaded_files + [user_prompt]

        logger.info("Sending data to gemini-2.5-flash-lite...")
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7,
                    response_mime_type="application/json"
                )
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:].strip()
            if response_text.endswith("```"):
                response_text = response_text[:-3].strip()
                
            result_json = json.loads(response_text.strip())
            
            # Post-process timestamps and enforce strict 30-100s limits via slicing
            shot_dict = {s["shot_id"]: s for s in shots}
            
            episodes = result_json if isinstance(result_json, list) else result_json.get("episodes", [])
            final_episodes = []
            
            def t_to_s(t):
                h,m,s = map(float, t.split(':'))
                return h*3600 + m*60 + s
            
            for ep in episodes:
                included_ids = ep.get("included_shot_ids", [])
                if not included_ids:
                    continue
                    
                current_chunk_ids = []
                chunk_start_sec = None
                part = 1
                
                for sid in included_ids:
                    if sid not in shot_dict:
                        continue
                    
                    shot = shot_dict[sid]
                    s_start = t_to_s(shot.get("start") or shot.get("start_time", "00:00:00"))
                    s_end = t_to_s(shot.get("end") or shot.get("end_time", "00:00:00"))
                    
                    if chunk_start_sec is None:
                        chunk_start_sec = s_start
                        
                    # If adding this scene exceeds 100s
                    if (s_end - chunk_start_sec > 100):
                        if len(current_chunk_ids) > 0:
                            # Flush current chunk
                            new_ep = dict(ep)
                            new_ep["included_shot_ids"] = list(current_chunk_ids)
                            if part > 1 or (s_end - chunk_start_sec > 100):
                                new_ep["binge_worthy_title"] = f"{ep.get('binge_worthy_title', 'Episode')} (Part {part})"
                                
                            c_start = shot_dict[current_chunk_ids[0]].get("start") or shot_dict[current_chunk_ids[0]].get("start_time", "00:00:00")
                            c_end = shot_dict[current_chunk_ids[-1]].get("end") or shot_dict[current_chunk_ids[-1]].get("end_time", "00:00:00")
                            new_ep["start_time"] = c_start
                            new_ep["end_time"] = c_end
                            
                            # Subtract boring_part_time_seconds if present
                            boring_time = ep.get("boring_part_time_seconds", 0)
                            base_duration = int(t_to_s(c_end) - t_to_s(c_start))
                            new_ep["duration_seconds"] = max(1, base_duration - boring_time)
                            
                            final_episodes.append(new_ep)
                            
                            # Reset for next chunk
                            current_chunk_ids = [sid]
                            chunk_start_sec = s_start
                            part += 1
                        else:
                            # Single scene itself is > 100s, have to accept it but clamp duration logic if needed
                            current_chunk_ids.append(sid)
                    else:
                        current_chunk_ids.append(sid)
                        
                # Flush remaining scenes
                if current_chunk_ids:
                    new_ep = dict(ep)
                    new_ep["included_shot_ids"] = list(current_chunk_ids)
                    if part > 1:
                        new_ep["binge_worthy_title"] = f"{ep.get('binge_worthy_title', 'Episode')} (Part {part})"
                        
                    c_start = shot_dict[current_chunk_ids[0]].get("start") or shot_dict[current_chunk_ids[0]].get("start_time", "00:00:00")
                    c_end = shot_dict[current_chunk_ids[-1]].get("end") or shot_dict[current_chunk_ids[-1]].get("end_time", "00:00:00")
                    new_ep["start_time"] = c_start
                    new_ep["end_time"] = c_end
                    
                    boring_time = ep.get("boring_part_time_seconds", 0)
                    base_duration = int(t_to_s(c_end) - t_to_s(c_start))
                    new_ep["duration_seconds"] = max(1, base_duration - boring_time)
                    
                    final_episodes.append(new_ep)
            
            result_json = final_episodes
                    
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            result_json = {"error": str(e)}

        payload = {
            "video_id": video_id,
            "status": "completed",
            "microdrama_candidates": result_json
        }
        
        out_path = self.story_dir / "story_candidates.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
            
        return payload
