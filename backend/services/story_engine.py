import os
import json
import logging
from pathlib import Path
from google import genai
from google.cloud import storage
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
            self.storage_client = storage.Client(project=self.project_id)
            self.gcs_bucket_name = "videograph-ai-microdrama-assets"
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
            
        scenes = metadata.get("scenes", [])
        
        # 1. Extract Video Chunks
        chunker = VideoChunker(str(self.output_base_dir))
        chunk_paths = chunker.extract_chunks(video_path, scene_metadata_path)
        
        # 2. Gather all Data
        scene_list_data = scenes
        transcript_data = []
        emotion_data = []
        audio_events_data = []
        face_data = []
        bgm_data_list = []
        virality_data_list = []
        nostalgia_data_list = []
        drama_data_list = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            
            t_data = self._read_json(self.output_base_dir / "speech" / f"transcript_{scene_id}.json")
            if t_data: transcript_data.append(t_data)
            
            e_data = self._read_json(self.output_base_dir / "emotions" / f"emotion_{scene_id}.json")
            if e_data: emotion_data.append(e_data)
            
            a_data = self._read_json(self.output_base_dir / "audio" / "features" / f"features_{scene_id}.json")
            if a_data: audio_events_data.append(a_data)
            
            f_data = self._read_json(self.output_base_dir / "faces" / f"face_intelligence_{scene_id}.json")
            if f_data: face_data.append(f_data)
            
            b_data = self._read_json(self.output_base_dir / "music" / f"bgm_intelligence_{scene_id}.json")
            if b_data: bgm_data_list.append(b_data)
            
            v_data = self._read_json(self.output_base_dir / "virality" / f"virality_intelligence_{scene_id}.json")
            if v_data: virality_data_list.append(v_data)
            
            n_data = self._read_json(self.output_base_dir / "nostalgia" / f"nostalgia_intelligence_{scene_id}.json")
            if n_data: nostalgia_data_list.append(n_data)
            
            d_data = self._read_json(self.output_base_dir / "drama" / f"drama_score_{scene_id}.json")
            if d_data: drama_data_list.append(d_data)

        # 3. Upload Video Chunks to GCS and pass as URI for Vertex AI
        uploaded_files = []
        logger.info("Uploading video chunks to GCS for Gemini Vertex AI...")
        
        try:
            bucket = self.storage_client.bucket(self.gcs_bucket_name)
        except Exception as e:
            logger.error(f"Failed to access GCS bucket: {e}")
            bucket = None
            
        for chunk_path in chunk_paths:
            try:
                if bucket:
                    blob_name = f"{video_id}/{Path(chunk_path).name}"
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(chunk_path)
                    gcs_uri = f"gs://{self.gcs_bucket_name}/{blob_name}"
                    part = genai.types.Part.from_uri(file_uri=gcs_uri, mime_type="video/mp4")
                    uploaded_files.append(part)
                    logger.info(f"Uploaded and linked {chunk_path} as {gcs_uri}")
                else:
                    logger.warning("GCS Bucket not available, skipping video upload.")
            except Exception as e:
                logger.error(f"Failed to upload video chunk {chunk_path}: {e}")

        # 4. Prompt Engineering
        system_prompt = """You are an elite OTT microdrama editor, cinematic storytelling analyst, and viral short-form strategist.

--- CRITICAL DEFINITION & CHARACTERISTICS OF A MICRO DRAMA ---
A Micro Drama is a highly condensed, emotionally driven narrative video format (typically 30-90 seconds) engineered for mobile-first consumption, high audience retention, and rapid emotional impact. 

Your task is to analyze structured entertainment scene data to extract the perfect Micro Drama episodes.

1. DURATION & PACING & EDITING RULES: 
   - Runtime MUST be strictly between 30 and 90 seconds.
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

--- OUTPUT FORMAT REQUIREMENTS ---
You MUST respond with ONLY valid, parseable JSON. Do not include markdown formatting like ```json or any prose outside the JSON.
The JSON must be a list of candidate objects matching this exact schema:

[
  {
    "start_time": "HH:MM:SS",
    "end_time": "HH:MM:SS",
    "duration_seconds": 0,
    "binge_worthy_title": "string",
    "first_3_second_hook_caption": "string",
    "emotional_hook_description": "string",
    "central_conflict_type": "string",
    "relatable_theme": "string",
    "dramatic_peak_timestamp": "HH:MM:SS",
    "cliffhanger_ending_description": "string",
    "retention_score_0_to_100": 0,
    "characters_present": ["string"],
    "virality_and_psychology_analysis": "string"
  }
]

Failure to adhere to this JSON schema will result in system failure.
"""
        
        user_prompt = f"""Analyze the provided multimodal intelligence data from a long-form Telugu video.
Identify the BEST microdrama candidates based on the defined characteristics.

Video Metadata:
- Job ID: {video_id}
- Language: Telugu

--- MULTIMODAL INTELLIGENCE DATA ---
1. SCENE LIST:
{json.dumps(scene_list_data, indent=2)}

2. TRANSCRIPT:
{json.dumps(transcript_data, indent=2)}

3. EMOTION TIMELINE:
{json.dumps(emotion_data, indent=2)}

4. AUDIO EVENTS:
{json.dumps(audio_events_data, indent=2)}

5. FACE INTELLIGENCE:
{json.dumps(face_data, indent=2)}

6. BGM & MUSIC CUES:
{json.dumps(bgm_data_list, indent=2)}

7. VIRALITY, DRAMA, & NOSTALGIA SCORES:
{json.dumps({
    "virality": virality_data_list,
    "nostalgia": nostalgia_data_list,
    "drama": drama_data_list
}, indent=2)}

--- INSTRUCTIONS ---
Extract ALL valid high-impact microdrama candidates from the provided video.
Identify every emotionally engaging, suspenseful, retention-optimized narrative window that satisfies the defined microdrama characteristics.
Do not artificially limit the number of candidates.
Ensure each candidate is between 30 and 90 seconds.
Ensure boundaries (start_time/end_time) are frame-accurate, do not cut dialogue mid-sentence, and do not bleed into unrelated scenes.
Ensure each candidate ends on a cliffhanger.
Return strictly the JSON array as defined in the system prompt. No prose.
"""

        # Build contents array with all uploaded video chunks followed by the user prompt
        contents = uploaded_files + [user_prompt]

        logger.info("Sending data to Gemini 2.0 Flash...")
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=genai.types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=0.7
                )
            )
            
            response_text = response.text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
                
            result_json = json.loads(response_text.strip())
            
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
