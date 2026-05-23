import os
import json
import logging
from pathlib import Path
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
            
        scenes = metadata.get("scenes", [])
        
        # 1. Extract Video Chunks
        chunker = VideoChunker(str(self.output_base_dir))
        chunk_paths = chunker.extract_chunks(video_path, scene_metadata_path)
        
        # 2. Gather all Data
        scene_list_data = scenes
        transcript_data = []
        emotion_data = []
        audio_events_data = []
        ocr_data = []
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
            
            o_data = self._read_json(self.output_base_dir / "ocr" / f"ocr_{scene_id}.json")
            if o_data: ocr_data.append(o_data)
            
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

        # 3. Upload Video Chunks to Gemini
        uploaded_files = []
        logger.info("Uploading video chunks to Gemini...")
        for chunk_path in chunk_paths:
            try:
                # Using GenAI SDK file upload
                uploaded_file = self.llm_client.files.upload(file=chunk_path)
                uploaded_files.append(uploaded_file)
            except Exception as e:
                logger.error(f"Failed to upload {chunk_path}: {e}")

        # 4. Prompt Engineering
        system_prompt = """You are an elite OTT microdrama editor and cinematic storytelling analyst.

Your task is to analyze structured entertainment scene data and identify
the highest-potential segments for short-form vertical microdrama episodes.

You think like:
- A Netflix trailer editor focused on emotional pacing
- A viral short-form strategist who understands platform retention mechanics
- A cinematic storytelling expert who tracks character arcs and tension

Your goal is NOT summarization.
Your goal is to identify 30–90 second narrative windows that:
  - Start with an emotional hook strong enough to stop scrolling in 3 seconds
  - Contain escalating conflict or tension in the middle
  - End with an unresolved cliffhanger — NEVER resolve the scene

For each candidate, return:
1. Exact start_time and end_time (HH:MM:SS format)
2. Emotional hook description
3. Central conflict type
4. Dramatic peak timestamp
5. Cliffhanger ending description
6. Binge-worthy title
7. First-3-second hook caption text
8. Retention score (0–100)
9. Characters present
10. Why this clip performs on short-form platforms

PRIORITIZE scenes with:
- Facial reaction shots (shock, tears, rage)
- Emotionally loaded dialogue
- Dramatic pauses followed by music swells
- Unresolved endings
- Power reversals and betrayal reveals

AVOID scenes with:
- Slow exposition
- Resolved conclusions
- Repetitive low-emotion dialogue

Return structured JSON only. No prose.
"""
        
        user_prompt = f"""Analyze the following scene data from a long-form video.

Job ID: {video_id}
Source Title: Micro-Drama Source
Language: Telugu

SCENE LIST:
{json.dumps(scene_list_data, indent=2)}

TRANSCRIPT:
{json.dumps(transcript_data, indent=2)}

EMOTION TIMELINE:
{json.dumps(emotion_data, indent=2)}

AUDIO EVENTS:
{json.dumps(audio_events_data, indent=2)}

FACE INTELLIGENCE:
{json.dumps(face_data, indent=2)}

OCR DETECTIONS:
{json.dumps(ocr_data, indent=2)}

BGM & MUSIC CUES:
{json.dumps(bgm_data_list, indent=2)}

VIRALITY & AUDIENCE PSYCHOLOGY:
{json.dumps(virality_data_list, indent=2)}

NOSTALGIA METRICS:
{json.dumps(nostalgia_data_list, indent=2)}

DRAMA SCORING:
{json.dumps(drama_data_list, indent=2)}

Generate microdrama candidates between 30–90 seconds each.
Return JSON matching the MicrodramaCandidate schema.
"""

        # Build contents array with all uploaded video chunks followed by the user prompt
        contents = uploaded_files + [user_prompt]

        logger.info("Sending data to Gemini 2.5 Pro...")
        try:
            response = self.llm_client.models.generate_content(
                model="gemini-2.5-pro",
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
