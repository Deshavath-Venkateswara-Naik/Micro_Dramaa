import os
import json
import logging
from pathlib import Path
import requests
import time
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Constants
SUBTITLE_API_BASE_URL = "http://43.157.32.181:8080"
# This needs to be set to your actual public IP/domain so the Subtitle API can reach your local files
PUBLIC_SERVER_URL = os.environ.get("PUBLIC_SERVER_URL", "http://YOUR_SERVER_IP:8000")

class SpeechProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.speech_dir = self.output_base_dir / "speech"
        self.speech_dir.mkdir(parents=True, exist_ok=True)
        
    def _upload_to_gcs(self, local_audio_path: str, scene_id: str) -> str:
        """Uploads audio to GCS and returns a signed URL for the Subtitle API."""
        try:
            from google.cloud import storage
            from datetime import timedelta
            
            bucket_name = "videograph-ai-microdrama-assets"
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            # Create a unique blob name
            blob_name = f"tmp_audio/{scene_id}_{os.path.basename(local_audio_path)}"
            blob = bucket.blob(blob_name)
            
            logger.info(f"Uploading {local_audio_path} to GCS bucket {bucket_name}...")
            blob.upload_from_filename(local_audio_path)
            
            # Generate signed URL valid for 1 hour
            url = blob.generate_signed_url(
                version="v4",
                expiration=timedelta(hours=1),
                method="GET"
            )
            return url
        except Exception as e:
            logger.error(f"GCS Upload Failed: {e}")
            return ""

    def transcribe_with_subtitle_api(self, audio_path: str, scene_id: str) -> dict:
        """Uses the asynchronous Subtitle API to transcribe audio and get word timestamps."""
        public_url = self._upload_to_gcs(audio_path, scene_id)
        if not public_url:
            logger.error(f"Could not resolve public URL for {audio_path}")
            return {"transcript": "", "segments": []}
            
        logger.info(f"Submitting job to Subtitle API for {scene_id} using URL: {public_url}")
        
        # 1. Create Job
        try:
            response = requests.post(
                f"{SUBTITLE_API_BASE_URL}/jobs",
                json={
                    "url": public_url,
                    "language": "te", # Telugu
                    "word_timestamps": True
                },
                timeout=10
            )
            if response.status_code != 202:
                logger.error(f"Failed to create Subtitle API job: {response.text}")
                return {"transcript": "", "segments": []}
                
            job_data = response.json()
            job_id = job_data.get("job_id")
            
        except Exception as e:
            logger.error(f"Error creating Subtitle API job: {e}")
            return {"transcript": "", "segments": []}
            
        # 2. Poll for Completion
        logger.info(f"Job {job_id} created. Polling for completion...")
        max_retries = 60 # 60 * 5s = 5 minutes timeout per scene
        
        for _ in range(max_retries):
            time.sleep(5)
            try:
                poll_res = requests.get(f"{SUBTITLE_API_BASE_URL}/jobs/{job_id}", timeout=10)
                if poll_res.status_code == 200:
                    status_data = poll_res.json()
                    status = status_data.get("status")
                    
                    if status == "Completed":
                        result_url = status_data.get("result_url")
                        logger.info(f"Job {job_id} completed. Fetching results from {result_url}")
                        return self._fetch_and_parse_results(result_url)
                    elif status == "Failed":
                        logger.error(f"Job {job_id} failed: {status_data.get('error')}")
                        return {"transcript": "", "segments": []}
                        
            except Exception as e:
                logger.warning(f"Error polling job {job_id}: {e}")
                
        logger.error(f"Job {job_id} timed out after polling.")
        return {"transcript": "", "segments": []}
        
    def _fetch_and_parse_results(self, result_url: str) -> dict:
        """Downloads the JSON result and formats it for our pipeline."""
        if not result_url:
            return {"transcript": "", "segments": []}
            
        try:
            res = requests.get(result_url, timeout=15)
            if res.status_code == 200:
                data = res.json()
                # Subtitle API format returns 'transcript' and 'words'
                full_text = data.get("transcript", "")
                if not full_text:
                    full_text = data.get("sub_text", "")
                    
                return {
                    "transcript": full_text.strip(),
                    "segments": data.get("words", [])
                }
        except Exception as e:
            logger.error(f"Failed to fetch subtitle results from {result_url}: {e}")
            
        return {"transcript": "", "segments": []}

    def calculate_dialogue_impact(self, transcript: str, audio_features: dict, scene_id: str, aligned_data: dict = None) -> dict:
        """Saves raw speech and transcript data for downstream Multimodal Fusion."""
        logger.info(f"Extracting Speech Features for {scene_id}")
        
        # Pull features from Stage 3
        delivery_intensity = float(audio_features.get("emotion_intensity", 0.5) * 100)
        dramatic_silence = bool(audio_features.get("dramatic_silence", False))
        
        impact_payload = {
            "scene_id": scene_id,
            "text": transcript,
            "delivery_intensity": round(delivery_intensity, 1),
            "dramatic_pause_detected": dramatic_silence,
            "whisperx_alignment": aligned_data or {"segments": []}
        }

        # Wrap it in the exact structure requested by the user
        final_file_payload = {
            "status": "completed",
            "message": f"Stage 4 Speech Intelligence completed for {scene_id}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": 1,
            "intelligence_results": [
                impact_payload
            ]
        }

        # Save Full Intelligence Payload (overwriting the basic transcript file)
        out_path = self.speech_dir / f"transcript_{scene_id}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(final_file_payload, f, indent=4, ensure_ascii=False)
            
        return impact_payload

    def process_speech(self, video_path: str, scene_metadata_path: str) -> list:
        """End-to-End Stage 4 Pipeline."""
        audio_features_dir = self.output_base_dir / "audio" / "features"
        dialogue_dir = self.output_base_dir / "audio" / "dialogue"
        
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
            start = scene.get("start")
            end = scene.get("end")
            
            if start == end:
                continue
            
            audio_path = dialogue_dir / f"dialogue_{scene_id}.wav"
            feature_path = audio_features_dir / f"features_{scene_id}.json"
            
            # 1. Load Audio Features from Stage 3
            audio_features = {}
            if feature_path.exists():
                with open(feature_path, 'r') as f:
                    features_data = json.load(f)
                    audio_features = features_data.get("audio_features", {})

            # 2. STT Architecture (Subtitle API)
            stt_results = self.transcribe_with_subtitle_api(str(audio_path), scene_id)
            transcript_text = stt_results.get("transcript", "")
            
            # 3. Dialogue Impact Engine
            impact_scores = self.calculate_dialogue_impact(
                transcript=transcript_text,
                audio_features=audio_features,
                scene_id=scene_id,
                aligned_data={"segments": stt_results.get("segments", [])}
            )
            
            intelligence_results.append(impact_scores)
            
        # Create Master JSON payload
        master_payload = {
            "status": "completed",
            "message": f"Stage 4 Speech Intelligence completed for {metadata.get('video_id', 'Unknown')}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "intelligence_results": intelligence_results
        }
        
        # Save the master file that contains ALL scenes
        master_path = self.speech_dir / "speech_intelligence.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return intelligence_results
