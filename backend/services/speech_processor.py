import os
import json
import logging
from pathlib import Path
import requests
import time
from dotenv import load_dotenv

try:
    from services.script_formatter import CinematicScriptFormatter
except ImportError:
    pass

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
        
    def _transcribe_with_subtitle_api(self, media_url: str, language: str) -> dict:
        """Sends job to Subtitle API, polls for completion, and returns transcript segments."""
        import requests
        import time
        
        # Subtitle API expects a 2-char language code, e.g., "hi" instead of "hi-IN"
        lang_code = language[:2] if language else "hi"
        
        url = f"{SUBTITLE_API_BASE_URL}/jobs"
        payload = {
            "url": media_url,
            "language": lang_code,
            "word_timestamps": True
        }
        
        try:
            logger.info(f"Submitting job to Subtitle API for URL: {media_url}")
            res = requests.post(url, json=payload, timeout=30)
            res.raise_for_status()
            job_data = res.json()
            job_id = job_data.get("job_id")
            
            if not job_id:
                logger.error("No job_id returned from Subtitle API.")
                return {"transcript": "", "segments": []}
                
            logger.info(f"Job {job_id} submitted successfully. Polling for completion...")
            
            # Poll for completion
            poll_interval_seconds = 5
            timeout_minutes = 10
            max_poll_attempts = 120 # 120 * 5s = 10 minutes
            
            for attempt in range(max_poll_attempts):
                poll_url = f"{url}/{job_id}"
                poll_res = requests.get(poll_url, timeout=30)
                poll_res.raise_for_status()
                poll_data = poll_res.json()
                status = poll_data.get("status")
                
                if status == "Completed":
                    result_url = poll_data.get("result_url")
                    if not result_url:
                        logger.error("Job completed but no result_url provided.")
                        return {"transcript": "", "segments": []}
                        
                    logger.info(f"Job completed! Fetching results from {result_url}")
                    result_res = requests.get(result_url, timeout=30)
                    result_res.raise_for_status()
                    result_json = result_res.json()
                    
                    # Flatten segments into a list of word objects for diarization
                    def time_to_seconds(t_str: str) -> float:
                        try:
                            h, m, s = t_str.split(':')
                            return int(h) * 3600 + int(m) * 60 + float(s)
                        except Exception:
                            return 0.0

                    words = []
                    if "sub_text" in result_json:
                        for item in result_json["sub_text"]:
                            words.append({
                                "word": item.get("text", ""),
                                "start": time_to_seconds(item.get("start_time", "00:00:00.000")),
                                "end": time_to_seconds(item.get("end_time", "00:00:00.000"))
                            })
                    else:
                        for segment in result_json.get("segments", []):
                            for word in segment.get("words", []):
                                if "start" in word and "end" in word:
                                    words.append({
                                        "word": word.get("word", word.get("text", "")),
                                        "start": float(word["start"]),
                                        "end": float(word["end"])
                                    })
                                
                    return {"transcript": result_json.get("text", ""), "segments": words}
                elif status == "Failed":
                    error_msg = poll_data.get("error", "Unknown error")
                    logger.error(f"Subtitle API job failed: {error_msg}")
                    return {"transcript": "", "segments": []}
                    
                time.sleep(poll_interval_seconds)
                
            logger.error("Subtitle API job timed out.")
            return {"transcript": "", "segments": []}
            
        except Exception as e:
            logger.error(f"Failed to process with Subtitle API: {e}")
            return {"transcript": "", "segments": []}

    def process_speech(self, video_id: str, video_path: str = None, language: str = "hi-IN") -> dict:
        """End-to-End Stage 4 Pipeline: Global Dialogue Extraction & Diarization."""
        dialogue_dir = self.output_base_dir / "audio" / "dialogue"
        dialogue_dir.mkdir(parents=True, exist_ok=True)
        global_dialogue_path = dialogue_dir / "global_dialogue.wav"
        
        full_audio_path = self.output_base_dir / "audio" / "full_audio.wav"
        # If the clean vocals track already exists, skip audio extraction and demucs entirely
        if not global_dialogue_path.exists():
            if not full_audio_path.exists():
                if video_path and Path(video_path).exists():
                    logger.info(f"Extracting full audio from {video_path} to {full_audio_path}")
                    import subprocess
                    self.output_base_dir.joinpath("audio").mkdir(parents=True, exist_ok=True)
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(video_path),
                        "-vn",          # No video
                        "-ac", "1",     # Mono
                        "-ar", "16000", # 16kHz
                        "-c:a", "pcm_s16le", # 16-bit PCM
                        str(full_audio_path)
                    ]
                    try:
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except subprocess.CalledProcessError as e:
                        logger.error(f"FFmpeg extraction failed: {e}")
                        return {"dialogues": []}
                else:
                    logger.error(f"Cannot proceed: {global_dialogue_path} not found, {full_audio_path} not found, and no valid video_path provided.")
                    return {"dialogues": []}
                
            # Skip Demucs extraction as requested, use full audio for diarization
            global_dialogue_path = full_audio_path
                
        # 2. Transcribe with Subtitle API
        source_url_path = self.output_base_dir / "source_url.txt"
        if not source_url_path.exists():
            logger.error(f"source_url.txt not found in {self.output_base_dir}")
            return {"dialogues": []}
            
        with open(source_url_path, "r") as f:
            media_url = f.read().strip()
            
        logger.info(f"Transcribing via Subtitle API with URL: {media_url}")
        stt_results = self._transcribe_with_subtitle_api(media_url, language)
        
        # Save results locally for debugging
        final_json_path = self.output_base_dir / "audio" / "chunks" / f"{video_id}_final_merged.json"
        final_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(stt_results, f, ensure_ascii=False, indent=2)
        
        words = stt_results.get("segments", [])
        if not words:
            logger.warning("No words found in transcription for diarization.")
            return {"dialogues": []}

        # 3. Run WhisperX Diarization
        try:
            import whisperx
            from whisperx.diarize import DiarizationPipeline
            import torch
            import pandas as pd
            
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.error("HF_TOKEN not found in .env for diarization.")
                return {"dialogues": []}
                
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Running Pyannote Speaker Diarization pipeline...")
            
            audio_array = whisperx.load_audio(str(global_dialogue_path))
            # Initialize Diarization pipeline
            diarize_model = DiarizationPipeline(token=hf_token, device=device)
            diarize_segments = diarize_model(audio_array)
            
            # 4. Merge words with speakers
            dialogues = []
            current_speaker = None
            current_dialogue = None
            
            for word_obj in words:
                w_start = float(word_obj.get("start", 0))
                w_end = float(word_obj.get("end", 0))
                w_text = word_obj.get("word", word_obj.get("text", "")).strip()
                
                best_speaker = "UNKNOWN"
                max_overlap = 0
                for _, row in diarize_segments.iterrows():
                    s_start = row['start']
                    s_end = row['end']
                    
                    overlap = max(0, min(w_end, s_end) - max(w_start, s_start))
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_speaker = row['speaker']
                        
                if current_speaker != best_speaker or current_dialogue is None:
                    if current_dialogue is not None:
                        dialogues.append(current_dialogue)
                        
                    current_speaker = best_speaker
                    current_dialogue = {
                        "speaker": current_speaker,
                        "start": round(w_start, 2),
                        "end": round(w_end, 2),
                        "text": w_text
                    }
                else:
                    current_dialogue["end"] = round(w_end, 2)
                    current_dialogue["text"] += f" {w_text}"
                    
                    if w_text and w_text[-1] in ['.', '!', '?']:
                        dialogues.append(current_dialogue)
                        current_dialogue = None

            if current_dialogue is not None:
                dialogues.append(current_dialogue)
                
            result_json = {"dialogues": dialogues}
            
            out_path = self.output_base_dir / "dialogue_diarization.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Successfully saved diarization to {out_path}")
            
            # Formatter Step
            try:
                formatter = CinematicScriptFormatter(str(self.output_base_dir))
                fmt_res = formatter.process(str(out_path), str(full_audio_path))
                if fmt_res.get("status") == "success":
                    result_json["script_txt"] = fmt_res.get("script_txt")
                    result_json["script_json"] = fmt_res.get("script_json")
            except Exception as e:
                logger.error(f"Script formatting failed: {e}")
                
            return result_json
            
        except ImportError:
            logger.error("whisperx not installed. Cannot perform diarization.")
            return {"dialogues": []}
        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            return {"dialogues": []}
