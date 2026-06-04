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
        
    def _transcribe_single_with_sarvam_api(self, audio_path: str, save_name: str = "chunk", language: str = "hi-IN") -> dict:
        """Sends audio directly to Sarvam AI STT and returns the transcript."""
        sarvam_api_key = os.environ.get("SARVAM_API_KEY")
        if not sarvam_api_key:
            logger.error("SARVAM_API_KEY not found in .env")
            return {"transcript": "", "segments": []}
            
        url = "https://api.sarvam.ai/speech-to-text"
        headers = {"api-subscription-key": sarvam_api_key}
        data = {"language_code": language}
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with open(audio_path, 'rb') as f:
                    files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
                    res = requests.post(url, headers=headers, data=data, files=files, timeout=60)
                    
                if res.status_code == 200:
                    result_data = res.json()
                    
                    # Save chunk JSON
                    chunk_json_path = self.output_base_dir / "audio" / "chunks" / f"{save_name}.json"
                    chunk_json_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(chunk_json_path, 'w', encoding='utf-8') as f:
                        json.dump(result_data, f, ensure_ascii=False, indent=2)
                        
                    transcript = result_data.get("transcript", "")
                    return {"transcript": transcript.strip(), "segments": []}
                elif res.status_code == 429:
                    wait_time = (attempt + 1) * 5
                    logger.warning(f"Rate limited by Sarvam API. Retrying in {wait_time}s (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Sarvam API error: {res.status_code} - {res.text}")
                    return {"transcript": "", "segments": []}
                    
            except Exception as e:
                logger.error(f"Failed to call Sarvam API: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return {"transcript": "", "segments": []}
        return {"transcript": "", "segments": []}

    def _split_audio_with_ffmpeg(self, audio_path: str, segment_duration: int = 300) -> list:
        chunk_dir = self.output_base_dir / "audio" / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear existing chunks
        for f in chunk_dir.glob("chunk_*.wav"):
            f.unlink()
            
        import subprocess
        logger.info(f"Splitting {audio_path} into {segment_duration}s chunks...")
        cmd = [
            "ffmpeg", "-y",
            "-i", audio_path,
            "-f", "segment",
            "-segment_time", str(segment_duration),
            "-c", "copy",
            str(chunk_dir / "chunk_%03d.wav")
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        chunks = sorted(list(chunk_dir.glob("chunk_*.wav")))
        return [str(c) for c in chunks]

    def transcribe_with_sarvam_api(self, audio_path: str, shot_id: str, language: str = "hi-IN") -> dict:
        """Transcribe audio strictly using sequential 29s chunks via Sarvam REST API."""
        final_json_path = self.output_base_dir / "audio" / "chunks" / f"{shot_id}_final_merged.json"
        
        # Avoid redundant STT chunking if we already successfully processed this audio
        if final_json_path.exists():
            logger.info(f"Found existing merged STT results at {final_json_path}. Skipping Sarvam API calls.")
            try:
                with open(final_json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Failed to decode existing final_merged.json, running STT again...")

        segment_duration = 29 # 29 seconds to strictly avoid the 30-second duration limit on Sarvam
        try:
            chunks = self._split_audio_with_ffmpeg(audio_path, segment_duration)
        except Exception as e:
            logger.error(f"Failed to split audio: {e}")
            return {"transcript": "", "segments": []}
            
        all_segments = []
        full_transcript = []
        
        logger.info(f"Processing {len(chunks)} chunks (30s each) sequentially via Sarvam AI.")
        for i, chunk_path in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk_path}")
            stt = self._transcribe_single_with_sarvam_api(chunk_path, f"{shot_id}_part{i}", language)
            
            chunk_text = stt.get("transcript", "")
            
            if not chunk_text:
                logger.warning(f"Chunk {i+1} returned empty transcription. Continuing to next chunk.")
            else:
                # Offset timestamps: we create a single block segment for this chunk
                offset = i * segment_duration
                # We do not have word-level timestamps from Sarvam API, so we provide chunk boundaries
                # which WhisperX can use to align later.
                segment_obj = {
                    "start": float(offset),
                    "end": float(offset + segment_duration),
                    "text": chunk_text
                }
                all_segments.append(segment_obj)
                full_transcript.append(chunk_text)
                
            # To avoid hitting strict rate limits on Sarvam (approx 10 requests/minute)
            time.sleep(6)
                
        if not full_transcript:
            logger.error("All chunks failed to return transcription from Sarvam API.")
            return {"transcript": "", "segments": []}
            
        final_result = {
            "transcript": " ".join(full_transcript),
            "segments": all_segments
        }
        
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Saved merged Sarvam API results to {final_json_path}")
        return final_result

    def _transcribe_with_whisperx(self, audio_path: str, language: str = "hi-IN") -> dict:
        """Local fallback for word-level transcription using WhisperX."""
        try:
            import whisperx
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using local WhisperX for transcription on {device}")
            
            # Load model
            model = whisperx.load_model("large-v2", device, compute_type="float16" if device == "cuda" else "int8")
            
            # Transcribe
            audio = whisperx.load_audio(audio_path)
            # Use requested language if possible (extract first two chars for Whisper, e.g. "hi" from "hi-IN")
            whisper_lang = language[:2] if language else "hi"
            result = model.transcribe(audio, batch_size=8, language=whisper_lang)
            
            # Align
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
            words = []
            for segment in result["segments"]:
                for word in segment.get("words", []):
                    if "start" in word and "end" in word:
                        words.append({
                            "word": word["word"],
                            "start": word["start"],
                            "end": word["end"]
                        })
            
            full_text = " ".join([w["word"] for w in words])
            return {"transcript": full_text.strip(), "segments": words}
        except ImportError:
            logger.error("whisperx is not installed. Cannot use local fallback.")
            return {"transcript": "", "segments": []}
        except Exception as e:
            logger.error(f"Local WhisperX transcription failed: {e}")
            return {"transcript": "", "segments": []}

    def transcribe_audio_with_fallback(self, audio_path: str, shot_id: str, language: str = "hi-IN") -> dict:
        """Attempts transcription with Sarvam API, falls back to local WhisperX if it fails."""
        results = self.transcribe_with_sarvam_api(audio_path, shot_id, language)
        if not results or not results.get("segments"):
            logger.warning(f"Sarvam API failed for {shot_id}. Falling back to local WhisperX transcription...")
            results = self._transcribe_with_whisperx(audio_path, language)
        else:
            # Align Sarvam chunk segments using WhisperX to get word-level timestamps
            logger.info("Aligning Sarvam chunk transcripts to get word-level timestamps using WhisperX...")
            try:
                import whisperx
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                audio_array = whisperx.load_audio(audio_path)
                whisper_lang = language[:2] if language else "hi"
                model_a, metadata = whisperx.load_align_model(language_code=whisper_lang, device=device)
                align_result = whisperx.align(results["segments"], model_a, metadata, audio_array, device, return_char_alignments=False)
                
                formatted_words = []
                for segment in align_result["segments"]:
                    for word in segment.get("words", []):
                        if "start" in word and "end" in word:
                            formatted_words.append({
                                "word": word["word"],
                                "start": word["start"],
                                "end": word["end"]
                            })
                results["segments"] = formatted_words
            except Exception as e:
                logger.error(f"WhisperX alignment failed on Sarvam results: {e}. Diarization may be inaccurate.")
                
        return results

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
                
            # 1. Run Demucs to get the clean vocals (dialogue.wav)
            logger.info("Running Demucs to extract global dialogue track...")
            import subprocess
            try:
                # Output to a temporary demucs folder, then move vocals.wav
                tmp_demucs = self.output_base_dir / "audio" / "demucs_tmp"
                cmd = [
                    "demucs",
                    "--two-stems", "vocals",
                    "-n", "htdemucs",
                    "--out", str(tmp_demucs),
                    str(full_audio_path)
                ]
                subprocess.run(cmd, check=True)
                
                # Demucs outputs to: tmp_demucs/htdemucs/full_audio/vocals.wav
                extracted_vocals = tmp_demucs / "htdemucs" / "full_audio" / "vocals.wav"
                if extracted_vocals.exists():
                    subprocess.run(["mv", str(extracted_vocals), str(global_dialogue_path)])
                else:
                    logger.error("Demucs failed to output vocals.wav")
                    # Fallback to full audio if extraction failed
                    global_dialogue_path = full_audio_path
            except Exception as e:
                logger.error(f"Demucs extraction failed: {e}")
                global_dialogue_path = full_audio_path
                
        # 2. Transcribe with Sarvam AI REST API (with local WhisperX fallback)
        logger.info(f"Transcribing global dialogue with Sarvam AI: {global_dialogue_path}")
        stt_results = self.transcribe_audio_with_fallback(str(global_dialogue_path), f"{video_id}_global", language)
        
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
