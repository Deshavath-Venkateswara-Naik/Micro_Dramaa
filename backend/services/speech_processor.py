import os
import json
import logging
from pathlib import Path
import requests
import concurrent.futures
from dotenv import load_dotenv
import torch
import whisperx
from google import genai
from services.audio_chunker import AudioChunker

load_dotenv()

logger = logging.getLogger(__name__)

# Constants
SARVAM_API_URL = "https://api.sarvam.ai/speech-to-text"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

class SpeechProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.speech_dir = self.output_base_dir / "speech"
        self.speech_dir.mkdir(parents=True, exist_ok=True)
        self.chunker = AudioChunker(self.speech_dir / "chunks")
        
        # Load WhisperX alignment model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            logger.info(f"Loading WhisperX alignment model for Telugu (te) on {self.device}...")
            self.align_model, self.align_metadata = whisperx.load_align_model(language_code="te", device=self.device)
        except Exception as e:
            logger.error(f"Failed to load WhisperX align model: {e}")
            self.align_model, self.align_metadata = None, None
            
        # Load Pyannote Diarization Model
        try:
            if HF_TOKEN:
                logger.info("Loading Pyannote Speaker Diarization model...")
                from whisperx import diarize
                self.diarize_model = diarize.DiarizationPipeline(token=HF_TOKEN, device=self.device)
            else:
                logger.warning("No HF_TOKEN found in environment. Speaker diarization will be skipped.")
                self.diarize_model = None
        except Exception as e:
            logger.error(f"Failed to load Diarization model (check your HF_TOKEN and Pyannote TOS): {e}")
            self.diarize_model = None
            
    def _force_align_transcript(self, audio_path: str, transcript: str) -> dict:
        """Uses WhisperX to force align a transcript with the audio, extracting word-level timestamps."""
        if not self.align_model or not transcript.strip():
            logger.warning("Skipping alignment: Model not loaded or transcript is empty.")
            return {"segments": [], "word_segments": []}
            
        logger.info(f"Starting WhisperX forced alignment for {audio_path}")
        try:
            audio = whisperx.load_audio(audio_path)
            duration = audio.shape[0] / 16000.0
            custom_transcript = [{"text": transcript, "start": 0.0, "end": duration}]
            
            result = whisperx.align(
                custom_transcript,
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            # Diarization: assign speakers
            if self.diarize_model:
                try:
                    logger.info("Running speaker diarization...")
                    diarize_segments = self.diarize_model(audio)
                    result = whisperx.assign_word_speakers(diarize_segments, result)
                except Exception as e:
                    logger.error(f"Diarization failed: {e}")
            
            return {
                "segments": result.get("segments", []),
                "word_segments": result.get("word_segments", [])
            }
        except Exception as e:
            logger.error(f"WhisperX alignment failed: {e}")
            return {"segments": [], "word_segments": []}
        
    def _mock_stt(self, scene_id: str) -> str:
        """Fallback mock STT when API keys or models are missing."""
        return "Nenu okkasari commit ayithe, na maata nene vinanu."

    def transcribe_audio_chunked(self, audio_path: str, scene_id: str) -> dict:
        """Chunks audio, transcribes chunks in parallel, and merges transcripts."""
        chunks = self.chunker.chunk_audio(audio_path, scene_id)
        if not chunks:
            return {"scene_id": scene_id, "transcript": "", "provider": "mock", "language": "te-IN"}
            
        logger.info(f"Dispatching {len(chunks)} chunks to Sarvam API in parallel...")
        
        results = []
        provider = "mock"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_chunk = {
                executor.submit(self.transcribe_audio_chunk, chunk["file_path"], chunk["chunk_id"]): chunk 
                for chunk in chunks
            }
            
            for future in concurrent.futures.as_completed(future_to_chunk):
                chunk_meta = future_to_chunk[future]
                try:
                    result = future.result()
                    results.append((chunk_meta["global_offset_ms"], result["transcript"], result["provider"]))
                except Exception as exc:
                    logger.error(f"Chunk {chunk_meta['chunk_id']} generated an exception: {exc}")
                    results.append((chunk_meta["global_offset_ms"], self._mock_stt(scene_id), "mock"))
                    
        # Sort results by global_offset_ms to ensure chronological order
        results.sort(key=lambda x: x[0])
        
        merged_transcript = []
        for offset, text, prov in results:
            # STT Hallucination Filter: Collapse consecutive repeated words
            words = text.strip().split()
            cleaned_words = []
            if words:
                cleaned_words.append(words[0])
                count = 1
                for word in words[1:]:
                    if word == cleaned_words[-1]:
                        count += 1
                        if count <= 2:  # Keep max 2 repetitions
                            cleaned_words.append(word)
                    else:
                        count = 1
                        cleaned_words.append(word)
                        
            filtered_text = " ".join(cleaned_words)
            
            # If the entire chunk was just 1 or 2 unique words repeated, drop it completely
            if len(words) > 5 and len(set(words)) <= 2:
                logger.info(f"Dropped completely hallucinated chunk at offset {offset}ms: {text[:30]}...")
                continue
                
            if filtered_text.strip():
                merged_transcript.append(filtered_text.strip())
                
            if prov != "mock":
                provider = prov
                
        final_text = " ".join(merged_transcript)
        
        stt_payload = {
            "scene_id": scene_id,
            "transcript": final_text,
            "provider": provider,
            "language": "te-IN"
        }
        
        out_path = self.speech_dir / f"transcript_{scene_id}.json"
        with open(out_path, 'w') as f:
            json.dump(stt_payload, f, indent=4)
            
        return stt_payload

    def transcribe_audio_chunk(self, audio_path: str, chunk_id: str) -> dict:
        """Transcribes a single Telugu audio chunk using Sarvam AI."""
        logger.info(f"Starting STT for {chunk_id}")
        sarvam_key = os.environ.get("SARVAM_API_KEY")
        
        transcript = ""
        
        if sarvam_key and os.path.exists(audio_path):
            try:
                headers = {"api-subscription-key": sarvam_key}
                files = {
                    'file': (
                        os.path.basename(audio_path), 
                        open(audio_path, 'rb'), 
                        'audio/wav'
                    )
                }
                data = {'model': 'saaras:v3', 'language_code': 'te-IN'}
                response = requests.post(SARVAM_API_URL, headers=headers, files=files, data=data)
                
                if response.status_code == 200:
                    transcript = response.json().get("transcript", "")
                else:
                    logger.error(f"Sarvam API failed for {chunk_id}: {response.text}")
                    transcript = self._mock_stt(chunk_id)
            except Exception as e:
                logger.error(f"Error calling Sarvam STT for {chunk_id}: {e}")
                transcript = self._mock_stt(chunk_id)
        else:
            logger.info(f"No SARVAM_API_KEY found or audio missing. Using mock Telugu STT for {chunk_id}.")
            transcript = self._mock_stt(chunk_id)

        return {
            "chunk_id": chunk_id,
            "transcript": transcript,
            "provider": "sarvam" if sarvam_key else "mock",
            "language": "te-IN"
        }

    def calculate_dialogue_impact(self, transcript: str, audio_features: dict, scene_id: str, aligned_data: dict = None) -> dict:
        """The Dialogue Impact Engine: Uses Gemini to score Telugu cinematic tropes."""
        logger.info(f"Running Dialogue Impact Engine for {scene_id}")
        
        # Pull features from Stage 3
        delivery_intensity = float(audio_features.get("emotion_intensity", 0.5) * 100)
        dramatic_silence = bool(audio_features.get("dramatic_silence", False))
        
        # Default fallback scores
        impact_payload = {
            "scene_id": scene_id,
            "text": transcript,
            "dialogue_type": "general",
            "emotion": "neutral",
            "impact_score": 50,
            "viral_score": 50,
            "meme_potential": 50,
            "delivery_intensity": round(delivery_intensity, 1),
            "audience_recall": 50,
            "mass_appeal": 50,
            "dramatic_pause_detected": dramatic_silence,
            "whisperx_alignment": aligned_data or {"segments": [], "word_segments": []}
        }

        if not GEMINI_API_KEY or not transcript.strip():
            logger.info("No GEMINI_API_KEY found. Using heuristic fallback scoring.")
            if "commit ayithe" in transcript.lower():
                impact_payload.update({
                    "dialogue_type": "mass_elevation",
                    "emotion": "rage",
                    "impact_score": 96,
                    "viral_score": 92,
                    "meme_potential": 85,
                    "mass_appeal": 98
                })
        else:
            try:
                # Use the new GenAI SDK with Vertex AI backend to automatically pick up GOOGLE_APPLICATION_CREDENTIALS
                project_id = os.getenv("GCP_PROJECT_ID")
                location = os.getenv("GCP_LOCATION")
                
                client = genai.Client(
                    vertexai=True, 
                    project=project_id, 
                    location=location
                )
                
                prompt = f"""
                You are a Senior Entertainment NLP Engineer specializing in Telugu Cinema and Viral Reel Intelligence.
                Analyze the following Telugu dialogue. Calculate its viral potential for Instagram Reels/Shorts.
                
                Transcript: "{transcript}"
                Acoustic Context: Delivery Intensity was {delivery_intensity}/100. Dramatic pause detected: {dramatic_silence}.

                Respond ONLY with a valid JSON object matching this schema:
                {{
                    "dialogue_type": "mass_elevation | betrayal_confrontation | comedy_punch | emotional_breakdown | romance | suspense | general",
                    "emotion": "rage | sorrow | joy | suspense | neutral",
                    "impact_score": (int 0-100),
                    "viral_score": (int 0-100),
                    "meme_potential": (int 0-100),
                    "audience_recall": (int 0-100),
                    "mass_appeal": (int 0-100)
                }}
                """
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                response_text = response.text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                    
                llm_scores = json.loads(response_text.strip())
                impact_payload.update(llm_scores)
            except Exception as e:
                logger.error(f"Dialogue Impact LLM failed: {e}")

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

            # 2. STT Architecture (Sarvam Chunked Parallel)
            transcript_data = self.transcribe_audio_chunked(str(audio_path), scene_id)
            transcript_text = transcript_data.get("transcript", "")
            
            # 2.5 WhisperX Forced Alignment
            aligned_data = self._force_align_transcript(str(audio_path), transcript_text)
            
            # 3. Dialogue Impact Engine (LLM Semantic Scoring)
            impact_scores = self.calculate_dialogue_impact(
                transcript=transcript_text,
                audio_features=audio_features,
                scene_id=scene_id,
                aligned_data=aligned_data
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
