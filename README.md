# Micro-Drama Cinematic Intelligence Pipeline

## Overview
Micro-Drama is a comprehensive, multi-stage AI-driven cinematic intelligence pipeline designed to automate the generation of highly engaging, 30-120 second vertical (9:16) reels from long-form video content. By leveraging advanced multimodal AI—including speech recognition, facial emotion detection, audio processing, and Google's Gemini LLM—the system intelligently extracts, sequences, and renders cinematic moments that maximize narrative coverage and audience engagement.

## Key Features
* **Automated Video Ingestion & Segmentation**: Standardizes raw videos and precisely cuts them into cinematic boundaries.
* **Multimodal Intelligence Layer**:
  * **Audio Processing**: Splits audio stems using HTDemucs and computes per-second volume energy.
  * **Speech Intelligence**: Transcribes dialogue (including Telugu) with parallel chunked Sarvam AI STT, performs speaker diarization, and strictly filters hallucinated English text.
  * **Face & Emotion Intelligence**: Extracts frames and analyzes facial expressions using InsightFace and HSEmotion to compute emotional curves.
  * **Visual Semantics**: Generates CLIP embeddings for extracted mid-frames to understand scene context.
* **Story & Narrative Engine**: Fuses multimodal signals using Gemini LLM to map a comprehensive movie plot and segment logical narrative scenes.
* **Episodic Sequencer**: Generates a continuous, chronological sequence of 30-120 second plot-driven episodes, programmatically enforcing duration limits and splitting overly long candidates.
* **Automated Rendering**: Outputs accurately titled, perfectly cropped cinematic reels (vertical 9:16 format) using ffmpeg/MoviePy.

## Pipeline Stages
The backend operates through a structured sequence of API-driven stages, which can be fully automated using the `run_pipeline.py` orchestration script:

1. **Ingestion (`/api/v1/ingest/url`)**: Downloads and standardizes the video.
2. **Segmentation (`/api/v1/segment`)**: Cinematic boundary detection (visual cuts).
3. **Audio Processing (`/api/v1/process-audio`)**: Global audio energy and volume peak calculation.
4. **Sound Events (`/api/v1/extract-sound-events`)**: Detects cinematic background noises (e.g., door slams, gunshots).
5. **CLIP Embeddings (`/api/v1/process-clip-embeddings`)**: Visual semantic extraction for cut continuity.
6. **Speech Intelligence (`/api/v1/process-speech`)**: Global STT, speaker diarization, and LLM-driven emotion mapping.
7. **Data Fusion & Plotting (`/api/v1/gemini-llm`)**: Fuses multimodal signals to segment logical narrative scenes and writes the movie plot.
8. **Microdrama Generation (`/api/v1/generate-microdrama`)**: Proposes a chronological list of plot-driven episodes under 100 seconds.
9. **Duration Enforcement (`/api/v1/generate-microdrama2`)**: Automatically validates and splits any proposed episodes exceeding the 120s limit.
10. **Renderer (`/api/v1/render-microdrama`)**: Final physical reel generation.

## Documentation
For complete details on all backend endpoints, request payloads, and schema structures, refer to the API Reference.

## Technology Stack
* **Backend**: FastAPI (Python)
* **AI Models**: Gemini 2.5 Flash/Flash-Lite, Sarvam AI STT, YAMNet (Sound Events), CLIP
* **Video Processing**: PySceneDetect, FFmpeg
* **Data Handling**: JSON-driven metadata storage (`shots.json`, `scenes_and_plot.json`, `script_transcript.json`, etc.)

---

## Detailed Phase Breakdown

### Phase 1: Ingestion & Baseline
**1. Ingestion API (`/api/v1/ingest/url`)**
Accepts a raw video URL (e.g., YouTube), generates a unique video ID, extracts basic metadata, and standardizes the video format (H.264, AAC, 1080p, 30fps) so the AI models have a clean asset to work with.

**2. Segmentation API (`/api/v1/segment`)**
Runs PySceneDetect over the standardized video to detect camera cuts and outputs the exact timestamps of every shot into `shots.json`. This provides the baseline timeline for all future mapping.

### Phase 2: Parallel Feature Extraction
**3. Audio Processing API (`/api/v1/process-audio`)**
Analyzes the global audio track to calculate volume peaks, silences, and energy profiles, outputting to `full_audio_intelligence.json`.

**4. Sound Events API (`/api/v1/extract-sound-events`)**
Uses YAMNet to detect background noises (door slams, gunshots, laughter) in the global audio track, outputting to `sound_events.json`. This gives context to non-verbal storytelling.

**5. CLIP Embeddings API (`/api/v1/process-clip-embeddings`)**
Extracts middle frames from every shot and runs them through the CLIP model to generate semantic vectors (`clip_embeddings.json`). This tells the system how visually different two consecutive shots are (e.g., a hard location change vs. a reverse angle).

**6. Speech Intelligence API (`/api/v1/process-speech`)**
Sends the full audio to Sarvam AI for highly accurate transcription and diarization. It then asks an LLM to assign an emotion to each line of dialogue, outputting a screenplay-like `script_transcript.json`.

### Phase 3: The "Brain" (Data Fusion)
**7. Gemini LLM Intelligence API (`/api/v1/gemini-llm`)**
The core fusion engine. It aggregates `shots.json`, `clip_embeddings.json`, `full_audio_intelligence.json`, `sound_events.json`, and `script_transcript.json`. It deterministically merges shots into scenes, asks Gemini to label them, and leverages Gemini to write a comprehensive movie plot based solely on the extracted metadata (`scenes_and_plot.json`).

### Phase 4: Assembly & Rendering
**8. Microdrama Generation API (`/api/v1/generate-microdrama`)**
Consumes the generated scenes and plot. The LLM selects highly dramatic, viral sequences that are between 30 to 100 seconds long, outputting to `microdrama_candidates.json`.

**9. Duration Enforcement API (`/api/v1/generate-microdrama2`)**
A programmatic safety pass. It scans the candidate micro-dramas and enforces the strict 120-second limit. If the LLM accidentally proposed a clip that is mathematically too long, this script automatically splits it into appropriately sized chunks.

**10. Renderer API (`/api/v1/render-microdrama`)**
The final physical step. It consumes the validated candidate episodes and uses FFmpeg to physically cut the original MP4 into the final, ready-to-post vertical micro-drama clips, saving them into the video's render folder.