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
The backend operates through a structured sequence of API-driven stages:

1. **Ingestion (`/api/v1/ingest`)**: Standardize and prepare video assets.
2. **Segmentation (`/api/v1/segment`)**: Cinematic boundary detection (visual cuts and audio pauses).
3. **Audio Processing (`/api/v1/process-audio`)**: Stem isolation and audio energy calculation.
4. **Speech Intelligence (`/api/v1/process-speech`)**: High-accuracy STT, speaker diarization, and hallucination filtering.
5. **Face Intelligence (`/api/v1/process-faces`)**: Frame-by-frame deep face and emotion analysis.
6. **CLIP Embeddings (`/api/v1/process-clip-embeddings`)**: Visual semantic extraction.
7. **Story & Scene Analysis (`/api/v1/gemini-llm`)**: Generates structured narrative scenes and comprehensive plot mapping using Gemini.
8. **Microdrama Generation (`/api/v1/generate-microdrama`)**: Proposes a chronological list of plot-driven episodes.
9. **Duration Enforcement (`/api/v1/generate-microdrama2`)**: Automatically validates and splits episodes exceeding the 120s limit.
10. **Renderer (`/api/v1/render-microdrama`)**: Final physical reel generation.

## Documentation
For complete details on all backend endpoints, request payloads, and schema structures, refer to the API Reference.

## Technology Stack
* **Backend**: FastAPI (Python)
* **AI Models**: Gemini LLM (Vertex AI SDK), Sarvam AI STT, HTDemucs, InsightFace, HSEmotion, CLIP
* **Video Processing**: MoviePy, FFmpeg
* **Data Handling**: JSON-driven metadata storage (`scenes_and_plot.json`, `dialogue_diarization.json`, etc.)

---

## Detailed Phase Breakdown

### Phase 1: Preparation & Foundation
**1. Ingestion API (`/ingest`)**
Accepts a raw video upload, generates a unique video ID, extracts basic metadata, and standardizes the video format (H.264, AAC, 1080p, 30fps) so the AI models have a clean asset to work with.

**2. Segmentation API (`/segment`)**
Runs the cinematic segmentation pipeline on the standardized video. It detects visual cuts and audio pauses to slice the movie into logical cinematic "scenes" rather than random chunks.

### Phase 2: Multimodal Feature Extraction
**3. Audio Processing API (`/process-audio`)**
Takes the video and isolates the different audio layers. It extracts the raw audio, denoises it, separates it into distinct stems (dialogue track, background music track) using HTDemucs, and computes quantitative volume metrics.

**4. Speech Intelligence API (`/process-speech`)**
Takes the isolated dialogue track and transcribes it into text (optimized for Telugu via Sarvam AI). It performs robust speaker diarization and includes specialized filtering logic to eliminate "hallucinated" English advertising text from the transcripts.

**5. Face Intelligence API (`/process-faces`)**
Analyzes video frames to detect faces, calculates if the frame is a "close-up", and extracts facial expressions to build an emotional timeline for the scene using InsightFace and HSEmotion.

**6. CLIP Embeddings API (`/process-clip-embeddings`)**
Extracts mid-frames from cinematic shots and calculates CLIP semantic embeddings to understand the visual content and context.

### Phase 3: AI Scoring & Narrative Generation
**7. Gemini LLM Intelligence API (`/gemini-llm`)**
The "brain" of the operation. It aggregates multimodal data—specifically shot timestamps, audio energy, diarized transcripts, and visual semantic markers—and leverages Gemini to generate a structured JSON output of narrative scenes and a comprehensive movie plot.

**8. Microdrama Generation API (`/generate-microdrama`)**
Consumes the `scenes_and_plot.json` and `dialogue_diarization.json` to extract a comprehensive set of narrative-rich clips. It shifts from selecting sparse highlights to generating a thorough, chronological list of all essential plot points, explicitly excluding side-content.

**9. Duration Enforcement API (`/generate-microdrama2`)**
A programmatic enforcement mechanism that recalculates episode durations, automatically splits any candidates exceeding 120 seconds into appropriately sized chunks, and filters out clips shorter than 25 seconds to ensure compliance with the target 30-120 second duration rule.

### Phase 4: Final Assembly
**10. Renderer API (`/render-microdrama`)**
The final physical step. It consumes the validated candidate episodes, resolves any invisible video processing defects, crops the video horizontally into a vertical 9:16 aspect ratio, and exports the final, titled cinematic reels.