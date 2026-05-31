# Micro-Drama Cinematic Intelligence Pipeline

## Overview
Micro-Drama is a comprehensive, multi-stage AI-driven cinematic intelligence pipeline designed to automate the generation of highly engaging, 30-90 second vertical (9:16) reels from long-form video content. By leveraging advanced multimodal AI—including speech recognition, facial emotion detection, audio processing, and Google's Gemini 2.0 Flash—the system intelligently extracts, sequences, and renders cinematic moments that maximize audience engagement.

## Key Features
* **Automated Video Ingestion & Segmentation**: Standardizes raw videos and precisely cuts them into cinematic boundaries.
* **Multimodal Intelligence Layer**:
  * **Audio Processing**: Splits audio stems using HTDemucs.
  * **Speech Intelligence**: Transcribes dialogue (including Telugu) with parallel chunked Sarvam AI STT and filters hallucinations.
  * **Face & Emotion Intelligence**: Extracts frames and analyzes facial expressions using DeepFace (RetinaFace backend) to compute emotional curves.
  * **BGM & Music Mapping**: Correlates isolated BGM tracks with emotional context.
* **Story & Narrative Engine**: Fuses multimodal signals using Gemini 2.0 Flash (Vertex AI SDK) to compute indicators like "Hero Elevation", "Mother Sentiment", and virality potential.
* **Episodic Sequencer**: Assembles selected story candidates into cohesive, binge-worthy sequences with strict chronological enforcement.
* **Automated Rendering**: Outputs accurately titled, perfectly cropped cinematic reels using MoviePy.

## Pipeline Stages
The backend operates through a structured sequence of API-driven stages:

1. **Ingestion (`/api/v1/ingest`)**: Standardize and prepare video assets.
2. **Segmentation (`/api/v1/segment`)**: Cinematic boundary detection.
3. **Audio Processing (`/api/v1/process-audio`)**: Stem isolation and extraction.
4. **Speech Intelligence (`/api/v1/process-speech`)**: High-accuracy STT and NLP.
5. **Face Intelligence (`/api/v1/process-faces`)**: Frame-by-frame deep face analysis.
6. **Emotion Intelligence (`/api/v1/emotion/process`)**: Cross-modal emotional mapping.
7. **BGM Processing (`/api/v1/bgm/process`)**: Musical sentiment analysis.
8. **Virality Scoring (`/api/v1/virality/process`)**: Prediction of social media performance and dopamine triggers.
9. **Nostalgia Intelligence (`/api/v1/nostalgia/process`)**: Specialized scene scoring.
10. **Drama Engine (`/api/v1/drama/process`)**: Multi-layer narrative scoring.
11. **Story Intelligence (`/api/v1/process-story`)**: Candidate generation using Gemini 2.5 Pro / 2.0 Flash.
12. **Episodic Sequencing (`/api/v1/sequencer/process`)**: Narrative orchestration.
13. **Renderer (`/api/v1/renderer/process`)**: Final reel generation.

## Documentation
For complete details on all backend endpoints, request payloads, and schema structures, refer to the [API Reference](API_REFERENCE.md).

## Technology Stack
* **Backend**: FastAPI (Python)
* **AI Models**: Gemini 2.0 Flash (Vertex AI), Sarvam AI STT, HTDemucs, DeepFace
* **Video Processing**: MoviePy
* **Data Handling**: JSON-driven metadata storage (`master_series_sequence.json`, etc.)


Phase 1: Preparation & Foundation
1. Ingestion API (/ingest)
What it does: Accepts a raw video upload, generates a unique video ID, extracts basic metadata, and standardizes the video format (H.264, AAC, 1080p, 30fps) so the AI models have a clean asset to work with.
Underlying Tech: FFmpeg / MoviePy
2. Segmentation API (/segment)
What it does: Runs the cinematic segmentation pipeline on the standardized video. It detects visual cuts and audio pauses to slice the movie into logical cinematic "scenes" rather than random chunks.
Underlying Tech/Models: PySceneDetect (for visual shot boundaries) and Librosa (for audio silence isolation).
Phase 2: Multimodal Feature Extraction
3. Audio Processing API (/process-audio)
What it does: Takes the video and isolates the different audio layers. It extracts the raw audio, denoises it, and separates it into distinct stems (dialogue track, background music track, sound effects).
Underlying Tech/Models: HTDemucs (a deep learning model for high-fidelity audio stem separation).
4. Speech Intelligence API (/process-speech)
What it does: Takes the isolated dialogue track and transcribes it into text (highly optimized for Telugu). It also runs a denoising filter to remove STT "hallucinations" or repeating words.
Underlying Tech/Models: Sarvam AI STT API (specialized Speech-to-Text).
5. Face & Emotion Intelligence (via Fusion/Processors)
What it does: Analyzes video frames to detect faces, calculates if the frame is a "close-up" (>15% screen area), and extracts facial expressions to build an emotional timeline for the scene.
Underlying Tech/Models: DeepFace using the RetinaFace backend detector.
6. BGM Processing API (/bgm/process)
What it does: Analyzes the isolated Background Music (BGM) tracks to map cinematic music curves (e.g., finding the acoustic intensity or emotional crescendo of a scene).
Phase 3: AI Scoring & Narrative Generation
7. Multimodal Fusion API (/process-fusion)
What it does: A preparation step that gathers all the extracted data (Speech text, Face emotions, BGM intensity) and fuses it into a single, comprehensive metadata payload for each scene.
8. Story Intelligence API (/process-story)
What it does: The "brain" of the operation. It sends the video chunks and the fused multimodal data to an LLM. The AI watches the scene, evaluates cinematic tropes (like "Hero Elevation" or "Mother Sentiment"), scores its virality, and identifies low-energy filler segments (boring_part_time) to enforce a strict 30-100 second duration for micro-dramas.
Underlying Tech/Models: Google Gemini 2.0 Flash / 2.5 Pro (via Vertex AI SDK).
Phase 4: Final Assembly
9. Episodic Sequencer API (/sequencer/process)
What it does: Takes the "story candidates" approved by Gemini and orchestrates them into a cohesive, chronological episodic sequence. It maps out the exact timestamps for the final cuts and outputs a master_series_sequence.json instruction file, ensuring no AI timestamp hallucinations occur.
10. Renderer API (/renderer/process)
What it does: The final physical step. It consumes the master_series_sequence.json file, cuts out the boring parts, crops the video horizontally into a vertical 9:16 aspect ratio, and exports the final, titled cinematic reels.
Underlying Tech/Models: MoviePy (Python automated video rendering).