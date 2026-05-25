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
