# Micro-Drama API Documentation

This document provides a comprehensive reference for all the API endpoints available in the Micro-Drama backend. The backend is built using FastAPI and orchestrates a multi-stage cinematic intelligence pipeline.

## `POST` /api/v1/ingest

**Summary**: Ingest Video

**Description**:
Ingests a raw video file, standardizes it, and extracts metadata.

### Request Body

- **Content-Type**: `multipart/form-data`
  - **Schema**:
    - `video` (string): Video
    - `language` (string): Language
    - `title` (string): Title

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/segment

**Summary**: Segment Video

**Description**:
Triggers the Stage 2 Cinematic Segmentation pipeline for a previously ingested video.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/process-audio

**Summary**: Process Audio

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/process-speech

**Summary**: Process Speech

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/process-faces

**Summary**: Process Faces

**Description**:
Stage 5: Cinematic Face Intelligence Engine
Extracts frames, analyzes facial emotions using DeepFace, and calculates cinematic scores.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/emotion/process

**Summary**: Process Emotion

**Description**:
Stage 6: Emotion Intelligence Engine
Fuses multimodal signals from face, speech, and audio to track cinematic emotional curves.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/bgm/process

**Summary**: Process Bgm

**Description**:
Stage 7: BGM & Music Intelligence Engine
Analyzes isolated BGM tracks and fuses them with Stage 6 emotions to map cinematic music curves.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/virality/process

**Summary**: Process Virality

**Description**:
Stage 9: Virality & Audience Psychology Engine
The Final Intelligence Layer. Predicts social media performance and dopamine triggers.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/nostalgia/process

**Summary**: Process Nostalgia

**Description**:
Stage 10: Run Nostalgia Intelligence Engine on all scenes.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/drama/process

**Summary**: Process Drama

**Description**:
Stage 11: Run Multi-Layer Drama Scoring Engine on all scenes.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path
    - `genre` (string): Genre

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/sequencer/process

**Summary**: Process Sequencer

**Description**:
Layer 4: Run Episodic Sequencing Engine on Story Candidates.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/renderer/process

**Summary**: Process Renderer

**Description**:
Stage 13: Render the final Micro-Drama reel using MoviePy.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `POST` /api/v1/process-story

**Summary**: Process Story

**Description**:
Layer 3: Story Intelligence Engine
Aggregates all Layer 2 signals and generates microdrama candidates using Gemini 2.5 Pro.

### Request Body

- **Content-Type**: `application/json`
  - **Schema**:
    - `video_id` (string): Video Id
    - `video_path` (string): Video Path
    - `scene_metadata_path` (string): Scene Metadata Path

### Responses

- **200**: Successful Response
- **422**: Validation Error

---

## `GET` /

**Summary**: Read Root

### Responses

- **200**: Successful Response

---

