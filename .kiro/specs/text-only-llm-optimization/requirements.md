# Requirements Document

## Introduction

This feature optimizes the Micro_Drama backend pipeline to eliminate all multimodal (image/video) inputs to the Gemini LLM, replacing them with structured text-only data. The current `story_engine.py` extracts one keyframe image per shot and sends hundreds of images to Gemini for microdrama candidate extraction — the most expensive operation in the pipeline. Additionally, `multimodal_fusion.py` and `emotion_processor.py` each make one LLM call per shot (potentially 50-200 calls per video). This optimization consolidates all LLM interactions into a maximum of 2-3 text-only calls per video, leveraging the rich structured data already extracted by local processors (CLIP embeddings, face emotions, audio energy, BGM analysis, diarization, transcript).

## Glossary

- **Pipeline**: The Micro_Drama backend processing system that ingests a video and produces microdrama candidates
- **Story_Engine**: The `story_engine.py` service responsible for identifying microdrama candidates from processed video data
- **Fusion_Processor**: The `multimodal_fusion.py` service that scores each shot for dialogue impact, cinematic BGM, virality, and nostalgia
- **Emotion_Processor**: The `emotion_processor.py` service that scores each shot for hero elevation, mother sentiment, and virality reaction
- **LLM_Orchestrator**: The new unified text-only LLM calling component that replaces the three separate LLM stages
- **Pre_Scoring_Engine**: A new local computation module that scores segments for drama potential before LLM invocation
- **Gemini_LLM**: Google's Gemini language model accessed via the Vertex AI GenAI SDK (model: gemini-2.5-flash-lite)
- **Structured_Context**: The combined JSON/text representation of all locally-extracted signals (shots, transcript, face emotions, audio energy, BGM, CLIP similarity scores)
- **CLIP_Similarity_Score**: The cosine similarity between CLIP embeddings of adjacent shots, indicating visual scene continuity or change
- **Keyframe**: A single representative image frame extracted from the midpoint of each shot (currently sent to Gemini)
- **Microdrama_Candidate**: A 30-90 second video segment identified as having high emotional impact, hook potential, and cliffhanger ending
- **Story_Candidates_Output**: The `story_candidates.json` file consumed by downstream pipeline stages

## Requirements

### Requirement 1: Eliminate Image and Video Upload to LLM

**User Story:** As a pipeline operator, I want the Story_Engine to stop sending keyframe images to Gemini, so that I reduce API costs and processing latency.

#### Acceptance Criteria

1. THE Story_Engine SHALL send only text and JSON data to the Gemini_LLM for microdrama candidate extraction.
2. WHEN processing a video, THE Story_Engine SHALL NOT include any `Part.from_bytes` image data, video bytes, or binary media in the Gemini API request contents.
3. WHEN processing a video, THE Story_Engine SHALL NOT extract keyframe images from the video file for LLM submission purposes.
4. THE Story_Engine SHALL remove all dependencies on `moviepy` frame extraction for LLM input construction.

### Requirement 2: Text-Only LLM Input Enforcement

**User Story:** As a pipeline operator, I want all LLM calls across the pipeline to use only structured text data, so that costs remain predictable and low.

#### Acceptance Criteria

1. THE LLM_Orchestrator SHALL accept only JSON-serialized structured data as input to all Gemini API calls.
2. THE Fusion_Processor SHALL NOT make any direct Gemini API calls after this optimization is applied.
3. THE Emotion_Processor SHALL NOT make any direct Gemini API calls after this optimization is applied.
4. WHEN constructing LLM request contents, THE LLM_Orchestrator SHALL validate that no binary or base64-encoded media is included in the payload.

### Requirement 3: Enhanced Structured Context Assembly

**User Story:** As a pipeline operator, I want the LLM to receive richer text-based context about each shot, so that microdrama extraction accuracy improves without needing visual input.

#### Acceptance Criteria

1. THE LLM_Orchestrator SHALL include the following structured data in the text context sent to Gemini_LLM:
   - Shot list with start/end timestamps and shot types
   - Full transcript with speaker diarization and word-level timestamps
   - Face emotion timeline per shot (emotion labels, closeup flags, timestamps)
   - Audio energy peaks with timestamps and intensity values
   - BGM intelligence per shot (bgm_type, intensity scores, transition points)
   - CLIP_Similarity_Score between each pair of adjacent shots
   - Dialogue density metric per shot (words per second)
2. WHEN CLIP embeddings are available, THE LLM_Orchestrator SHALL compute cosine similarity between adjacent shot embeddings and include the resulting scores in the structured context.
3. WHEN face emotion data is available for a shot, THE LLM_Orchestrator SHALL include the condensed emotion timeline (timestamp, emotion label, closeup flag) in the structured context for that shot.
4. THE LLM_Orchestrator SHALL format all structured data as compact JSON with minimal whitespace to maximize token efficiency.

### Requirement 4: Improved Prompt Design for Higher Accuracy

**User Story:** As a pipeline operator, I want the LLM prompts to leverage all structured signals for precise timestamp selection, so that microdrama boundaries are more accurate than the current image-based approach.

#### Acceptance Criteria

1. THE LLM_Orchestrator SHALL instruct Gemini_LLM to use transcript timestamps as ground truth for dialogue boundary alignment.
2. THE LLM_Orchestrator SHALL instruct Gemini_LLM to cross-reference face emotion peaks with audio energy peaks to identify dramatic climax points.
3. THE LLM_Orchestrator SHALL instruct Gemini_LLM to use CLIP_Similarity_Score drops (below a threshold) as indicators of visual scene changes.
4. THE LLM_Orchestrator SHALL instruct Gemini_LLM to enforce strict duration constraints (30-90 seconds) by verifying included shot timestamps sum to within the allowed range.
5. THE LLM_Orchestrator SHALL include explicit scoring rubrics in the prompt that reference the structured data fields (fusion scores, emotion scores, BGM intensity) for retention score calculation.

### Requirement 5: Batch LLM Call Consolidation

**User Story:** As a pipeline operator, I want the total number of LLM calls per video reduced from (N_shots × 2 + 1) to a maximum of 3, so that API costs and latency decrease dramatically.

#### Acceptance Criteria

1. WHEN processing a video with N shots, THE LLM_Orchestrator SHALL make a maximum of 3 Gemini API calls total (regardless of the number of shots).
2. THE LLM_Orchestrator SHALL consolidate per-shot fusion scoring, per-shot emotion scoring, and microdrama candidate extraction into a single comprehensive LLM call when the total token count fits within the Gemini context window.
3. IF the total structured context exceeds the Gemini context window limit, THEN THE LLM_Orchestrator SHALL split the video into temporal segments and make one LLM call per segment, with a maximum of 3 segments.
4. THE LLM_Orchestrator SHALL return fusion scores, emotion scores, and microdrama candidates in a single structured JSON response per call.

### Requirement 6: Local Pre-Scoring Engine

**User Story:** As a pipeline operator, I want low-value segments filtered out before the LLM call, so that the LLM focuses on high-potential segments and produces better candidates.

#### Acceptance Criteria

1. WHEN processing a video, THE Pre_Scoring_Engine SHALL compute a local drama score for each shot using the following signals:
   - Face emotion intensity (hero_elevation_score, mother_sentiment_score from raw face data)
   - Audio energy peak magnitude within the shot timeframe
   - Dialogue density (words per second from transcript)
   - BGM intensity score
2. THE Pre_Scoring_Engine SHALL assign a composite drama score (0-100) to each shot based on a weighted combination of the individual signal scores.
3. THE Pre_Scoring_Engine SHALL mark shots with a composite drama score below 20 as low-value segments.
4. WHEN assembling context for the LLM, THE LLM_Orchestrator SHALL exclude detailed data for low-value segments and include only a summary marker indicating skipped shot ranges.
5. THE Pre_Scoring_Engine SHALL execute all scoring computations locally without any external API calls.

### Requirement 7: Cost Reduction Target

**User Story:** As a pipeline operator, I want at least 80% reduction in Gemini API costs per video, so that the pipeline is economically viable for high-volume processing.

#### Acceptance Criteria

1. THE Pipeline SHALL achieve at least 80% reduction in Gemini API token costs per video compared to the current multimodal approach.
2. WHEN processing a video, THE Pipeline SHALL log the total input token count and output token count for all Gemini calls made during the run.
3. THE Pipeline SHALL NOT send any image tokens to the Gemini API (image tokens currently constitute the majority of cost).

### Requirement 8: Processing Speed Improvement

**User Story:** As a pipeline operator, I want the LLM-dependent stages to complete at least 3x faster, so that end-to-end video processing time decreases.

#### Acceptance Criteria

1. THE Pipeline SHALL complete all LLM-dependent processing stages (fusion, emotion, story extraction) in at most one-third of the time taken by the current multimodal approach for the same video.
2. THE Pipeline SHALL eliminate video file I/O operations (frame extraction, image encoding) from the LLM processing path.
3. WHEN processing a video, THE Pipeline SHALL log the wall-clock duration of the LLM-dependent stages for performance monitoring.

### Requirement 9: Accuracy Maintenance and Improvement

**User Story:** As a pipeline operator, I want microdrama candidate quality to be equal or better than the current approach, so that the optimization does not degrade output quality.

#### Acceptance Criteria

1. THE LLM_Orchestrator SHALL produce microdrama candidates with timestamp boundaries that align within 2 seconds of actual dialogue and scene boundaries (as verified by transcript timestamps).
2. THE LLM_Orchestrator SHALL produce candidates where each candidate contains at least one identified emotional hook (face emotion peak or audio energy peak within the first 5 seconds of the candidate).
3. THE LLM_Orchestrator SHALL produce candidates where each candidate ends with an identified cliffhanger moment (unresolved dialogue, emotion peak, or BGM suspense cue at the final 3 seconds).
4. THE LLM_Orchestrator SHALL produce retention scores that correlate with the density of structured signals (higher fusion scores, emotion peaks, and BGM intensity within the candidate timeframe correspond to higher retention scores).

### Requirement 10: Backward Compatibility of Output Format

**User Story:** As a pipeline operator, I want the output file format to remain unchanged, so that downstream consumers (sequencer, renderer) continue to work without modification.

#### Acceptance Criteria

1. THE Story_Engine SHALL produce a `story_candidates.json` file with the identical JSON schema as the current implementation.
2. THE Story_Engine output SHALL include all existing fields: `video_id`, `status`, and `microdrama_candidates` array.
3. WHEN producing microdrama candidates, THE Story_Engine SHALL include all existing candidate fields: `binge_worthy_title`, `included_shot_ids`, `start_time`, `end_time`, `duration_seconds`, `first_3_second_hook_caption`, `emotional_hook_description`, `central_conflict_type`, `relatable_theme`, `dramatic_peak_timestamp`, `cliffhanger_ending_description`, `retention_score_0_to_100`, `boring_part_time_seconds`, `characters_present`, `virality_and_psychology_analysis`.
4. THE Story_Engine SHALL preserve the existing post-processing logic that enforces the 30-100 second duration constraint and splits oversized episodes into parts.
