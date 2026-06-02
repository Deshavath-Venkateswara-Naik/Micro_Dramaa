import os
import json
import logging
import math
from pathlib import Path
from google import genai

logger = logging.getLogger(__name__)

class GeminiLLMProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        
        self.project_id = os.getenv("GCP_PROJECT_ID")
        self.location = os.getenv("GCP_LOCATION")
        
        try:
            self.llm_client = genai.Client(
                vertexai=True, 
                project=self.project_id, 
                location=self.location
            )
        except Exception as e:
            logger.warning(f"Failed to init GenAI client: {e}")
            self.llm_client = None

    def _read_json(self, path: Path) -> dict:
        if path.exists():
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading JSON {path}: {e}")
        else:
            logger.warning(f"File not found: {path}")
        return {}

    @staticmethod
    def _time_to_seconds(t) -> float:
        """Converts 'HH:MM:SS.mmm' or numeric to float seconds."""
        if t is None:
            return 0.0
        if isinstance(t, (int, float)):
            return float(t)
        try:
            parts = str(t).split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
            if len(parts) == 2:
                m, s = parts
                return int(m) * 60 + float(s)
            return float(t)
        except Exception:
            return 0.0

    @staticmethod
    def _cosine_similarity(a: list, b: list) -> float:
        """Computes cosine similarity between two embedding vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _compute_visual_boundaries(self, clip_embeddings: list) -> list:
        """
        Converts raw CLIP embeddings into actionable visual-change signals.
        Returns per-shot adjacent cosine similarity. A LOW similarity means the
        visuals changed sharply (likely a location/scene change = strong boundary).
        Raw 512-d float vectors are NOT sent to the LLM (it cannot read them).
        """
        if not isinstance(clip_embeddings, list) or len(clip_embeddings) < 2:
            return []

        # Sort by shot_id to guarantee chronological adjacency
        try:
            ordered = sorted(clip_embeddings, key=lambda x: x.get("shot_id", 0))
        except Exception:
            ordered = clip_embeddings

        signals = []
        prev = None
        for item in ordered:
            shot_id = item.get("shot_id")
            emb = item.get("embedding", [])
            if prev is None:
                signals.append({
                    "shot_id": shot_id,
                    "visual_similarity_to_prev": None,
                    "visual_change": "scene_start"
                })
            else:
                sim = round(self._cosine_similarity(prev, emb), 3)
                # Lower similarity => bigger visual jump. Thresholds tuned for CLIP ViT-B/32.
                if sim < 0.55:
                    change = "hard_visual_change"
                elif sim < 0.75:
                    change = "moderate_visual_change"
                else:
                    change = "visually_continuous"
                signals.append({
                    "shot_id": shot_id,
                    "visual_similarity_to_prev": sim,
                    "visual_change": change
                })
            prev = emb
        return signals

    def _summarize_energy(self, energy_data: list, shots: list) -> list:
        """
        Aggregates per-second audio energy into per-shot summaries and detects
        silence valleys (which often mark scene transitions) and loud peaks
        (which often mark dramatic beats). Avoids dumping thousands of raw rows.
        """
        if not isinstance(energy_data, list) or not energy_data:
            return []

        shot_list = shots.get("shots", []) if isinstance(shots, dict) else shots
        if not shot_list:
            return []

        summaries = []
        for shot in shot_list:
            s = self._time_to_seconds(shot.get("start_time") or shot.get("start"))
            e = self._time_to_seconds(shot.get("end_time") or shot.get("end"))
            window = [row.get("energy", 0.0) for row in energy_data
                      if s <= row.get("start", -1) < e]
            if not window:
                summaries.append({
                    "shot_id": shot.get("shot_id"),
                    "avg_energy": 0.0, "max_energy": 0.0,
                    "min_energy": 0.0, "audio_profile": "silent"
                })
                continue
            avg_e = round(sum(window) / len(window), 3)
            max_e = round(max(window), 3)
            min_e = round(min(window), 3)
            # Classify the audio behaviour of the shot
            if max_e >= 0.7 and avg_e >= 0.4:
                profile = "loud_intense"          # action / shouting / climax
            elif max_e - min_e >= 0.4:
                profile = "dynamic_swell"          # builds tension -> peak
            elif avg_e < 0.05:
                profile = "near_silence"           # pause / dramatic beat / transition
            else:
                profile = "steady_dialogue"
            summaries.append({
                "shot_id": shot.get("shot_id"),
                "avg_energy": avg_e,
                "max_energy": max_e,
                "min_energy": min_e,
                "audio_profile": profile
            })
        return summaries

    @staticmethod
    def _align_dialogue_to_shots(diarization: dict, shots: list) -> list:
        """
        Maps diarized dialogue lines onto each shot so the LLM can see who is
        speaking inside every shot window, plus speaker-change density.
        """
        shot_list = shots.get("shots", []) if isinstance(shots, dict) else shots
        dialogues = diarization.get("dialogues", []) if isinstance(diarization, dict) else []
        if not shot_list:
            return []

        def t2s(t):
            return GeminiLLMProcessor._time_to_seconds(t)

        aligned = []
        for shot in shot_list:
            s = t2s(shot.get("start_time") or shot.get("start"))
            e = t2s(shot.get("end_time") or shot.get("end"))
            lines = [d for d in dialogues if d.get("start", -1) < e and d.get("end", 0) > s]
            speakers = sorted({d.get("speaker") for d in lines if d.get("speaker")})
            aligned.append({
                "shot_id": shot.get("shot_id"),
                "speakers_present": speakers,
                "speaker_count": len(speakers),
                "line_count": len(lines),
                "is_dialogue_continuation": len(speakers) == 1
            })
        return aligned

    # Tuning knobs for the deterministic merge
    HARD_CUT_SIM = 0.55          # below this = strong location change (split on its own)
    SOFT_CUT_SIM = 0.70          # below this + speaker change = likely new scene
    MIN_SCENE_SECONDS = 8.0      # scenes shorter than this get absorbed into neighbour
    MAX_SCENE_SECONDS = 180.0    # safety cap: a scene this long must split at any visual cut

    def _merge_shots_into_scenes(self, shots, visual_signals, energy_summary,
                                 dialogue_alignment, diarization) -> list:
        """
        DETERMINISTIC scene grouping (no LLM, no cost, no truncation).

        Collapses the (potentially thousands of) raw shots into a manageable list
        of candidate scenes BEFORE any LLM call. A new scene starts when the
        visuals change hard, OR change moderately while different people are
        talking, OR the scene has simply run too long and a cut appears.

        Speaker comparison is done against the PREVIOUS shot's speakers (recent
        context) — NOT the whole accumulated scene — otherwise a long scene
        eventually contains everyone and no change ever looks "new".
        """
        shot_list = shots.get("shots", []) if isinstance(shots, dict) else shots
        if not shot_list:
            return []

        vis_map = {v.get("shot_id"): v for v in visual_signals}
        eng_map = {e.get("shot_id"): e for e in energy_summary}
        dlg_map = {d.get("shot_id"): d for d in dialogue_alignment}
        dialogues = diarization.get("dialogues", []) if isinstance(diarization, dict) else []

        scenes = []
        current = None
        prev_speakers = set()      # speakers of the immediately preceding shot

        def speakers_of(shot_id):
            return set(dlg_map.get(shot_id, {}).get("speakers_present", []) or [])

        for shot in shot_list:
            sid = shot.get("shot_id")
            start = self._time_to_seconds(shot.get("start_time") or shot.get("start"))
            end = self._time_to_seconds(shot.get("end_time") or shot.get("end"))
            sim = vis_map.get(sid, {}).get("visual_similarity_to_prev")
            profile = eng_map.get(sid, {}).get("audio_profile", "")
            spk = speakers_of(sid)

            if current is None:
                current = {"shot_ids": [sid], "start": start, "end": end, "speakers": set(spk)}
                prev_speakers = spk
                continue

            has_cut = sim is not None
            hard_cut = has_cut and sim < self.HARD_CUT_SIM
            soft_cut = has_cut and sim < self.SOFT_CUT_SIM
            # Speaker change measured against the PREVIOUS shot, not the whole scene
            speaker_changed = bool(spk) and bool(prev_speakers) and not (spk & prev_speakers)
            scene_too_long = (end - current["start"]) > self.MAX_SCENE_SECONDS

            start_new = (
                hard_cut                                   # strong visual jump alone
                or (soft_cut and speaker_changed)          # moderate jump + new people
                or (scene_too_long and has_cut)            # overlong scene, cut available
            )

            if start_new:
                scenes.append(current)
                current = {"shot_ids": [sid], "start": start, "end": end, "speakers": set(spk)}
            else:
                current["shot_ids"].append(sid)
                current["end"] = end
                current["speakers"] |= spk

            prev_speakers = spk if spk else prev_speakers

        if current is not None:
            scenes.append(current)

        # Absorb tiny scenes (camera flickers / reaction inserts) into the previous scene
        merged = []
        for sc in scenes:
            dur = sc["end"] - sc["start"]
            if merged and dur < self.MIN_SCENE_SECONDS:
                prev = merged[-1]
                prev["end"] = sc["end"]
                prev["shot_ids"].extend(sc["shot_ids"])
                prev["speakers"] |= sc["speakers"]
            else:
                merged.append(sc)

        # Attach a dialogue excerpt + final formatted fields to each scene
        def fmt(t):
            h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
            return f"{h:02d}:{m:02d}:{s:06.3f}"

        result = []
        for i, sc in enumerate(merged):
            lines = [d for d in dialogues
                     if d.get("start", -1) < sc["end"] and d.get("end", 0) > sc["start"]]
            excerpt = " ".join(d.get("text", "") for d in lines).strip()
            if len(excerpt) > 600:
                excerpt = excerpt[:600] + "..."
            result.append({
                "scene_number": i + 1,
                "start_time": fmt(sc["start"]),
                "end_time": fmt(sc["end"]),
                "included_shot_ids": sc["shot_ids"],
                "characters_present": sorted(sc["speakers"]),
                "dialogue_excerpt": excerpt
            })
        return result

    def _enrich_scenes_with_llm(self, scenes: list, batch_size: int = 40) -> list:
        """
        Adds a 'setting' label and one-line 'description' to each pre-merged scene.
        Runs in small batches so the output NEVER hits the token cap. The LLM only
        labels what we already segmented — it does NOT decide boundaries, so it
        cannot over-segment or truncate the scene list.
        """
        if not self.llm_client or not scenes:
            return scenes

        system_prompt = (
            "You are a film script supervisor. For each scene you are given its "
            "characters and a dialogue excerpt. Return a concise 'setting' label "
            "(location + time-of-day if inferable, else 'Unknown') and a one-sentence "
            "'description' of the dramatic event. Respond ONLY with a JSON object "
            "mapping each scene_number (as a string) to {\"setting\":..., \"description\":...}."
        )

        for i in range(0, len(scenes), batch_size):
            batch = scenes[i:i + batch_size]
            compact = [{
                "scene_number": s["scene_number"],
                "characters_present": s["characters_present"],
                "dialogue_excerpt": s["dialogue_excerpt"]
            } for s in batch]
            user_prompt = (
                "Label these scenes. Return JSON keyed by scene_number.\n"
                + json.dumps(compact, ensure_ascii=False, separators=(',', ':'))
            )
            try:
                resp = self.llm_client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=user_prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                labels = json.loads(resp.text.strip())
                for s in batch:
                    info = labels.get(str(s["scene_number"]), {})
                    s["setting"] = info.get("setting", "Unknown")
                    s["description"] = info.get("description", "")
            except Exception as e:
                logger.warning(f"Scene labeling batch {i//batch_size} failed: {e}")
                for s in batch:
                    s.setdefault("setting", "Unknown")
                    s.setdefault("description", "")
        return scenes

    def _generate_plot(self, scenes: list) -> str:
        """One cheap text-only call to summarize the whole film from scene labels."""
        if not self.llm_client or not scenes:
            return ""
        # Include timestamps so the LLM understands pacing and time frames.
        outline = [{"scene": s["scene_number"],
                    "start_time": s.get("start_time", ""),
                    "end_time": s.get("end_time", ""),
                    "setting": s.get("setting", ""),
                    "summary": s.get("description", "")} for s in scenes]
        try:
            resp = self.llm_client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=("Write a highly detailed, dramatic, and structured plot summary of this film based on the "
                          "ordered scene outline provided. \n\n"
                          "CRITICAL INSTRUCTION: The plot of the movie is very important. Based on this plot, we are generating micro-dramas. "
                          "Therefore, you MUST provide the plot clearly, focusing entirely on the main movie or serial storyline. "
                          "You MUST explicitly ignore and filter out any unnecessary scenes, advertisements, title cards, and credits. "
                          "Your summary must highlight the narrative structure and emotional peaks. Format your response with the following sections:\n\n"
                          "1. CORE NARRATIVE ARC: The overarching story from beginning to end, clearly outlining the main plot without any filler.\n"
                          "2. MAJOR DRAMATIC BEATS: Bullet points of the most intense conflicts, betrayals, revelations, or emotional peaks.\n"
                          "3. CHARACTER DYNAMICS: Key relationships, motivations, and emotional stakes for the main characters.\n"
                          "4. PEAK MICRODRAMA ZONES: Identify specific scenes or sequences (e.g., 'Scenes 12-15') that have the highest potential for suspenseful, viral, or emotionally gripping microdramas.\n\n"
                          "Analyzing the scene time frames is very important to understand pacing. "
                          "Respond with clearly formatted markdown.\n\n"
                          + json.dumps(outline, ensure_ascii=False, separators=(',', ':'))),
                config=genai.types.GenerateContentConfig(temperature=0.3)
            )
            return resp.text.strip()
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")
            return ""

    def process_gemini_llm(self, video_id: str) -> dict:
        if not self.llm_client:
            logger.error("GenAI client not initialized.")
            return {"error": "GenAI client not initialized"}
            
        video_dir = self.output_base_dir
        
        # Read the 4 required JSON files
        shots_path = video_dir / "shots.json"
        energy_path = video_dir / "full_audio_intelligence.json"
        diarization_path = video_dir / "dialogue_diarization.json"
        clip_embeddings_path = video_dir / "clip_embeddings.json"
        
        shots_data = self._read_json(shots_path)
        energy_data = self._read_json(energy_path)
        diarization_data = self._read_json(diarization_path)
        clip_embeddings = self._read_json(clip_embeddings_path)

        # --- DERIVED SIGNALS (the accuracy upgrade) ---
        # Instead of discarding the CLIP embeddings, convert them into adjacent
        # visual-similarity scores. Instead of dumping raw per-second energy,
        # summarize it per shot. Align dialogue + speakers to each shot.
        visual_signals = self._compute_visual_boundaries(
            clip_embeddings if isinstance(clip_embeddings, list) else []
        )
        energy_summary = self._summarize_energy(
            energy_data if isinstance(energy_data, list) else [], shots_data
        )
        dialogue_alignment = self._align_dialogue_to_shots(diarization_data, shots_data)

        # --- STEP 1: DETERMINISTIC MERGE (no LLM) ---
        # Collapse thousands of raw shots into a manageable list of candidate
        # scenes using the fused signals. This is exact, free, and — crucially —
        # cannot truncate or explode into "one scene per shot".
        scenes = self._merge_shots_into_scenes(
            shots_data, visual_signals, energy_summary,
            dialogue_alignment, diarization_data
        )
        logger.info(f"[{video_id}] Deterministic merge produced {len(scenes)} candidate scenes "
                    f"from {len(visual_signals)} shots.")

        if not scenes:
            result_json = {"error": "No shots available to segment into scenes."}
            out_path = video_dir / "scenes_and_plot.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
            return {"video_id": video_id, "status": "failed", "scenes_and_plot": result_json}

        # --- STEP 2: LABEL SCENES (batched, text-only, cannot truncate) ---
        logger.info(f"[{video_id}] Labeling {len(scenes)} scenes via Gemini in batches...")
        scenes = self._enrich_scenes_with_llm(scenes)

        # --- STEP 3: PLOT SUMMARY (single cheap text-only call) ---
        plot = self._generate_plot(scenes)

        # Strip the internal dialogue_excerpt before persisting (keep output clean)
        for s in scenes:
            s.pop("dialogue_excerpt", None)

        result_json = {
            "total_scenes": len(scenes),
            "Scenes": scenes,
            "plot_of_the_movie": plot
        }

        # Save to storage
        out_path = video_dir / "scenes_and_plot.json"
        try:
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(result_json, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to write scenes_and_plot.json: {e}")

        return {
            "video_id": video_id,
            "status": "completed" if "error" not in result_json else "failed",
            "scenes_and_plot": result_json
        }
