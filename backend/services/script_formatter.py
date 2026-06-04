import os
import json
import logging
from pathlib import Path
import librosa
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed.")

try:
    from panns_inference import SoundEventDetection, labels as panns_labels
    PANNS_AVAILABLE = True
except ImportError:
    PANNS_AVAILABLE = False
    logger.warning("panns_inference not installed.")

# Relevant cinematic sound event classes to extract and their detection thresholds
CINEMATIC_EVENTS = {
    "Music": 0.55, # Lowered from 0.85 to allow more music detection
    "Door": 0.15,
    "Laughter": 0.2,
    "Crying, sobbing": 0.2,
    "Applause": 0.3,
    "Explosion": 0.2, 
    "Cheering": 0.3, 
    "Gunshot, gunfire": 0.2, 
    "Siren": 0.3, 
    "Bell": 0.2, 
    "Telephone": 0.2, 
    "Screaming": 0.2,
    "Thunder": 0.3, 
    "Wind": 0.4, 
    "Rain": 0.5
}

class CinematicScriptFormatter:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Wav2Vec2 for Emotion
        if TRANSFORMERS_AVAILABLE:
            try:
                logger.info("Initializing Wav2Vec2 Emotion model in CinematicScriptFormatter...")
                self.emotion_model = pipeline(
                    "audio-classification", 
                    model="superb/wav2vec2-base-superb-er", 
                    device=0 if self.device == "cuda" else -1
                )
            except Exception as e:
                logger.error(f"Failed to load Emotion model: {e}")
                self.emotion_model = None
        else:
            self.emotion_model = None

        # Load PANNs
        if PANNS_AVAILABLE:
            try:
                logger.info("Initializing PANNs Sound Event Detection model...")
                self.sed_model = SoundEventDetection(checkpoint_path=None, device=self.device)
            except Exception as e:
                logger.error(f"Failed to initialize PANNs: {e}")
                self.sed_model = None
        else:
            self.sed_model = None

    def detect_sound_events(self, audio_path: str) -> list:
        if not self.sed_model:
            return []
            
        logger.info(f"Running PANNs SED on {audio_path}")
        try:
            # PANNs expects 32kHz
            audio, _ = librosa.load(audio_path, sr=32000)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1) # to mono
                
            audio = audio[None, :] # batch dimension
            framewise_output = self.sed_model.inference(audio)
            
            # framewise_output shape: (1, time_steps, classes_num)
            framewise_output = framewise_output[0]
            
            events = []
            frames_per_second = 100.0 # PANNs typically outputs at 100fps
            
            # Find indices for cinematic events
            cinematic_indices = {}
            for i, label in enumerate(panns_labels):
                if label in CINEMATIC_EVENTS:
                    cinematic_indices[i] = label
            
            # Peak picking / thresholding for events
            current_events = {} # label_idx -> start_time
            
            for t_idx in range(framewise_output.shape[0]):
                time_sec = t_idx / frames_per_second
                
                for c_idx, label in cinematic_indices.items():
                    prob = framewise_output[t_idx, c_idx]
                    threshold = CINEMATIC_EVENTS[label]
                    
                    if prob > threshold and c_idx not in current_events:
                        # Event starts
                        current_events[c_idx] = time_sec
                    elif prob < (threshold - 0.1) and c_idx in current_events:
                        # Event ends
                        start_time = current_events.pop(c_idx)
                        end_time = time_sec
                        
                        # Only keep events longer than 0.2s to reduce noise
                        if end_time - start_time >= 0.2:
                            events.append({
                                "label": label,
                                "start": round(start_time, 2),
                                "end": round(end_time, 2)
                            })
                            
            # Add remaining open events
            for c_idx, start_time in current_events.items():
                end_time = framewise_output.shape[0] / frames_per_second
                if end_time - start_time >= 0.2:
                    events.append({
                        "label": cinematic_indices[c_idx],
                        "start": round(start_time, 2),
                        "end": round(end_time, 2)
                    })
            
            # Filter overlapping same-label events or extremely close ones
            merged_events = []
            events.sort(key=lambda x: x["start"])
            for e in events:
                if not merged_events:
                    merged_events.append(e)
                    continue
                last_e = merged_events[-1]
                # If it's the exact same sound happening within 10 seconds of the last one, merge them
                if e["label"] == last_e["label"] and (e["start"] - last_e["end"]) < 10.0:
                    last_e["end"] = max(last_e["end"], e["end"])
                else:
                    merged_events.append(e)
                    
            return merged_events
        except Exception as e:
            logger.error(f"SED failed: {e}")
            return []

    def extract_emotions(self, dialogues: list, full_audio_path: str) -> list:
        if not self.emotion_model or not dialogues:
            return dialogues
            
        logger.info(f"Extracting emotions for {len(dialogues)} dialogues using Wav2Vec2...")
        try:
            # Load the audio full track
            y, sr = librosa.load(full_audio_path, sr=16000) # wav2vec2 wants 16kHz
            
            for d in dialogues:
                start_sample = min(int(d["start"] * sr), len(y))
                end_sample = min(int(d["end"] * sr), len(y))
                
                segment = y[start_sample:end_sample]
                
                # Make sure the segment is at least 0.1s; pad to 1s to prevent CNN kernel errors
                if len(segment) < int(0.1 * sr):
                    d["emotion"] = "neutral"
                    continue
                    
                if len(segment) < sr:
                    pad_len = sr - len(segment)
                    segment = np.pad(segment, (0, pad_len), 'constant')
                
                # Convert back to a temporary audio format or run directly?
                # Pipeline expects raw waveform as float32 array if passing numpy
                try:
                    res = self.emotion_model(segment)
                    # output: [{'score': 0.8, 'label': 'neu'}, ...]
                    label_map = {"neu": "neutral", "ang": "angry", "hap": "happy", "sad": "sad"}
                    # The emotion is typically the one with highest score
                    top_pred = max(res, key=lambda x: x["score"])
                    emotion = label_map.get(top_pred["label"], top_pred["label"])
                    
                    # For script formatting, we don't necessarily want to annotate 'neutral'
                    d["emotion"] = emotion
                except Exception as ex:
                    logger.warning(f"Failed emotion on segment: {ex}")
                    d["emotion"] = "neutral"
                    
            return dialogues
        except Exception as e:
            logger.error(f"Emotion extraction failed: {e}")
            return dialogues

    def format_time(self, seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def generate_script(self, dialogues: list, sound_events: list) -> str:
        script_lines = []
        
        # We need a unified timeline
        # Timeline items: {"type": "event", "start": X, "data": ...}
        timeline = []
        for d in dialogues:
            timeline.append({
                "type": "dialogue",
                "start": d["start"],
                "data": d
            })
            
        for e in sound_events:
            # For sound events, we just place a tag at the start time
            timeline.append({
                "type": "sound_event",
                "start": e["start"],
                "data": e
            })
            
        timeline.sort(key=lambda x: x["start"])
        
        for item in timeline:
            if item["type"] == "sound_event":
                e = item["data"]
                time_str = self.format_time(e["start"])
                label = e["label"]
                script_lines.append(f"{time_str} [{label.capitalize()}]\n")
            elif item["type"] == "dialogue":
                d = item["data"]
                time_str = self.format_time(d["start"])
                speaker = d.get("speaker", "UNKNOWN")
                emotion = d.get("emotion", "neutral")
                
                # Format: Speaker_A (angry):
                if emotion and emotion != "neutral":
                    script_lines.append(f"{time_str}\n{speaker} ({emotion}):\n{d['text']}\n")
                else:
                    script_lines.append(f"{time_str}\n{speaker}:\n{d['text']}\n")
                    
        return "\n".join(script_lines)

    def process(self, dialogue_diarization_path: str, full_audio_path: str) -> dict:
        """End-to-end processing to create the final transcript script."""
        try:
            with open(dialogue_diarization_path, 'r') as f:
                data = json.load(f)
            dialogues = data.get("dialogues", [])
        except Exception as e:
            logger.error(f"Failed to load diarization: {e}")
            return {"status": "error", "message": "Diarization not found"}
            
        # 1. Extract emotions
        dialogues = self.extract_emotions(dialogues, full_audio_path)
        
        # 2. Extract sound events
        sound_events = self.detect_sound_events(full_audio_path)
        
        # 3. Generate the script text
        script_text = self.generate_script(dialogues, sound_events)
        
        # Save output
        script_txt_path = self.output_base_dir / "script_transcript.txt"
        with open(script_txt_path, 'w', encoding='utf-8') as f:
            f.write(script_text)
            
        # Save enriched JSON
        script_json_path = self.output_base_dir / "script_transcript.json"
        with open(script_json_path, 'w', encoding='utf-8') as f:
            json.dump({
                "dialogues": dialogues,
                "sound_events": sound_events
            }, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Successfully generated Cinematic Script at {script_txt_path}")
        return {
            "status": "success",
            "script_txt": str(script_txt_path),
            "script_json": str(script_json_path)
        }
