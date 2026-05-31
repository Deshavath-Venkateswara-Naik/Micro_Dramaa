import logging
from typing import List, Dict, Any
from scenedetect import detect, ContentDetector
import librosa

logger = logging.getLogger(__name__)

class VisualSegmenter:
    """Handles visual shot boundary detection."""
    
    @staticmethod
    def detect_scenes(video_path: str, threshold: float = 27.0) -> List[Dict[str, float]]:
        """
        Uses PySceneDetect to find shot boundaries (hard cuts and fast dissolves).
        Returns a list of dictionaries with start and end times in seconds.
        """
        try:
            # We use ContentDetector which works well for general cinematic content
            scene_list = detect(video_path, ContentDetector(threshold=threshold))
            
            scenes = []
            for i, scene in enumerate(scene_list):
                scenes.append({
                    "shot_id": i + 1,
                    "start_time": scene[0].get_timecode(),
                    "end_time": scene[1].get_timecode(),
                    "start_frame": scene[0].get_frames(),
                    "end_frame": scene[1].get_frames()
                })
            return scenes
        except Exception as e:
            logger.error(f"PySceneDetect failed: {str(e)}")
            return []

    @staticmethod
    def run_transnetv2_stub(video_path: str) -> List[float]:
        """
        Stub for TransNetV2 deep learning model.
        In a full production environment, this would run inference on a GPU
        to get highly accurate hard cut timestamps.
        """
        logger.info("TransNetV2 stub executed (GPU inference simulated).")
        # Returning empty as we rely on PySceneDetect for this implementation
        return []

class AudioSegmenter:
    """Handles audio silence and intensity detection."""
    
    @staticmethod
    def detect_silences(audio_path: str, top_db: float = 40.0) -> List[Dict[str, float]]:
        """
        Uses Librosa to find non-silent intervals and infer silences (pauses).
        """
        try:
            y, sr = librosa.load(audio_path, sr=16000)
            
            # Get non-silent intervals (in samples)
            non_mute_intervals = librosa.effects.split(y, top_db=top_db)
            
            silences = []
            duration = librosa.get_duration(y=y, sr=sr)
            
            # Calculate silences between non-mute intervals
            last_end = 0.0
            for interval in non_mute_intervals:
                start_sec = interval[0] / sr
                end_sec = interval[1] / sr
                
                if start_sec - last_end > 0.5: # If silence is longer than 0.5 seconds
                    silences.append({
                        "start_time": last_end,
                        "end_time": start_sec,
                        "duration": start_sec - last_end
                    })
                last_end = end_sec
                
            # Check for trailing silence
            if duration - last_end > 0.5:
                silences.append({
                    "start_time": last_end,
                    "end_time": duration,
                    "duration": duration - last_end
                })
                
            return silences
        except Exception as e:
            logger.error(f"Audio processing failed: {str(e)}")
            return []

class SemanticFusionEngine:
    """Fuses visual cuts and audio contexts into Cinematic Boundaries."""
    
    @staticmethod
    def fuse(visual_shots: List[Dict[str, float]], audio_silences: List[Dict[str, float]], video_duration: float = 2.0) -> List[Dict[str, Any]]:
        """
        Evaluates each visual cut. If the visual cut aligns with an audio silence,
        it receives a high 'boundary_score', meaning it's a good place for a scene break.
        """
        scenes = []
        
        if not visual_shots:
            # If no cuts were detected, treat the whole video as a single scene
            duration = video_duration
            if audio_silences and audio_silences[-1]["end_time"] > 0:
                duration = max(duration, audio_silences[-1]["end_time"])
                
            return [{
                "shot_id": "SH_001",
                "start": "00:00:00",
                "end": SemanticFusionEngine._format_time(duration),
                "shot_type": "continuous_action",
                "boundary_score": 1.0,
                "dramatic_pause_detected": False,
                "music_transition": False
            }]
        
        # In a real engine, we'd start at shot 0 and group until we hit a high boundary score.
        # Here we simulate the scene grouping.
        
        current_scene_start = 0.0
        
        for i, shot in enumerate(visual_shots):
            cut_time_val = shot["end_time"]
            if isinstance(cut_time_val, str):
                h, m, s = cut_time_val.split(":")
                cut_time = int(h) * 3600 + int(m) * 60 + float(s)
            else:
                cut_time = float(cut_time_val)
            
            # Check if this cut aligns with any silence
            aligned_silence = None
            for silence in audio_silences:
                # If cut falls within the silence window or very close to it
                if silence["start_time"] - 0.5 <= cut_time <= silence["end_time"] + 0.5:
                    aligned_silence = silence
                    break
                    
            boundary_score = 0.1 # Base score for a simple cut
            dramatic_pause = False
            
            if aligned_silence:
                if aligned_silence["duration"] >= 1.5:
                    boundary_score = 0.9 # Hard cut + Long silence = Definite Scene Boundary
                    dramatic_pause = True
                else:
                    boundary_score = 0.6 # Cut + short breath/pause
            
            # If it's the last shot, force a boundary
            if i == len(visual_shots) - 1:
                boundary_score = 1.0
                
            # If the boundary score is high, we close the current scene
            if boundary_score > 0.5:
                # Format to HH:MM:SS for output
                start_str = SemanticFusionEngine._format_time(current_scene_start)
                end_str = SemanticFusionEngine._format_time(cut_time)
                
                scenes.append({
                    "shot_id": f"SH_{len(scenes) + 1:03d}",
                    "start": start_str,
                    "end": end_str,
                    "shot_type": "emotional_dialogue" if dramatic_pause else "action_dialogue",
                    "boundary_score": round(boundary_score, 2),
                    "dramatic_pause_detected": dramatic_pause,
                    "music_transition": False # Placeholder for advanced music detection
                })
                current_scene_start = cut_time
                
        return scenes

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

class SegmentationService:
    @staticmethod
    def process_video(video_id: str, video_path: str, audio_path: str = None) -> dict:
        """
        Orchestrates the entire Stage 2 pipeline.
        """
        logger.info(f"Starting Stage 2 segmentation for {video_id}")
        
        # 1. Visual Node
        shots = VisualSegmenter.detect_scenes(video_path)
        
        from .storage import StorageService
        StorageService.save_json(video_id, "shots.json", {"shots": shots})
        
        # 2. Audio Node (If audio is not extracted yet, we use the video file and librosa handles it)
        if not audio_path:
            audio_path = video_path
            
        silences = AudioSegmenter.detect_silences(audio_path)
        
        # Get actual video duration for fallback
        try:
            from moviepy.editor import VideoFileClip
            with VideoFileClip(video_path) as clip:
                video_duration = clip.duration
        except Exception as e:
            logger.warning(f"Failed to get video duration: {e}")
            video_duration = 2.0
            
        # 3. Fusion Node
        cinematic_scenes = SemanticFusionEngine.fuse(shots, silences, video_duration)
        
        return {
            "video_id": video_id,
            "total_shots_detected": len(shots),
            "total_cinematic_shots": len(cinematic_scenes),
            "shots": cinematic_scenes
        }
