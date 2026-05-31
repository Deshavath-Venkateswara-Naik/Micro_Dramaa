import json
import logging
from pathlib import Path
from moviepy.editor import VideoFileClip
import cv2

# These libraries are installed based on requirements.txt
try:
    from insightface.app import FaceAnalysis
except ImportError:
    pass

try:
    import torch
    from functools import partial
    # Fix PyTorch 2.6 weights_only default behavior for loading legacy models
    torch.load = partial(torch.load, weights_only=False)
    from hsemotion.facial_emotions import HSEmotionRecognizer
except ImportError:
    pass

logger = logging.getLogger(__name__)

class FaceIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.faces_dir = self.output_base_dir / "faces"
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize InsightFace for extremely fast face detection
        try:
            self.face_app = FaceAnalysis(name='buffalo_l')
            self.face_app.prepare(ctx_id=-1, det_size=(640, 640)) # ctx_id=-1 for CPU
        except Exception as e:
            logger.warning(f"Could not load InsightFace: {e}")
            self.face_app = None

        # Initialize HSEmotion for state-of-the-art cinematic emotion recognition
        try:
            self.emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf', device='cpu')
        except Exception as e:
            logger.warning(f"Could not load HSEmotion: {e}")
            self.emotion_recognizer = None

    def _time_to_sec(self, t_str: str) -> float:
        h, m, s = map(float, t_str.split(':'))
        return h * 3600 + m * 60 + s

    def process_faces(self, video_id: str, video_path: str, scene_metadata_path: str) -> list:
        if not self.face_app or not self.emotion_recognizer:
            logger.error("Face or Emotion models not loaded.")
            return []

        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Scene metadata not found at {scene_metadata_path}")
            return []

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        shots = metadata.get("shots", [])
        all_scene_faces = []

        try:
            with VideoFileClip(video_path) as clip:
                frame_w, frame_h = clip.size
                frame_area = frame_w * frame_h

                for shot in shots:
                    shot_id = shot.get("shot_id")
                    start_sec = self._time_to_sec(shot.get("start") or shot.get("start_time", "00:00:00"))
                    end_sec = self._time_to_sec(shot.get("end") or shot.get("end_time", "00:00:00"))
                    
                    if end_sec > clip.duration:
                        end_sec = clip.duration

                    # Scalable Strategy: Sample exactly 3 frames (Start, Mid, End)
                    mid_sec = (start_sec + end_sec) / 2.0
                    sample_times = [start_sec + 0.1, mid_sec, end_sec - 0.1] # slight offset to avoid boundaries
                    
                    scene_face_data = []

                    for t in sample_times:
                        if t >= clip.duration or t < 0:
                            continue
                            
                        # Extract frame using moviepy
                        frame = clip.get_frame(t) # RGB format numpy array
                        # OpenCV/InsightFace expects BGR
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                        # Detect Faces
                        faces = self.face_app.get(frame_bgr)
                        
                        frame_faces_info = []
                        for idx, face in enumerate(faces):
                            # bbox is [x1, y1, x2, y2]
                            box = face.bbox.astype(int)
                            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(frame_w, box[2]), min(frame_h, box[3])
                            face_width = x2 - x1
                            face_height = y2 - y1
                            
                            # Skip invalid boxes
                            if face_width <= 0 or face_height <= 0:
                                continue
                                
                            face_area = face_width * face_height
                            is_closeup = bool((face_area / frame_area) > 0.15) # >15% of screen
                            
                            # Emotion Recognition (HSEmotion)
                            # Crop face
                            face_crop = frame_bgr[y1:y2, x1:x2]
                            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                            
                            try:
                                emotion, scores = self.emotion_recognizer.predict_emotions(face_crop_rgb, logits=False)
                            except Exception:
                                emotion = "neutral"
                                
                            frame_faces_info.append({
                                "actor_id": f"actor_{idx}",
                                "emotion": emotion,
                                "is_closeup": is_closeup
                            })

                        scene_face_data.append({
                            "timestamp": round(t, 2),
                            "faces_detected": len(faces),
                            "face_details": frame_faces_info
                        })

                    # Save raw scene data
                    out_data = {
                        "shot_id": shot_id,
                        "timeline": scene_face_data
                    }
                    out_path = self.faces_dir / f"raw_faces_{shot_id}.json"
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(out_data, f, indent=4)
                        
                    all_scene_faces.append(out_data)

            # Store aggregated face/emotion data in storage root
            aggregated_path = self.output_base_dir / "face_emotion.json"
            with open(aggregated_path, 'w', encoding='utf-8') as f:
                json.dump(all_scene_faces, f, indent=4)

            return all_scene_faces

        except Exception as e:
            logger.error(f"Face Intelligence Engine failed: {e}")
            return []
