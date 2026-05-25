import os
import json
import logging
from pathlib import Path
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from google import genai
from hsemotion.facial_emotions import HSEmotionRecognizer

logger = logging.getLogger(__name__)

class FaceIntelligenceProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.face_dir = self.output_base_dir / "faces"
        self.face_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.output_base_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        # We assume the user has GCP_PROJECT_ID and GCP_LOCATION set for Vertex AI
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

        # Initialize InsightFace model
        logger.info("Initializing InsightFace model...")
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']) # Use CPU by default unless CUDA is explicitly configured
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize HSEmotion model
        logger.info("Initializing HSEmotion model...")
        import torch
        original_load = torch.load
        def patched_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_load(*args, **kwargs)
        torch.load = patched_load
        
        self.emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
        
        torch.load = original_load
        
        # Initialize Actor Cache
        self.actor_cache = {}
        self.next_actor_id = 1

    def _calculate_iou(self, boxA, boxB):
        # Determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        # Compute the area of intersection rectangle
        interArea = max(0, xB - xA) * max(0, yB - yA)
        
        # Compute the area of both the prediction and ground-truth rectangles
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        # Compute the intersection over union
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def _match_actor(self, embedding, threshold=0.6):
        best_match = None
        best_score = float('inf') # lower cosine distance is better
        
        for actor_id, cached_emb in self.actor_cache.items():
            score = cosine(embedding, cached_emb)
            if score < best_score:
                best_score = score
                best_match = actor_id
                
        if best_score < threshold:
            return best_match
            
        # Register new actor
        new_actor_id = f"Actor_{self.next_actor_id}"
        self.next_actor_id += 1
        self.actor_cache[new_actor_id] = embedding
        return new_actor_id

    def _extract_and_analyze_frames(self, video_path: str, start_time_sec: float, end_time_sec: float, fps: int = 1, scene_id: str = "unknown") -> list:
        """Extracts frames at a given FPS, saves them as JPEG, and analyzes faces/emotions."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            video_fps = 24.0 # default fallback
            
        start_frame = int(start_time_sec * video_fps)
        end_frame = int(end_time_sec * video_fps)
        
        # Calculate step size based on target extraction fps
        frame_step = int(video_fps / fps) if video_fps > fps else 1
        
        scene_frames_dir = self.frames_dir / scene_id
        scene_frames_dir.mkdir(parents=True, exist_ok=True)
        
        timeline = []
        
        current_frame = start_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        
        # Reset tracking per scene
        face_tracker = {}
        
        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp_sec = current_frame / video_fps
            
            # Save the 1FPS frame to disk for later VLM/OCR processing
            frame_filename = f"frame_{current_frame}.jpg"
            frame_filepath = scene_frames_dir / frame_filename
            cv2.imwrite(str(frame_filepath), frame)
            
            try:
                # Analyze frame with InsightFace
                faces = self.face_app.get(frame)
                
                # frame dimensions to calculate closeup
                h, w, _ = frame.shape
                frame_area = h * w
                
                frame_data = {
                    "timestamp_sec": round(timestamp_sec, 2),
                    "frame_path": str(frame_filepath),
                    "faces": []
                }
                
                current_frame_tracker = {}
                
                for face in faces:
                    if face.det_score < 0.6:
                        continue
                        
                    box = face.bbox # [x1, y1, x2, y2]
                    embedding = face.embedding
                    
                    box_w = max(0, int(box[2] - box[0]))
                    box_h = max(0, int(box[3] - box[1]))
                    face_area = box_w * box_h
                    
                    # If face occupies > 15% of screen, it's a closeup
                    is_closeup = (face_area / frame_area) > 0.15
                    
                    # Tracking: Try to match with previous frame using IoU
                    actor_id = None
                    best_iou = 0
                    best_iou_actor = None
                    for prev_id, prev_box in face_tracker.items():
                        iou = self._calculate_iou(box, prev_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_iou_actor = prev_id
                            
                    if best_iou > 0.5:
                        actor_id = best_iou_actor
                    else:
                        # Recognition: Match embedding
                        actor_id = self._match_actor(embedding)
                        
                    current_frame_tracker[actor_id] = box
                    
                    # Emotion recognition using HSEmotion
                    try:
                        x1 = max(0, int(box[0]))
                        y1 = max(0, int(box[1]))
                        x2 = min(frame.shape[1], int(box[2]))
                        y2 = min(frame.shape[0], int(box[3]))
                        
                        if (y2 > y1) and (x2 > x1):
                            face_img = frame[y1:y2, x1:x2]
                            face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                            emotion, scores = self.emotion_recognizer.predict_emotions(face_img_rgb, logits=False)
                        else:
                            emotion = "neutral"
                    except Exception as e:
                        logger.warning(f"HSEmotion failed: {e}")
                        emotion = "neutral"
                    
                    box_dict = {
                        "x": int(box[0]),
                        "y": int(box[1]),
                        "w": box_w,
                        "h": box_h
                    }
                    
                    frame_data["faces"].append({
                        "actor_id": actor_id,
                        "emotion": emotion,
                        "is_closeup": is_closeup,
                        "box": box_dict
                    })
                    
                face_tracker = current_frame_tracker
                    
                if frame_data["faces"]:
                    timeline.append(frame_data)
                    
            except Exception as e:
                logger.error(f"Error processing frame at {timestamp_sec}s: {e}")
                
            current_frame += frame_step
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            
        cap.release()
        return timeline

    def _calculate_cinematic_impact(self, scene_id: str, timeline: list) -> dict:
        """Returns empty dict since cinematic impact is now handled by Multimodal Fusion."""
        return {}

    def process_faces(self, video_path: str, scene_metadata_path: str) -> list:
        """End-to-end processing for Stage 5."""
        metadata_path = Path(scene_metadata_path)
        if not metadata_path.exists():
            logger.error(f"Metadata not found: {scene_metadata_path}")
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        scenes = metadata.get("scenes", [])
        intelligence_results = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            
            # Convert start/end strings "HH:MM:SS" to seconds
            def time_to_sec(t_str):
                h, m, s = map(int, t_str.split(':'))
                return h * 3600 + m * 60 + s
                
            start_sec = time_to_sec(scene.get("start", "00:00:00"))
            end_sec = time_to_sec(scene.get("end", "00:00:00"))
            
            if start_sec >= end_sec:
                continue
                
            logger.info(f"Extracting & analyzing faces for {scene_id} ({start_sec}s - {end_sec}s)")
            timeline = self._extract_and_analyze_frames(video_path, start_sec, end_sec, fps=1, scene_id=scene_id)
            
            logger.info(f"Calculating cinematic scores for {scene_id}")
            cinematic_scores = self._calculate_cinematic_impact(scene_id, timeline)
            
            scene_payload = {
                "scene_id": scene_id,
                "status": "completed",
                "cinematic_scores": cinematic_scores,
                "reaction_timeline": timeline
            }
            
            # Save individual scene JSON
            out_path = self.face_dir / f"face_intelligence_{scene_id}.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(scene_payload, f, indent=4, ensure_ascii=False)
                
            intelligence_results.append(scene_payload)
            
        # Save master JSON
        master_payload = {
            "status": "completed",
            "message": f"Stage 5 Face Intelligence completed for {metadata.get('video_id', 'Unknown')}",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "face_intelligence_results": intelligence_results
        }
        
        master_path = self.face_dir / "face_intelligence.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return intelligence_results
