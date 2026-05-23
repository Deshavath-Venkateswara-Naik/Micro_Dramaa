import os
import json
import logging
from pathlib import Path
import cv2

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.ocr_dir = self.output_base_dir / "ocr"
        self.ocr_dir.mkdir(parents=True, exist_ok=True)
        
        self.paddle_ocr = None
        self.easy_ocr = None
        
        if PADDLE_AVAILABLE:
            logger.info("Initializing PaddleOCR (English)...")
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        else:
            logger.warning("PaddleOCR not available.")
            
    def _get_easyocr(self):
        if not EASYOCR_AVAILABLE:
            return None
        if not self.easy_ocr:
            logger.info("Initializing EasyOCR fallback...")
            self.easy_ocr = easyocr.Reader(['en'])
        return self.easy_ocr

    def process_frames(self, scene_id: str, scene_metadata: dict) -> list:
        frames_dir = self.output_base_dir / "frames" / scene_id
        if not frames_dir.exists():
            logger.warning(f"No frames found for {scene_id} at {frames_dir}")
            return []
            
        face_data_path = self.output_base_dir / "faces" / f"face_intelligence_{scene_id}.json"
        
        if face_data_path.exists():
            with open(face_data_path, 'r') as f:
                face_data = json.load(f)
            reaction_timeline = face_data.get("reaction_timeline", [])
        else:
            logger.warning(f"No face intelligence payload found for {scene_id}, scanning frames directly is not fully supported yet.")
            reaction_timeline = []
            
        all_detections = []
        
        for t_data in reaction_timeline:
            timestamp_sec = t_data.get("timestamp_sec", 0.0)
            frame_path = t_data.get("frame_path")
            
            if not frame_path or not os.path.exists(frame_path):
                continue
                
            detections = self._process_single_frame(frame_path, timestamp_sec)
            all_detections.extend(detections)
            
        # Collapse identical text across sequential frames to avoid spamming the JSON
        filtered_detections = self._collapse_duplicate_detections(all_detections)
        
        payload = {
            "scene_id": scene_id,
            "status": "completed",
            "ocr_detections": filtered_detections
        }
        
        out_path = self.ocr_dir / f"ocr_{scene_id}.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=4, ensure_ascii=False)
            
        return filtered_detections

    def _process_single_frame(self, frame_path: str, timestamp: float) -> list:
        img = cv2.imread(frame_path)
        if img is None:
            return []
            
        h, w = img.shape[:2]
        results = []
        
        if self.paddle_ocr:
            try:
                # result is a list of lists: [[[box], (text, confidence)], ...]
                ocr_result = self.paddle_ocr.ocr(frame_path, cls=True)
                if ocr_result and ocr_result[0]:
                    for line in ocr_result[0]:
                        box = line[0]
                        text, conf = line[1]
                        
                        # Calculate center y
                        y_coords = [pt[1] for pt in box]
                        center_y = sum(y_coords) / len(y_coords)
                        
                        # Simple classification heuristic
                        text_type = "ambient"
                        if center_y > h * 0.8:
                            text_type = "subtitle"
                        elif h * 0.3 < center_y < h * 0.7:
                            text_type = "title_card"
                            
                        results.append({
                            "timestamp": timestamp,
                            "text": text,
                            "type": text_type,
                            "confidence": round(float(conf), 2)
                        })
                return results
            except Exception as e:
                logger.error(f"PaddleOCR failed for {frame_path}: {e}")
                
        # Fallback to EasyOCR
        easy_ocr = self._get_easyocr()
        if easy_ocr:
            try:
                ocr_result = easy_ocr.readtext(frame_path)
                for (bbox, text, prob) in ocr_result:
                    y_coords = [pt[1] for pt in bbox]
                    center_y = sum(y_coords) / len(y_coords)
                    
                    text_type = "ambient"
                    if center_y > h * 0.8:
                        text_type = "subtitle"
                    elif h * 0.3 < center_y < h * 0.7:
                        text_type = "title_card"
                        
                    results.append({
                        "timestamp": timestamp,
                        "text": text,
                        "type": text_type,
                        "confidence": round(float(prob), 2)
                    })
            except Exception as e:
                logger.error(f"EasyOCR failed for {frame_path}: {e}")
                
        return results
        
    def _collapse_duplicate_detections(self, detections: list) -> list:
        """Removes consecutive identical text detections to clean up the timeline."""
        if not detections:
            return []
            
        filtered = [detections[0]]
        for d in detections[1:]:
            last_d = filtered[-1]
            if d["text"] == last_d["text"] and abs(d["timestamp"] - last_d["timestamp"]) < 2.0:
                continue # Same text within 2 seconds
            filtered.append(d)
        return filtered

    def process_ocr(self, scene_metadata_path: str) -> list:
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
            logger.info(f"Extracting OCR text for scene {scene_id}...")
            detections = self.process_frames(scene_id, scene)
            intelligence_results.append({
                "scene_id": scene_id,
                "ocr_detections": detections
            })
            
        master_payload = {
            "status": "completed",
            "message": f"Stage 4.6 OCR Analysis completed",
            "output_dir": str(self.output_base_dir),
            "processed_scenes": len(intelligence_results),
            "ocr_results": intelligence_results
        }
        
        master_path = self.ocr_dir / "master_ocr_intelligence.json"
        with open(master_path, 'w', encoding='utf-8') as f:
            json.dump(master_payload, f, indent=4, ensure_ascii=False)
            
        return intelligence_results
