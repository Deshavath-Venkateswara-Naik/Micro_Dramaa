import os
import json
import logging
from pathlib import Path
from moviepy.editor import VideoFileClip
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)

class ClipProcessor:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "openai/clip-vit-base-patch32"
        self.model = None
        self.processor = None

    def _initialize_model(self):
        if self.model is None:
            logger.info(f"Loading CLIP model '{self.model_name}' on {self.device}...")
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def extract_embeddings(self, video_id: str, video_path: str) -> dict:
        shots_json_path = self.output_base_dir / "shots.json"
        if not shots_json_path.exists():
            error_msg = f"shots.json not found at {shots_json_path}. Please run Stage 2 Segmentation first."
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}

        try:
            with open(shots_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                shots = data.get("shots", []) if isinstance(data, dict) else data
        except Exception as e:
            logger.error(f"Failed to read shots.json: {e}")
            return {"status": "error", "message": str(e)}

        if not shots:
            return {"status": "error", "message": "shots.json is empty."}

        # Initialize model lazily
        self._initialize_model()

        embeddings_data = []

        logger.info(f"Extracting CLIP embeddings for {len(shots)} shots from {video_path}")
        try:
            with VideoFileClip(str(video_path)) as clip:
                duration = clip.duration

                for idx, shot in enumerate(shots):
                    def parse_time(time_str: str) -> float:
                        if not time_str: return 0.0
                        parts = time_str.split(':')
                        if len(parts) == 3:
                            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
                        elif len(parts) == 2:
                            return float(parts[0]) * 60 + float(parts[1])
                        return float(time_str)

                    start_time = parse_time(shot.get("start_time", "0"))
                    end_time = parse_time(shot.get("end_time", str(duration)))

                    
                    # Calculate middle frame time
                    midpoint = (start_time + end_time) / 2.0
                    
                    # Safety check
                    if midpoint > duration:
                        midpoint = duration - 0.1
                    if midpoint < 0:
                        midpoint = 0.0
                        
                    # Extract frame
                    frame_numpy = clip.get_frame(midpoint)
                    frame_pil = Image.fromarray(frame_numpy)
                    
                    # Compute embedding
                    inputs = self.processor(images=frame_pil, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        image_features = self.model.get_image_features(**inputs)
                    
                    # Normalize and convert to list
                    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                    embedding_list = image_features.cpu().numpy().tolist()[0]
                    
                    embeddings_data.append({
                        "shot_id": shot.get("shot_id", f"shot_{idx}"),
                        "midpoint_time": round(midpoint, 3),
                        "embedding": [round(val, 4) for val in embedding_list] # Round to save space
                    })
                    
                    if (idx + 1) % 10 == 0:
                        logger.info(f"Processed {idx + 1}/{len(shots)} shots.")

        except Exception as e:
            logger.error(f"Error during frame extraction or CLIP processing: {e}")
            return {"status": "error", "message": str(e)}

        # Save results
        out_path = self.output_base_dir / "clip_embeddings.json"
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(embeddings_data, f, indent=2)
            logger.info(f"Saved CLIP embeddings to {out_path}")
        except Exception as e:
            logger.error(f"Failed to save embeddings JSON: {e}")
            return {"status": "error", "message": f"Failed to save JSON: {e}"}

        return {
            "status": "completed",
            "message": f"CLIP embeddings successfully extracted for {len(shots)} shots.",
            "output_dir": str(self.output_base_dir),
            "total_shots": len(embeddings_data)
        }
