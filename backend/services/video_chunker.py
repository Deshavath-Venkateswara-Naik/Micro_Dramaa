import os
import subprocess
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class VideoChunker:
    def __init__(self, output_base_dir: str):
        self.output_base_dir = Path(output_base_dir)
        self.chunks_dir = self.output_base_dir / "video_chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_chunks(self, video_path: str, scene_metadata_path: str) -> list:
        video_path = Path(video_path)
        metadata_path = Path(scene_metadata_path)
        
        if not video_path.exists():
            logger.error(f"Video not found: {video_path}")
            return []
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        scenes = metadata.get("scenes", [])
        chunk_paths = []
        
        for scene in scenes:
            scene_id = scene.get("scene_id")
            start_time = scene.get("start")
            end_time = scene.get("end")
            
            if not all([scene_id, start_time, end_time]):
                continue
                
            output_chunk = self.chunks_dir / f"chunk_{scene_id}.mp4"
            
            if output_chunk.exists():
                chunk_paths.append(str(output_chunk))
                continue
                
            # Use stream copy for fast extraction without re-encoding
            cmd = [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-ss", str(start_time),
                "-to", str(end_time),
                "-c:v", "copy",
                "-c:a", "copy",
                str(output_chunk)
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True)
                chunk_paths.append(str(output_chunk))
                logger.info(f"Extracted video chunk for {scene_id}")
            except subprocess.CalledProcessError as e:
                logger.error(f"FFmpeg extraction failed for {scene_id}: {e.stderr.decode()}")
                
        return chunk_paths
